"""run_sr_all.py - 3 비교군 SR 출력 일괄 생성

입력 : module5_SR_comparison/data/sen1floods11/S2Hand/*_S2Hand.tif (4ch BGRN, 512×512, 10m)
출력 :
  A (real S2 10m)        : output/sr_outputs/A/*.tif (입력 그대로 복사 ※ default 는 skip)
  B (bicubic 2.5m)       : output/sr_outputs/B/*.tif (4× bicubic upsample, 2048×2048)
  D (Prithvi-SR 2.5m)    : output/sr_outputs/D/*.tif (best_sr_model.pth, tile-and-stitch)

A 는 다운스트림 스크립트가 원본을 직접 읽으므로 기본적으로 미생성. 필요시 --methods A 로 활성화.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import rasterio
from affine import Affine

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from shared.utils.proj_env import PROJ_DATA  # noqa: F401

from module5_SR_comparison.src.sr_inference import bicubic_sr, prithvi_sr

ALL_METHODS = ["A", "B", "D"]
DEFAULT_METHODS = ["B", "D"]


# =====================================================================
# I/O helpers
# =====================================================================

def read_chip(path: Path):
    with rasterio.open(path) as src:
        data = src.read()
        profile = src.profile.copy()
    return data, profile


def write_chip_hr(path: Path, data: np.ndarray, src_profile: dict, scale: int = 4) -> None:
    profile = src_profile.copy()
    src_transform: Affine = profile["transform"]
    new_transform = src_transform * Affine.scale(1.0 / scale, 1.0 / scale)
    profile.update(
        height=data.shape[1],
        width=data.shape[2],
        transform=new_transform,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


def write_chip_native(path: Path, data: np.ndarray, src_profile: dict) -> None:
    profile = src_profile.copy()
    profile.update(compress="lzw", tiled=True, blockxsize=256, blockysize=256)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


# =====================================================================
# Method runners
# =====================================================================

def run_A(lr_data: np.ndarray, _: dict, **_kw) -> np.ndarray:
    return lr_data


def run_B(lr_data: np.ndarray, _: dict, **_kw) -> np.ndarray:
    return bicubic_sr(lr_data, scale=4)


def run_D(
    lr_data: np.ndarray, _: dict, *,
    prithvi_model, device: str,
    tile_lr: int = 224, stride_lr: int = 144, **_kw,
) -> np.ndarray:
    return prithvi_sr(lr_data, model=prithvi_model, device=device,
                      tile_lr=tile_lr, stride_lr=stride_lr)


# =====================================================================
# Main
# =====================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    here = Path(__file__).resolve()
    module_root = here.parents[1]
    default_in = module_root / "data" / "sen1floods11" / "S2Hand"
    default_out = module_root / "output" / "sr_outputs"
    default_prithvi = module_root / "best_sr_model.pth"

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                        choices=ALL_METHODS,
                        help="실행할 method 목록 (A=real-S2 copy, B=bicubic, D=Prithvi-SR)")
    parser.add_argument("--input-dir", type=Path, default=default_in)
    parser.add_argument("--output-dir", type=Path, default=default_out)
    parser.add_argument("--prithvi-ckpt", type=Path, default=default_prithvi)
    parser.add_argument("--device", default=None,
                        help="cuda / cpu (None 이면 cuda 가능 시 자동)")
    parser.add_argument("--max-chips", type=int, default=None,
                        help="처음 N개만 처리 (디버깅용)")
    parser.add_argument("--tile-lr", type=int, default=224)
    parser.add_argument("--stride-lr", type=int, default=144)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def resolve_device(device_arg: Optional[str]) -> str:
    if device_arg:
        return device_arg
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    print(f"[setup] device={device}, methods={args.methods}")

    chips = sorted(args.input_dir.glob("*_S2Hand.tif"))
    if args.max_chips:
        chips = chips[: args.max_chips]
    if not chips:
        print(f"  ERROR: no chips under {args.input_dir}")
        return 1
    print(f"[setup] {len(chips)} chips found")

    prithvi_model = None
    if "D" in args.methods:
        if not args.prithvi_ckpt.exists():
            print(f"  ERROR: Prithvi-SR ckpt missing: {args.prithvi_ckpt}")
            return 1
        from module5_SR_comparison.src.prithvi_sr import load_prithvi_sr_model
        print(f"[setup] loading Prithvi-SR from {args.prithvi_ckpt}")
        prithvi_model = load_prithvi_sr_model(args.prithvi_ckpt, device=device)

    runners = {"A": run_A, "B": run_B, "D": run_D}
    kwargs = dict(
        prithvi_model=prithvi_model,
        device=device,
        tile_lr=args.tile_lr,
        stride_lr=args.stride_lr,
    )

    stats: Dict[str, Dict[str, int]] = {m: {"ok": 0, "skip": 0, "fail": 0} for m in args.methods}
    per_chip_timing: Dict[str, List[float]] = {m: [] for m in args.methods}

    for i, chip_path in enumerate(chips, 1):
        stem = chip_path.name.replace("_S2Hand.tif", "")
        try:
            lr_data, profile = read_chip(chip_path)
        except Exception as e:
            print(f"  [{i}/{len(chips)}] {stem} READ FAIL: {e}")
            for m in args.methods:
                stats[m]["fail"] += 1
            continue

        for m in args.methods:
            out_path = args.output_dir / m / f"{stem}.tif"
            if out_path.exists() and not args.overwrite:
                stats[m]["skip"] += 1
                continue
            try:
                t0 = time.time()
                hr_data = runners[m](lr_data, profile, **kwargs)
                dt = time.time() - t0
                per_chip_timing[m].append(dt)
                if m == "A":
                    write_chip_native(out_path, hr_data, profile)
                else:
                    write_chip_hr(out_path, hr_data, profile, scale=4)
                stats[m]["ok"] += 1
            except Exception as e:
                stats[m]["fail"] += 1
                print(f"  [{i}/{len(chips)}] {stem} method={m} FAIL: {e}")
                traceback.print_exc(limit=2)

        if i % 5 == 0 or i == len(chips):
            print(f"  [{i}/{len(chips)}] {stem} done")

    summary = {
        "n_chips": len(chips),
        "methods": args.methods,
        "device": device,
        "stats": stats,
        "mean_sec_per_chip": {
            m: (float(np.mean(v)) if v else None) for m, v in per_chip_timing.items()
        },
    }
    summary_path = args.output_dir / "_run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    for m in args.methods:
        s = stats[m]
        avg = per_chip_timing[m]
        avg_str = f"{np.mean(avg):.2f}s/chip" if avg else "n/a"
        print(f"  {m}: ok={s['ok']:4d}  skip={s['skip']:4d}  fail={s['fail']:4d}  avg={avg_str}")
    print(f"Summary: {summary_path}")

    total_fail = sum(s["fail"] for s in stats.values())
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
