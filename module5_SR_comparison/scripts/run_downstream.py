"""run_downstream.py - 3 method × 2 classifier 다운스트림 평가

입력:
  - A: module5/data/sen1floods11/S2Hand/{stem}_S2Hand.tif (4ch BGRN, 512×512, 10m)
  - B/D: module5/output/sr_outputs/{B|D}/{stem}.tif (4ch BGRN, 2048×2048, 2.5m)
  - 라벨: module5/data/sen1floods11/LabelHand/{stem}_LabelHand.tif (0/1/-1, 10m)

분류기:
  - U-Net : smp.Unet(resnet34, ImageNet, in_channels=4) — 학습 없음, 추론만
  - NDWI  : (G-NIR)/(G+NIR) + Otsu

평가:
  - 10m grid 통일 (B/D 는 2.5m logit/NDWI → 4×4 mean pool → threshold)
  - valid_mask = (label != -1)
  - metric: IoU, F1, Boundary IoU, Small CC IoU

출력:
  - output/flood_masks/{method}/{classifier}/{stem}.tif (10m binary)
  - output/metrics.csv (per-chip × method × classifier × metric)
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from shared.utils.proj_env import PROJ_DATA  # noqa: F401

from module5_SR_comparison.src.flood_eval import (
    load_unet,
    unet_water_mask_at_10m,
    mean_pool_4x4,
)
from module5_SR_comparison.src.ndwi import ndwi_water_mask
from module5_SR_comparison.src.metrics import all_metrics

ALL_METHODS = ["A", "B", "D"]
DEFAULT_METHODS = ["A", "B", "D"]
CLASSIFIERS = ["unet", "ndwi"]


# =====================================================================
# I/O helpers
# =====================================================================

def read_data(path: Path) -> Tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        return src.read(), src.profile.copy()


def read_label(path: Path) -> Tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        data = src.read(1)
        profile = src.profile.copy()
    return data, profile


def write_mask(path: Path, mask_uint8: np.ndarray, ref_profile: dict) -> None:
    profile = ref_profile.copy()
    profile.update(
        count=1,
        dtype="uint8",
        nodata=None,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    profile["height"], profile["width"] = mask_uint8.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask_uint8.astype(np.uint8), 1)


# =====================================================================
# Per-method input resolver
# =====================================================================

def resolve_input(
    method: str,
    stem: str,
    data_dir: Path,
    sr_outputs_dir: Path,
) -> Tuple[Path, float]:
    """method 별 입력 경로 + 해상도(m) 반환."""
    if method == "A":
        return data_dir / "S2Hand" / f"{stem}_S2Hand.tif", 10.0
    if method in ("B", "D"):
        return sr_outputs_dir / method / f"{stem}.tif", 2.5
    raise ValueError(f"unknown method: {method}")


# =====================================================================
# Per-chip evaluation
# =====================================================================

def eval_chip(
    stem: str,
    methods: List[str],
    classifiers: List[str],
    data_dir: Path,
    sr_outputs_dir: Path,
    masks_dir: Path,
    unet_model,
    device: str,
    boundary_dilation: int,
    small_cc_max_area: int,
    overwrite: bool,
) -> List[dict]:
    label_path = data_dir / "LabelHand" / f"{stem}_LabelHand.tif"
    label_10m, label_profile = read_label(label_path)
    valid_mask = (label_10m != -1)
    label_bin = (label_10m == 1).astype(np.uint8)

    rows: List[dict] = []
    region = stem.split("_", 1)[0] if "_" in stem else stem

    for method in methods:
        input_path, res_m = resolve_input(method, stem, data_dir, sr_outputs_dir)
        if not input_path.exists():
            print(f"  {stem} [{method}] MISSING input: {input_path}")
            for clf in classifiers:
                rows.append({
                    "chip_id": stem, "region": region, "method": method,
                    "classifier": clf, "iou": "", "f1": "",
                    "boundary_iou": "", "small_cc_iou": "",
                    "status": "missing_input",
                })
            continue

        try:
            data, _ = read_data(input_path)
        except Exception as e:
            print(f"  {stem} [{method}] READ FAIL: {e}")
            for clf in classifiers:
                rows.append({
                    "chip_id": stem, "region": region, "method": method,
                    "classifier": clf, "iou": "", "f1": "",
                    "boundary_iou": "", "small_cc_iou": "",
                    "status": "read_fail",
                })
            continue

        # ----- U-Net -----
        if "unet" in classifiers:
            out_path = masks_dir / method / "unet" / f"{stem}.tif"
            try:
                if out_path.exists() and not overwrite:
                    with rasterio.open(out_path) as src:
                        pred_10m = src.read(1)
                else:
                    pred_10m = unet_water_mask_at_10m(
                        unet_model, data, input_resolution_m=res_m, device=device,
                    )
                    write_mask(out_path, pred_10m, label_profile)

                m = all_metrics(pred_10m, label_bin, valid_mask=valid_mask,
                                boundary_dilation=boundary_dilation,
                                small_cc_max_area=small_cc_max_area)
                rows.append({
                    "chip_id": stem, "region": region, "method": method,
                    "classifier": "unet",
                    "iou": m["iou"], "f1": m["f1"],
                    "boundary_iou": m["boundary_iou"], "small_cc_iou": m["small_cc_iou"],
                    "status": "ok",
                })
            except Exception as e:
                print(f"  {stem} [{method}/unet] FAIL: {e}")
                traceback.print_exc(limit=1)
                rows.append({
                    "chip_id": stem, "region": region, "method": method,
                    "classifier": "unet", "iou": "", "f1": "",
                    "boundary_iou": "", "small_cc_iou": "",
                    "status": f"fail:{type(e).__name__}",
                })

        # ----- NDWI -----
        if "ndwi" in classifiers:
            out_path = masks_dir / method / "ndwi" / f"{stem}.tif"
            try:
                if out_path.exists() and not overwrite:
                    with rasterio.open(out_path) as src:
                        ndwi_10m = src.read(1)
                else:
                    ndwi_native, _ = ndwi_water_mask(data)
                    if res_m == 10.0:
                        ndwi_10m = ndwi_native
                    else:
                        # 2.5m binary → 4×4 mean pool → threshold 0.5
                        pooled = mean_pool_4x4(ndwi_native.astype(np.float32))
                        ndwi_10m = (pooled > 0.5).astype(np.uint8)
                    write_mask(out_path, ndwi_10m, label_profile)

                m = all_metrics(ndwi_10m, label_bin, valid_mask=valid_mask,
                                boundary_dilation=boundary_dilation,
                                small_cc_max_area=small_cc_max_area)
                rows.append({
                    "chip_id": stem, "region": region, "method": method,
                    "classifier": "ndwi",
                    "iou": m["iou"], "f1": m["f1"],
                    "boundary_iou": m["boundary_iou"], "small_cc_iou": m["small_cc_iou"],
                    "status": "ok",
                })
            except Exception as e:
                print(f"  {stem} [{method}/ndwi] FAIL: {e}")
                traceback.print_exc(limit=1)
                rows.append({
                    "chip_id": stem, "region": region, "method": method,
                    "classifier": "ndwi", "iou": "", "f1": "",
                    "boundary_iou": "", "small_cc_iou": "",
                    "status": f"fail:{type(e).__name__}",
                })

    return rows


# =====================================================================
# CLI / main
# =====================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    here = Path(__file__).resolve()
    module_root = here.parents[1]
    default_data = module_root / "data" / "sen1floods11"
    default_sr = module_root / "output" / "sr_outputs"
    default_masks = module_root / "output" / "flood_masks"
    default_csv = module_root / "output" / "metrics.csv"

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                        choices=ALL_METHODS)
    parser.add_argument("--classifiers", nargs="+", default=CLASSIFIERS,
                        choices=CLASSIFIERS)
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument("--sr-outputs-dir", type=Path, default=default_sr)
    parser.add_argument("--masks-dir", type=Path, default=default_masks)
    parser.add_argument("--metrics-csv", type=Path, default=default_csv)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-chips", type=int, default=None)
    parser.add_argument("--boundary-dilation", type=int, default=1)
    parser.add_argument("--small-cc-max-area", type=int, default=100)
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
    print(f"[setup] device={device}, methods={args.methods}, classifiers={args.classifiers}")

    label_dir = args.data_dir / "LabelHand"
    labels = sorted(label_dir.glob("*_LabelHand.tif"))
    stems = [p.name.replace("_LabelHand.tif", "") for p in labels]
    if args.max_chips:
        stems = stems[: args.max_chips]
    if not stems:
        print(f"  ERROR: no labels under {label_dir}")
        return 1
    print(f"[setup] {len(stems)} chips")

    unet_model = None
    if "unet" in args.classifiers:
        print(f"[setup] loading U-Net (smp.Unet resnet34 + ImageNet)")
        unet_model = load_unet(device=device, in_channels=4)

    all_rows: List[dict] = []
    t0 = time.time()
    for i, stem in enumerate(stems, 1):
        rows = eval_chip(
            stem=stem,
            methods=args.methods,
            classifiers=args.classifiers,
            data_dir=args.data_dir,
            sr_outputs_dir=args.sr_outputs_dir,
            masks_dir=args.masks_dir,
            unet_model=unet_model,
            device=device,
            boundary_dilation=args.boundary_dilation,
            small_cc_max_area=args.small_cc_max_area,
            overwrite=args.overwrite,
        )
        all_rows.extend(rows)
        if i % 5 == 0 or i == len(stems):
            dt = time.time() - t0
            print(f"  [{i}/{len(stems)}] {stem} done ({dt:.1f}s elapsed)")

    args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["chip_id", "region", "method", "classifier",
                  "iou", "f1", "boundary_iou", "small_cc_iou", "status"]
    with open(args.metrics_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)
    print(f"\nMetrics CSV: {args.metrics_csv}")
    print(f"  rows: {len(all_rows)}")

    print("\n[aggregate] method × classifier 평균 IoU / F1 / boundary_IoU:")
    agg: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    for r in all_rows:
        if r["status"] != "ok":
            continue
        key = (r["method"], r["classifier"])
        d = agg.setdefault(key, {"iou": [], "f1": [], "boundary_iou": [], "small_cc_iou": []})
        for k in d:
            v = r[k]
            if v == "" or v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fv):
                d[k].append(fv)

    print(f"  {'method':>6}  {'clf':>5}  {'n':>4}  {'IoU':>7}  {'F1':>7}  {'B-IoU':>7}  {'SmCC':>7}")
    for (m, c) in sorted(agg.keys()):
        d = agg[(m, c)]
        def _mean(xs): return float(np.mean(xs)) if xs else float("nan")
        print(f"  {m:>6}  {c:>5}  {len(d['iou']):>4d}  "
              f"{_mean(d['iou']):>7.4f}  {_mean(d['f1']):>7.4f}  "
              f"{_mean(d['boundary_iou']):>7.4f}  {_mean(d['small_cc_iou']):>7.4f}")

    n_fail = sum(1 for r in all_rows if r["status"].startswith("fail") or r["status"] == "read_fail")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
