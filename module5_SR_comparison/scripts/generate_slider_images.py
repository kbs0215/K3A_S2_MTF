"""generate_slider_images.py — 슬라이더용 비교 이미지 자동 생성

final_1877_chips/ 에서 대표 칩을 선정 → best_sr_model.pth 로 SR 추론 →
Bicubic·SR·HR 비교 PNG 를 module5_SR_comparison/images/ 에 자동 생성.

출력:
  images/hero.jpg          — 대표 HR 칩 RGB 전경
  images/s1-original.png   — 슬라이더 1 왼쪽 (Bicubic 4× upsample)
  images/s1-superx.png     — 슬라이더 1 오른쪽 (SR 출력)
  images/s2-original.png   — 슬라이더 2 왼쪽 (LR 10m → bicubic)
  images/s2-superx.png     — 슬라이더 2 오른쪽 (SR 출력)
  images/s3-original.png   — 슬라이더 3 왼쪽 (LR 10m → bicubic)
  images/s3-superx.png     — 슬라이더 3 오른쪽 (SR 출력)
"""

from __future__ import annotations

import os
import sys

# DLL loading path and OpenMP duplication workarounds for Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if sys.platform == "win32":
    for path in [
        r"C:\Users\kbs02\anaconda3\envs\gongjong\Library\bin",
        r"C:\Users\kbs02\anaconda3\envs\gongjong\Lib\site-packages\torch\lib",
    ]:
        if os.path.exists(path):
            try:
                os.add_dll_directory(path)
            except Exception:
                pass

# MUST import torch before any other GDAL/rasterio packages to avoid DLL conflicts
try:
    import torch
except Exception as e:
    print(f"Warning: Failed to import torch: {e}")

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# GDAL/PROJ env init
from shared.utils.proj_env import PROJ_DATA  # noqa: F401

import rasterio


# =====================================================================
# RGB rendering helpers
# =====================================================================

def tif_to_rgb(
    data: np.ndarray,
    band_order: Tuple[int, ...] = (2, 1, 0),
    pct_lo: float = 2.0,
    pct_hi: float = 98.0,
    stretch_params: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """4밴드 BGRN uint16 → RGB uint8 (히스토그램 스트레칭).

    band_order: RGB 로 재배치할 밴드 인덱스 (기본: R=2, G=1, B=0).
    stretch_params: [(lo, hi), ...] 3 밴드 — 외부에서 동일 스트레칭 강제 시 사용.
    반환: (H,W,3) uint8 RGB 이미지 + 적용된 stretch_params.
    """
    rgb = data[list(band_order)].astype(np.float32)  # (3, H, W)

    # 유효 마스크 (모든 밴드 > 0)
    valid = np.all(rgb > 0, axis=0)

    params: List[Tuple[float, float]] = []
    out = np.zeros_like(rgb)

    for i in range(3):
        band = rgb[i]
        if stretch_params is not None:
            lo, hi = stretch_params[i]
        else:
            valid_vals = band[valid]
            if valid_vals.size == 0:
                lo, hi = 0.0, 1.0
            else:
                lo = float(np.percentile(valid_vals, pct_lo))
                hi = float(np.percentile(valid_vals, pct_hi))
            if hi <= lo:
                hi = lo + 1.0
        params.append((lo, hi))
        out[i] = np.clip((band - lo) / (hi - lo), 0, 1)

    # (3, H, W) → (H, W, 3) uint8
    out_u8 = (out * 255).astype(np.uint8).transpose(1, 2, 0)
    # NoData 영역 검은색
    out_u8[~valid] = 0
    return out_u8, params


def save_png(path: Path, img_u8: np.ndarray) -> None:
    """(H,W,3) uint8 numpy → PNG 저장 (matplotlib 사용)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(img_u8.shape[1] / 100, img_u8.shape[0] / 100), dpi=100)
    ax.imshow(img_u8)
    ax.axis("off")
    fig.savefig(str(path), bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}  ({path.stat().st_size // 1024} KB)")


def save_png_raw(path: Path, img_u8: np.ndarray) -> None:
    """PIL 로 직접 저장 (matplotlib 여백 없이)."""
    try:
        from PIL import Image
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img_u8).save(str(path), optimize=True)
        print(f"  Saved: {path}  ({path.stat().st_size // 1024} KB)")
    except ImportError:
        save_png(path, img_u8)


def save_jpg_raw(path: Path, img_u8: np.ndarray, quality: int = 92) -> None:
    """PIL 로 JPEG 저장."""
    try:
        from PIL import Image
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img_u8).save(str(path), quality=quality, optimize=True)
        print(f"  Saved: {path}  ({path.stat().st_size // 1024} KB)")
    except ImportError:
        save_png(path.with_suffix(".png"), img_u8)


# =====================================================================
# Chip selection
# =====================================================================

def select_representative_chips(
    chips_dir: Path,
    n_chips: int = 3,
) -> List[dict]:
    """final_1877_chips/ 에서 대표 칩 선정.

    기준: valid_ratio 높고 + 시각적 다양성 (다른 씬에서 선정).
    반환: [{"hr_tif": Path, "lr_tif": Path, "meta": dict, "scene_dir": str}, ...]
    """
    scene_dirs = sorted([d for d in chips_dir.iterdir() if d.is_dir()])
    print(f"[select] {len(scene_dirs)} scenes found")

    # 씬별 최고 valid_ratio 칩 1개씩 수집
    candidates = []
    for sd in scene_dirs:
        meta_files = sorted(sd.glob("*_meta.json"))
        if not meta_files:
            continue

        best_meta = None
        best_ratio = 0.0
        for mf in meta_files:
            try:
                meta = json.loads(mf.read_text(encoding="utf-8"))
                ratio = meta.get("valid_ratio_hr", 0)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_meta = meta
                    best_meta["_meta_path"] = str(mf)
            except Exception:
                continue

        if best_meta and best_ratio >= 0.95:
            chip_id = best_meta["chip_id"]
            scene_stem = best_meta.get("scene_stem", "")
            pair_label = best_meta.get("pair_label", "")
            base = f"{pair_label}_{scene_stem}_{chip_id}"

            hr_tif = sd / f"{base}_hr.tif"
            lr_tif = sd / f"{base}_lr.tif"

            if hr_tif.exists() and lr_tif.exists():
                candidates.append({
                    "hr_tif": hr_tif,
                    "lr_tif": lr_tif,
                    "meta": best_meta,
                    "scene_dir": sd.name,
                    "valid_ratio": best_ratio,
                })

    # 다양한 씬에서 골고루 선정 (앞/중간/뒤)
    candidates.sort(key=lambda c: c["valid_ratio"], reverse=True)

    # 씬 기반 분산 선정: 서로 다른 날짜의 씬에서
    selected = []
    used_dates = set()
    for c in candidates:
        # scene_dir 예: "10_K3A_20161012043841_08557_00022455_L1R"
        parts = c["scene_dir"].split("_")
        date_str = parts[2] if len(parts) > 2 else ""
        date_key = date_str[:8]  # YYYYMMDD

        if date_key not in used_dates:
            selected.append(c)
            used_dates.add(date_key)
            if len(selected) >= n_chips:
                break

    # 부족하면 나머지에서 채움
    if len(selected) < n_chips:
        for c in candidates:
            if c not in selected:
                selected.append(c)
                if len(selected) >= n_chips:
                    break

    print(f"[select] {len(selected)} chips selected:")
    for s in selected:
        print(f"  - {s['scene_dir']} / {s['meta']['chip_id']}  (valid={s['valid_ratio']:.4f})")

    return selected


# =====================================================================
# SR inference
# =====================================================================

def bicubic_upsample_4x(lr_data: np.ndarray) -> np.ndarray:
    """(4, H, W) uint16 → (4, H*4, W*4) uint16 bicubic 업샘플."""
    from scipy.ndimage import zoom
    C, H, W = lr_data.shape
    result = np.zeros((C, H * 4, W * 4), dtype=np.float32)
    for c in range(C):
        result[c] = zoom(lr_data[c].astype(np.float32), 4.0, order=3)
    return np.clip(result, 0, 65535).astype(np.uint16)


def nearest_upsample_4x(lr_data: np.ndarray) -> np.ndarray:
    """(4, H, W) uint16 → (4, H*4, W*4) uint16 nearest-neighbor 업샘플 (10m 픽셀 블록 구조 유지)."""
    return np.repeat(np.repeat(lr_data, 4, axis=1), 4, axis=2)


def run_sr_inference(
    lr_data: np.ndarray,
    model,
    device: str,
) -> np.ndarray:
    """LR (4, 224, 224) uint16 → SR (4, 896, 896) uint16.

    모델 입력 크기가 224×224 이므로 직접 추론.
    더 큰 칩인 경우 tile-and-stitch 필요.
    """
    from module5_SR_comparison.src.prithvi_sr import prithvi_sr_infer_tile
    return prithvi_sr_infer_tile(model, lr_data, device=device)


def run_sr_inference_tiled(
    lr_data: np.ndarray,
    model,
    device: str,
    tile_lr: int = 224,
    stride_lr: int = 144,
) -> np.ndarray:
    """임의 크기 LR → SR (tile-and-stitch)."""
    from module5_SR_comparison.src.sr_inference import prithvi_sr
    return prithvi_sr(lr_data, model=model, device=device,
                      tile_lr=tile_lr, stride_lr=stride_lr)


# =====================================================================
# Main pipeline
# =====================================================================

def generate_images(
    chips_dir: Path,
    sr_ckpt: Path,
    output_dir: Path,
    n_chips: int = 3,
    device: str = "cpu",
    dry_run: bool = False,
) -> None:
    print("=" * 60)
    print("  슬라이더 이미지 생성 파이프라인")
    print("=" * 60)

    # 1. 칩 선정
    chips = select_representative_chips(chips_dir, n_chips=max(n_chips, 3))
    if len(chips) < 3:
        print(f"ERROR: 최소 3개 칩이 필요하지만 {len(chips)}개만 발견됨")
        return

    if dry_run:
        print("[dry-run] 칩 선정 완료, SR 추론 생략")
        return

    # 2. SR 모델 로드
    print(f"\n[model] best_sr_model.pth 로딩... ({sr_ckpt.stat().st_size // (1024*1024)} MB)")
    from module5_SR_comparison.src.prithvi_sr import load_prithvi_sr_model
    t0 = time.time()
    model = load_prithvi_sr_model(sr_ckpt, device=device)
    print(f"[model] 로드 완료 ({time.time() - t0:.1f}s)")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. 각 칩 처리
    slider_configs = [
        # (slider_id, left_label, right_label, left_source, right_source)
        ("s1", "Bicubic", "SR", "bicubic", "sr"),
        ("s2", "LR_bicubic", "SR", "bicubic", "sr"),
        ("s3", "LR_bicubic", "SR", "bicubic", "sr"),
    ]

    for idx, chip_info in enumerate(chips[:3]):
        slider_id = f"s{idx + 1}"
        hr_tif = chip_info["hr_tif"]
        lr_tif = chip_info["lr_tif"]
        scene = chip_info["scene_dir"]
        chip_id = chip_info["meta"]["chip_id"]

        print(f"\n{'-' * 50}")
        print(f"[{slider_id}] {scene} / {chip_id}")

        # Read TIF data
        with rasterio.open(hr_tif) as src:
            hr_data = src.read()  # (4, 896, 896) uint16
        with rasterio.open(lr_tif) as src:
            lr_data = src.read()  # (4, 224, 224) uint16

        print(f"  HR: {hr_data.shape} {hr_data.dtype}  LR: {lr_data.shape} {lr_data.dtype}")

        # Bicubic upsample
        print("  Bicubic 4x 업샘플...")
        bicubic_data = bicubic_upsample_4x(lr_data)

        # Nearest-neighbor upsample for S2 10m blocks
        print("  Nearest 4x 업샘플 (10m 블록 구조)...")
        nearest_data = nearest_upsample_4x(lr_data)

        # SR inference
        print("  SR 추론 중 (CPU - 수 분 소요)...")
        t0 = time.time()
        sr_data = run_sr_inference(lr_data, model, device=device)
        dt = time.time() - t0
        print(f"  SR 완료 ({dt:.1f}s)  shape={sr_data.shape}")

        # Generate PNG pairs with individual stretching for optimal contrast and dynamic range
        bicubic_rgb, _ = tif_to_rgb(bicubic_data)
        nearest_rgb, _ = tif_to_rgb(nearest_data)
        sr_rgb, _ = tif_to_rgb(sr_data)
        hr_rgb, _ = tif_to_rgb(hr_data)

        # Save slider images (용량 및 페이지 로딩 성능 최적화를 위해 512x512 크기로 다운샘플링)
        from PIL import Image
        
        def resize_rgb(arr: np.ndarray, size: int) -> np.ndarray:
            return np.array(Image.fromarray(arr).resize((size, size), Image.Resampling.LANCZOS))
            
        bicubic_rgb_resized = resize_rgb(bicubic_rgb, 512)
        nearest_rgb_resized = resize_rgb(nearest_rgb, 512)
        sr_rgb_resized = resize_rgb(sr_rgb, 512)
        hr_rgb_resized = resize_rgb(hr_rgb, 512)
        
        if slider_id == "s1":
            save_png_raw(output_dir / f"{slider_id}-original.png", bicubic_rgb_resized)
        else:
            save_png_raw(output_dir / f"{slider_id}-original.png", nearest_rgb_resized)
        save_png_raw(output_dir / f"{slider_id}-superx.png", sr_rgb_resized)

        # First chip → hero image (가로 800px로 다운샘플링 및 최적 압축)
        if idx == 0:
            hero_h = int(hr_rgb.shape[0] * (800 / hr_rgb.shape[1]))
            hero_rgb_resized = np.array(Image.fromarray(hr_rgb).resize((800, hero_h), Image.Resampling.LANCZOS))
            save_jpg_raw(output_dir / "hero.jpg", hero_rgb_resized, quality=85)

        # Save HR reference (for inspection)
        save_png_raw(output_dir / f"{slider_id}-hr-ref.png", hr_rgb_resized)

    print(f"\n{'=' * 60}")
    print(f"  완료! 이미지 -> {output_dir}")
    print(f"{'=' * 60}")


# =====================================================================
# CLI
# =====================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    module_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chips-dir", type=Path,
        default=PROJECT_ROOT / "data" / "output" / "final_1877_chips",
        help="칩 페어 디렉토리",
    )
    parser.add_argument(
        "--sr-ckpt", type=Path,
        default=module_root / "best_sr_model.pth",
        help="SR 모델 체크포인트",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=module_root / "images",
        help="출력 이미지 디렉토리",
    )
    parser.add_argument("--n-chips", type=int, default=3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="칩 선정만 하고 SR 추론 생략")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    device = args.device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    print(f"[config] device={device}")
    print(f"[config] chips_dir={args.chips_dir}")
    print(f"[config] sr_ckpt={args.sr_ckpt}")
    print(f"[config] output_dir={args.output_dir}")

    generate_images(
        chips_dir=args.chips_dir,
        sr_ckpt=args.sr_ckpt,
        output_dir=args.output_dir,
        n_chips=args.n_chips,
        device=device,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
