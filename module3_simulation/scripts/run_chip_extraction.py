"""run_chip_extraction.py - 정사보정 K3A + S2-모사 K3A 페어 칩 추출 배치

흐름:
  1) data/output/mtf_sim/*_mtfsim.json (성공 페어) 탐색
  2) HR: data/interim/grid/{label}_K3A_*_{B|G|R|N}_grid.tif (4밴드 2.5m, 정사보정+공통격자 K3A)
  3) LR: data/output/mtf_sim/{label}_*_simulated_final_2p5m.tif (4밴드, 방사+MTF 모사) → 4×4 mean → 10m
  4) LR 격자에서 stride_lr=150 슬라이딩(=1/3 중첩, overlap=74), HR=LR×4 동기화
  5) HR 칩 valid ratio < 0.8 → 스킵
  6) data/output/chips/{label}_{scene}/ 아래 페어 + 칩별 JSON 사이드카

칩 사양:
  HR: 896×896 @2.5m  (= 2240×2240 m)
  LR: 224×224 @10m   (= 동일 footprint, 4:1 비율)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window, transform as window_transform

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from module3_simulation.src.mtf import aggregate_4x4_mean
from shared.utils.proj_env import PROJ_DATA  # noqa: F401  GDAL/PROJ 환경 초기화

_PATHS_CONFIG = json.loads((PROJECT_ROOT / "config" / "paths.json").read_text(encoding="utf-8"))


K3A_BAND_ORDER = ["Blue", "Green", "Red", "NIR"]
K3A_BAND_SUFFIX = {
    "Blue":  "_B_grid.tif",
    "Green": "_G_grid.tif",
    "Red":   "_R_grid.tif",
    "NIR":   "_N_grid.tif",
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def find_hr_band_files(grid_dir: Path, label: str) -> Dict[str, Path]:
    """interim/grid/ 에서 해당 페어({label}_K3A_*) 의 4밴드 grid.tif 를 찾는다."""
    band_files: Dict[str, Path] = {}
    for tif in grid_dir.glob(f"{label}_K3A_*_grid.tif"):
        # label prefix 가 다른 페어와 부분매치 안 되도록 정확히 "{label}_" 시작 강제
        if not tif.name.startswith(f"{label}_K3A_"):
            continue
        for band, suf in K3A_BAND_SUFFIX.items():
            if tif.name.endswith(suf):
                band_files[band] = tif
                break
    return band_files


def load_hr_stack(band_files: Dict[str, Path]) -> Tuple[np.ndarray, dict]:
    """K3A 4밴드를 (4, H, W) 로 스택. 첫 밴드 profile 도 반환."""
    arrs = []
    profile: dict | None = None
    for band in K3A_BAND_ORDER:
        if band not in band_files:
            raise FileNotFoundError(f"HR 밴드 누락: {band}")
        with rasterio.open(band_files[band]) as src:
            arrs.append(src.read(1))
            if profile is None:
                profile = src.profile.copy()
    return np.stack(arrs, axis=0), profile  # type: ignore[return-value]


def make_lr_profile_from_hr(hr_profile: dict, lr_h: int, lr_w: int) -> dict:
    """HR 2.5m profile → 10m profile (origin·CRS 동일, pixel size 만 4× 확대)."""
    hr_t = hr_profile["transform"]
    lr_t = Affine(10.0, 0.0, hr_t.c, 0.0, -10.0, hr_t.f)
    lr_profile = hr_profile.copy()
    lr_profile.update(transform=lr_t, width=lr_w, height=lr_h)
    return lr_profile


def write_chip_tif(
    out_path: Path, data: np.ndarray, parent_profile: dict, window: Window,
) -> None:
    """parent_profile 의 transform 에서 window 영역을 잘라 chip TIF 저장."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chip_t = window_transform(window, parent_profile["transform"])
    chip_profile = parent_profile.copy()
    chip_profile.update(
        height=int(window.height),
        width=int(window.width),
        transform=chip_t,
        count=data.shape[0],
        dtype=data.dtype.name,
        tiled=True,
        blockxsize=min(256, int(window.width)),
        blockysize=min(256, int(window.height)),
        compress="lzw",
    )
    with rasterio.open(out_path, "w", **chip_profile) as dst:
        dst.write(data)


def _process_pair(
    label: str,
    scene_stem: str,
    mtf_tif_path: Path,
    grid_dir: Path,
    out_root: Path,
    args,
    logger: logging.Logger,
) -> Tuple[int, int]:
    """페어 1개에서 칩들을 추출. (n_saved, n_skipped) 반환."""
    logger.info("\n[페어 %s] %s", label, scene_stem)

    hr_files = find_hr_band_files(grid_dir, label)
    missing = [b for b in K3A_BAND_ORDER if b not in hr_files]
    if missing:
        logger.warning("  HR 밴드 누락 %s → 스킵", missing)
        return 0, 0

    hr_2p5m, hr_profile = load_hr_stack(hr_files)
    with rasterio.open(mtf_tif_path) as src:
        lr_simulated_2p5m = src.read()

    if hr_2p5m.shape != lr_simulated_2p5m.shape:
        logger.warning(
            "  HR shape %s ≠ MTF-sim shape %s → 공통 부분만 사용",
            hr_2p5m.shape, lr_simulated_2p5m.shape,
        )
        h = min(hr_2p5m.shape[1], lr_simulated_2p5m.shape[1])
        w = min(hr_2p5m.shape[2], lr_simulated_2p5m.shape[2])
        hr_2p5m = hr_2p5m[:, :h, :w]
        lr_simulated_2p5m = lr_simulated_2p5m[:, :h, :w]

    # MTF 모사 결과 (2.5m) → 4×4 mean → 10m
    lr_10m = aggregate_4x4_mean(lr_simulated_2p5m)
    h_lr, w_lr = lr_10m.shape[1:]

    # HR 도 4 의 배수로 정합 (LR 와 outer rect 동일하게)
    hr_crop = hr_2p5m[:, : h_lr * 4, : w_lr * 4]

    lr_profile_full = make_lr_profile_from_hr(hr_profile, h_lr, w_lr)

    LR_SIZE = args.lr_size
    HR_SIZE = LR_SIZE * 4
    STRIDE_LR = args.stride_lr
    STRIDE_HR = STRIDE_LR * 4

    if h_lr < LR_SIZE or w_lr < LR_SIZE:
        logger.warning(
            "  LR 격자(%dx%d) 가 칩(%d)보다 작음 → 스킵",
            h_lr, w_lr, LR_SIZE,
        )
        return 0, 0

    out_dir = out_root / f"{label}_{scene_stem}"

    n_saved = 0
    n_skipped = 0

    for y_lr in range(0, h_lr - LR_SIZE + 1, STRIDE_LR):
        for x_lr in range(0, w_lr - LR_SIZE + 1, STRIDE_LR):
            y_hr = y_lr * 4
            x_hr = x_lr * 4

            hr_chip = hr_crop[:, y_hr : y_hr + HR_SIZE, x_hr : x_hr + HR_SIZE]
            lr_chip = lr_10m[:, y_lr : y_lr + LR_SIZE, x_lr : x_lr + LR_SIZE]

            # valid 픽셀: HR 의 4 밴드 모두 nonzero (NoData=0 정책)
            valid_pixels = (hr_chip != 0).all(axis=0)
            valid_ratio = float(valid_pixels.sum()) / float(valid_pixels.size)

            if valid_ratio < args.valid_threshold:
                n_skipped += 1
                continue

            chip_id = f"y{y_hr:05d}_x{x_hr:05d}"
            base = f"{label}_{scene_stem}_{chip_id}"
            hr_path = out_dir / f"{base}_hr.tif"
            lr_path = out_dir / f"{base}_lr.tif"
            meta_path = out_dir / f"{base}_meta.json"

            if (
                hr_path.exists()
                and lr_path.exists()
                and meta_path.exists()
                and not args.overwrite
            ):
                n_skipped += 1
                continue

            hr_window = Window(col_off=x_hr, row_off=y_hr, width=HR_SIZE, height=HR_SIZE)
            lr_window = Window(col_off=x_lr, row_off=y_lr, width=LR_SIZE, height=LR_SIZE)

            write_chip_tif(hr_path, hr_chip, hr_profile, hr_window)
            write_chip_tif(lr_path, lr_chip, lr_profile_full, lr_window)

            hr_t = window_transform(hr_window, hr_profile["transform"])
            lr_t = window_transform(lr_window, lr_profile_full["transform"])
            xmin = hr_t.c
            ymax = hr_t.f
            xmax = xmin + HR_SIZE * hr_t.a
            ymin = ymax + HR_SIZE * hr_t.e  # hr_t.e < 0

            meta = {
                "pair_label": label,
                "scene_stem": scene_stem,
                "chip_id": chip_id,
                "bands": K3A_BAND_ORDER,
                "crs": str(hr_profile.get("crs")),
                "bbox_utm": [xmin, ymin, xmax, ymax],
                "hr": {
                    "size": [HR_SIZE, HR_SIZE],
                    "resolution_m": 2.5,
                    "pixel_offset_yx": [y_hr, x_hr],
                    "transform": list(hr_t)[:6],
                    "source_dir": str(hr_files[K3A_BAND_ORDER[0]].parent),
                },
                "lr": {
                    "size": [LR_SIZE, LR_SIZE],
                    "resolution_m": 10.0,
                    "pixel_offset_yx": [y_lr, x_lr],
                    "transform": list(lr_t)[:6],
                    "source_tif": str(mtf_tif_path),
                    "downsample": "aggregate_4x4_mean (permissive)",
                },
                "n_valid_pixels_hr": int(valid_pixels.sum()),
                "valid_ratio_hr": valid_ratio,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            n_saved += 1

    logger.info(
        "  저장 %d, 스킵 %d  (LR 격자 %dx%d, stride_lr=%d)",
        n_saved, n_skipped, h_lr, w_lr, STRIDE_LR,
    )
    return n_saved, n_skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="K3A HR(2.5m)/LR(10m) 칩 추출 배치")
    parser.add_argument("--mtf-dir", default=None, help="MTF 모사 결과 디렉토리")
    parser.add_argument("--grid-dir", default=None, help="K3A grid 디렉토리")
    parser.add_argument("--out-dir", default=None, help="칩 저장 루트 (기본: output/chips)")
    parser.add_argument("--lr-size", type=int, default=224, help="LR 칩 한 변 (default 224)")
    parser.add_argument(
        "--stride-lr", type=int, default=150,
        help="LR 격자 stride (default 150 = overlap 74/224 ≈ 33%%, 1/3 중첩 한계)",
    )
    parser.add_argument(
        "--valid-threshold", type=float, default=0.8,
        help="HR 칩 valid 비율 하한 (default 0.8)",
    )
    parser.add_argument("--overwrite", action="store_true", help="기존 칩 덮어쓰기")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("chip_extract")

    mtf_dir = (
        Path(args.mtf_dir).resolve()
        if args.mtf_dir
        else (PROJECT_ROOT / _PATHS_CONFIG["output_dir"] / "mtf_sim").resolve()
    )
    grid_dir = (
        Path(args.grid_dir).resolve()
        if args.grid_dir
        else (PROJECT_ROOT / _PATHS_CONFIG["grid_dir"]).resolve()
    )
    out_root = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else (PROJECT_ROOT / _PATHS_CONFIG["output_dir"] / "chips").resolve()
    )
    out_root.mkdir(parents=True, exist_ok=True)

    if not mtf_dir.exists():
        logger.error("MTF 디렉토리를 찾을 수 없음: %s", mtf_dir)
        sys.exit(1)
    if not grid_dir.exists():
        logger.error("grid 디렉토리를 찾을 수 없음: %s", grid_dir)
        sys.exit(1)

    json_files = sorted(mtf_dir.glob("*_mtfsim.json"))
    overlap_pct = (1 - args.stride_lr / args.lr_size) * 100
    logger.info("=" * 70)
    logger.info("칩 추출 시작: %d 페어 탐색", len(json_files))
    logger.info("  HR: %d×%d @2.5m  |  LR: %d×%d @10m",
                args.lr_size * 4, args.lr_size * 4, args.lr_size, args.lr_size)
    logger.info("  stride_lr=%d (overlap %.1f%%) | valid_threshold=%.0f%%",
                args.stride_lr, overlap_pct, args.valid_threshold * 100)
    logger.info("  mtf_dir : %s", mtf_dir)
    logger.info("  grid_dir: %s", grid_dir)
    logger.info("  out_dir : %s", out_root)
    logger.info("=" * 70)

    total_saved = 0
    total_skipped = 0
    n_pairs_processed = 0
    n_pairs_failed = 0

    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                rec = json.load(f)
            if rec.get("status") != "success":
                continue
            label = rec["pair_label"]
            scene_stem = rec["scene_stem"]
            mtf_tif_path = mtf_dir / Path(rec["output_tif"]).name
            if not mtf_tif_path.exists():
                logger.warning("[%s] MTF tif 없음: %s", label, mtf_tif_path)
                n_pairs_failed += 1
                continue

            saved, skipped = _process_pair(
                label, scene_stem, mtf_tif_path, grid_dir, out_root, args, logger,
            )
            total_saved += saved
            total_skipped += skipped
            n_pairs_processed += 1
        except Exception as e:
            logger.exception("[%s] 페어 처리 실패: %s", json_path.stem, e)
            n_pairs_failed += 1

    summary = {
        "n_pairs_processed": n_pairs_processed,
        "n_pairs_failed": n_pairs_failed,
        "total_chips_saved": total_saved,
        "total_chips_skipped": total_skipped,
        "lr_size": args.lr_size,
        "hr_size": args.lr_size * 4,
        "stride_lr": args.stride_lr,
        "stride_hr": args.stride_lr * 4,
        "overlap_ratio": (args.lr_size - args.stride_lr) / args.lr_size,
        "valid_threshold": args.valid_threshold,
        "out_dir": str(out_root),
    }
    with open(out_root / "_chip_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 70)
    logger.info(
        "완료: 페어 처리 %d, 실패 %d, 칩 저장 %d, 스킵 %d",
        n_pairs_processed, n_pairs_failed, total_saved, total_skipped,
    )
    logger.info("summary: %s", out_root / "_chip_summary.json")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
