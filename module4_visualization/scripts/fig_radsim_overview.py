"""fig_radsim_overview.py - 모사 종합 figure (3 행 x 3 열).

Row 1 (전체):  (a) K3A 2.5m | (b) Sim 10m | (c) S2 10m
Row 2 (확대):  (d) K3A zoom | (e) Sim zoom | (f) S2 zoom
Row 3 (히스토): (g) K3A 10m | (h) Sim 10m | (i) S2 10m  — 4밴드(B,G,R,NIR) 오버레이

Sim 은 **rad + MTF 가 적용된 최종 모사 영상을 4×4 mean 으로 10m 다운샘플** 한 결과
(`data/output/mtf_sim/{label}_K3A_*_simulated_final_2p5m.tif` → 10m).
모사의 목적이 "S2 처럼 보이게" 이므로 sim 은 항상 10m 격자로 비교한다.

영상 톤 정책 (memory feedback_figure_stretch):
  - K3A 패널은 자기들끼리 같은 톤 (독립 그룹 A)
  - Sim·S2 패널은 같은 톤 (그룹 B; 모사가 잘 됐다면 두 패널이 시각적으로 일치)
Row 2 (zoom) 은 Row 1 의 vmin/vmax 를 그대로 재사용 → 톤 일관성.

사용:
  python module4_visualization/scripts/fig_radsim_overview.py --label 10
  python module4_visualization/scripts/fig_radsim_overview.py --label 10 --zoom-bbox <minx> <miny> <maxx> <maxy>
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import from_bounds

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from module4_visualization.src import (
    aggregate_4x4_mean,
    locate_pair,
    percentile_stretch,
)
from shared.utils.paths import resolve_path

BAND_ORDER: Tuple[str, ...] = ("Blue", "Green", "Red", "NIR")
BAND_COLORS: Dict[str, str] = {
    "Blue": "#1f77b4",
    "Green": "#2ca02c",
    "Red": "#d62728",
    "NIR": "#7f3fbf",
}
K3A_SUFFIX_FOR_BAND: Dict[str, str] = {"Blue": "B", "Green": "G", "Red": "R", "NIR": "N"}
S2_CODE_FOR_BAND: Dict[str, str] = {"Blue": "B02", "Green": "B03", "Red": "B04", "NIR": "B08"}
SIM_INDEX_FOR_BAND: Dict[str, int] = {"Blue": 1, "Green": 2, "Red": 3, "NIR": 4}
RGB_INDICES: Tuple[int, int, int] = (2, 1, 0)  # BAND_ORDER (B,G,R,N) -> R,G,B 순서


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def read_sim_4band_with_meta(sim_path: Path) -> Tuple[np.ndarray, dict]:
    """rad+MTF 최종 모사 TIF 에서 BAND_ORDER 순서(B,G,R,N) 로 4밴드 + meta 읽기."""
    arrs: List[np.ndarray] = []
    meta: Optional[dict] = None
    with rasterio.open(sim_path) as src:
        for band in BAND_ORDER:
            arrs.append(src.read(SIM_INDEX_FOR_BAND[band]))
        meta = {"transform": src.transform, "crs": src.crs, "shape": (src.height, src.width)}
    return np.stack(arrs, axis=0), meta or {}


def read_k3a_4band_with_meta(k3a_grid: Dict[str, Path]) -> Tuple[np.ndarray, dict]:
    arrs: List[np.ndarray] = []
    meta: Optional[dict] = None
    for band in BAND_ORDER:
        with rasterio.open(k3a_grid[K3A_SUFFIX_FOR_BAND[band]]) as src:
            arrs.append(src.read(1))
            if meta is None:
                meta = {"transform": src.transform, "crs": src.crs, "shape": (src.height, src.width)}
    return np.stack(arrs, axis=0), meta or {}


def read_s2_4band_with_meta(s2_grid: Dict[str, Path]) -> Tuple[np.ndarray, dict]:
    arrs: List[np.ndarray] = []
    meta: Optional[dict] = None
    for band in BAND_ORDER:
        with rasterio.open(s2_grid[S2_CODE_FOR_BAND[band]]) as src:
            arrs.append(src.read(1))
            if meta is None:
                meta = {"transform": src.transform, "crs": src.crs, "shape": (src.height, src.width)}
    return np.stack(arrs, axis=0), meta or {}


def compute_default_zoom_bbox(meta: dict, frac: float,
                              center_frac: Tuple[float, float] = (0.5, 0.5)
                              ) -> Tuple[float, float, float, float]:
    """meta UTM extent 안에서 (fx, fy) 분수 위치를 중심으로 frac 비율의 정사각 bbox.

    center_frac=(0.5,0.5) 는 영상 중앙. (0.25,0.25) 는 좌상단 사분면 중앙.
    fx 는 가로(서→동), fy 는 세로(상→하) 방향 분수.
    영상 가장자리에 닿으면 안쪽으로 clamp.
    """
    tr = meta["transform"]
    h, w = meta["shape"]
    minx = tr.c
    maxy = tr.f
    maxx = minx + w * tr.a
    miny = maxy + h * tr.e  # tr.e 는 음수
    fx, fy = center_frac
    cx = minx + fx * (maxx - minx)
    cy = maxy + fy * (miny - maxy)  # fy=0 이면 maxy(상단), fy=1 이면 miny(하단)
    side = frac * min(maxx - minx, maxy - miny)
    half = side / 2.0
    # 가장자리 clamp
    cx = min(max(cx, minx + half), maxx - half)
    cy = min(max(cy, miny + half), maxy - half)
    return (cx - half, cy - half, cx + half, cy + half)


def upsample_nearest(arr: np.ndarray, factor: int = 4) -> np.ndarray:
    """(C,H,W) 배열을 nearest-neighbor 로 factor× 업샘플 (표시용 — 값은 보존, 4×4 블록 구조 노출)."""
    return np.repeat(np.repeat(arr, factor, axis=1), factor, axis=2)


def downsample_meta(meta: dict, factor: int = 4) -> dict:
    """4×4 mean 다운샘플 후의 transform/shape (origin 유지, pixel size × factor)."""
    from rasterio.transform import Affine
    tr = meta["transform"]
    new_tr = Affine(tr.a * factor, tr.b, tr.c, tr.d, tr.e * factor, tr.f)
    h, w = meta["shape"]
    return {"transform": new_tr, "crs": meta["crs"], "shape": (h // factor, w // factor)}


def crop_to_bbox(stack: np.ndarray, meta: dict, bbox: Tuple[float, float, float, float]) -> np.ndarray:
    """(C,H,W) 배열을 UTM bbox 에 맞게 픽셀 단위로 crop."""
    win = from_bounds(*bbox, transform=meta["transform"]).round_lengths().round_offsets()
    row_off = max(0, int(win.row_off))
    col_off = max(0, int(win.col_off))
    row_end = min(stack.shape[1], row_off + int(win.height))
    col_end = min(stack.shape[2], col_off + int(win.width))
    return stack[:, row_off:row_end, col_off:col_end]


def per_channel_vmin_vmax(stacks: List[np.ndarray], pmin: float, pmax: float, nodata: float = 0.0) -> np.ndarray:
    """여러 (3,H,W) RGB 스택을 채널별로 합쳐 (3,2) percentile 반환."""
    rng = np.random.default_rng(0)
    out = np.zeros((3, 2), dtype=np.float32)
    for c in range(3):
        chunks: List[np.ndarray] = []
        for s in stacks:
            ch = s[c].astype(np.float32).ravel()
            valid = ch[ch != nodata]
            if valid.size == 0:
                continue
            sample = valid if valid.size <= 2_000_000 else rng.choice(valid, 2_000_000, replace=False)
            chunks.append(sample)
        if not chunks:
            out[c] = (0.0, 1.0)
            continue
        cat = np.concatenate(chunks)
        vmin = float(np.percentile(cat, pmin))
        vmax = float(np.percentile(cat, pmax))
        if vmax <= vmin:
            vmax = vmin + 1.0
        out[c] = (vmin, vmax)
    return out


def shared_xrange(stacks: List[np.ndarray], pmin: float, pmax: float,
                  band_indices: Optional[List[int]] = None) -> Tuple[float, float]:
    """4밴드 스택들에서 공유 x 축 범위 (NIR 은 band_indices 로 제외 가능)."""
    if band_indices is None:
        band_indices = list(range(stacks[0].shape[0]))
    chunks: List[np.ndarray] = []
    rng = np.random.default_rng(0)
    for stack in stacks:
        for c in band_indices:
            ch = stack[c].astype(np.float32).ravel()
            valid = ch[ch != 0.0]
            if valid.size == 0:
                continue
            sample = valid if valid.size <= 1_000_000 else rng.choice(valid, 1_000_000, replace=False)
            chunks.append(sample)
    if not chunks:
        return (0.0, 1.0)
    cat = np.concatenate(chunks)
    lo = float(np.percentile(cat, pmin))
    hi = float(np.percentile(cat, pmax))
    if hi <= lo:
        hi = lo + 1.0
    return (lo, hi)


def imshow_panel(ax: plt.Axes, stack4: np.ndarray, vv: np.ndarray, title: str) -> None:
    """(4,H,W) → R/G/B 추출 후 percentile_stretch 로 imshow. nodata(R=G=B=0) 는 alpha=0 으로 투명."""
    rgb_stack = stack4[list(RGB_INDICES)]
    rgb01, _ = percentile_stretch(rgb_stack, 2.0, 98.0, vmin_vmax=vv, per_channel=True)
    nodata_mask = (rgb_stack == 0).all(axis=0)
    alpha = np.where(nodata_mask, 0.0, 1.0).astype(np.float32)
    rgba = np.concatenate([rgb01, alpha[..., None]], axis=-1)
    ax.imshow(rgba, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def hist_panel(ax: plt.Axes, stack_10m: np.ndarray, title: str, bins: int,
               xrange: Tuple[float, float], logy: bool, fill_alpha: float,
               sample_cap: Optional[int]) -> None:
    rng = np.random.default_rng(0)
    for c, band in enumerate(BAND_ORDER):
        ch = stack_10m[c].astype(np.float32).ravel()
        valid = ch[ch != 0.0]
        if valid.size == 0:
            continue
        if sample_cap is not None and valid.size > sample_cap:
            valid = rng.choice(valid, sample_cap, replace=False)
        color = BAND_COLORS[band]
        ax.hist(valid, bins=bins, range=xrange, histtype="stepfilled",
                color=color, alpha=fill_alpha, label=band)
        ax.hist(valid, bins=bins, range=xrange, histtype="step",
                color=color, linewidth=1.2)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(*xrange)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")


def main() -> None:
    parser = argparse.ArgumentParser(description="방사 모사 종합 figure (3x3)")
    parser.add_argument("--label", type=int, required=True)
    parser.add_argument("--zoom-bbox", nargs=4, type=float, default=None,
                        metavar=("MINX", "MINY", "MAXX", "MAXY"),
                        help="zoom UTM bbox (없으면 --zoom-frac 로 중앙 crop)")
    parser.add_argument("--zoom-frac", type=float, default=0.15, help="zoom-bbox 미지정 시 crop 비율 (기본 0.15)")
    parser.add_argument("--zoom-center", nargs=2, type=float, default=[0.5, 0.5], metavar=("FX", "FY"),
                        help="zoom 중심 분수 위치 (FX=가로 서→동, FY=세로 상→하, 둘 다 0~1, 기본 0.5 0.5)")
    parser.add_argument("--img-pmin", type=float, default=2.0, help="영상 stretch lower percentile")
    parser.add_argument("--img-pmax", type=float, default=98.0, help="영상 stretch upper percentile")
    parser.add_argument("--hist-pmin", type=float, default=1.0, help="히스토그램 x 축 lower percentile")
    parser.add_argument("--hist-pmax", type=float, default=99.0, help="히스토그램 x 축 upper percentile")
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--fill-alpha", type=float, default=0.3)
    parser.add_argument("--logy", action="store_true")
    parser.add_argument("--per-column-range", action="store_true", help="히스토 x 축 열별 독립 (기본 공유)")
    parser.add_argument("--include-nir-range", action="store_true", help="히스토 x 축 범위 계산에 NIR 포함")
    parser.add_argument("--sample-cap", type=int, default=2_000_000)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--figwidth", type=float, default=15.0)
    args = parser.parse_args()

    setup_logging()
    log = logging.getLogger("fig_radsim_overview")

    pair = locate_pair(args.label)
    if not pair.has_simulated():
        log.error("label=%d 에 simulated_final_2p5m.tif 가 없습니다: %s", args.label, pair.simulated)
        sys.exit(1)

    log.info("[label=%d] K3A 4밴드 로드 (2.5m)", args.label)
    k3a_25, k3a_meta = read_k3a_4band_with_meta(pair.k3a_grid)
    log.info("  K3A shape=%s transform=%s", k3a_25.shape, k3a_meta["transform"])

    log.info("[label=%d] Sim 최종 모사 4밴드 로드 (rad+MTF, 2.5m): %s",
             args.label, Path(pair.simulated).name)
    sim_25, sim_meta_25 = read_sim_4band_with_meta(pair.simulated)
    log.info("  Sim 2.5m shape=%s → 4×4 mean 으로 10m 다운샘플", sim_25.shape)
    sim_10 = aggregate_4x4_mean(sim_25)
    sim_meta = downsample_meta(sim_meta_25, factor=4)
    log.info("  Sim 10m shape=%s transform=%s", sim_10.shape, sim_meta["transform"])

    log.info("[label=%d] S2 4밴드 로드 (10m)", args.label)
    s2_10, s2_meta = read_s2_4band_with_meta(pair.s2_grid)
    log.info("  S2  shape=%s transform=%s", s2_10.shape, s2_meta["transform"])

    if args.zoom_bbox is not None:
        zoom_bbox = tuple(args.zoom_bbox)  # type: ignore[assignment]
    else:
        zoom_bbox = compute_default_zoom_bbox(s2_meta, args.zoom_frac,
                                              center_frac=tuple(args.zoom_center))
    log.info("  zoom UTM bbox=%s (center_frac=%s, frac=%s)",
             zoom_bbox, tuple(args.zoom_center), args.zoom_frac)

    # 영상 row crop (zoom). sim 은 10m 격자
    k3a_zoom = crop_to_bbox(k3a_25, k3a_meta, zoom_bbox)
    sim_zoom = crop_to_bbox(sim_10, sim_meta, zoom_bbox)
    s2_zoom = crop_to_bbox(s2_10, s2_meta, zoom_bbox)
    log.info("  zoom shapes — K3A:%s sim:%s S2:%s", k3a_zoom.shape, sim_zoom.shape, s2_zoom.shape)

    # Row 1 stretch (full): 그룹 A = K3A 단독, 그룹 B = sim+S2 공유
    k3a_rgb_full = k3a_25[list(RGB_INDICES)]
    sim_rgb_full = sim_10[list(RGB_INDICES)]
    s2_rgb_full = s2_10[list(RGB_INDICES)]
    vv_a_full = per_channel_vmin_vmax([k3a_rgb_full], args.img_pmin, args.img_pmax)
    vv_b_full = per_channel_vmin_vmax([sim_rgb_full, s2_rgb_full], args.img_pmin, args.img_pmax)
    log.info("  row1 vmin/vmax — group A (K3A): %s | group B (sim+S2): %s",
             vv_a_full.tolist(), vv_b_full.tolist())

    # Row 2 (zoom) 은 Row 1 의 vmin/vmax 를 그대로 재사용 → 톤 일관성

    # Row 3 histogram: K3A 는 4×4 mean 으로 10m, sim 은 이미 10m
    log.info("  K3A → 10m 다운샘플 (히스토그램용)")
    k3a_10 = aggregate_4x4_mean(k3a_25)

    range_band_indices = None if args.include_nir_range else [0, 1, 2]
    if args.per_column_range:
        h_ranges = [
            shared_xrange([k3a_10], args.hist_pmin, args.hist_pmax, range_band_indices),
            shared_xrange([sim_10], args.hist_pmin, args.hist_pmax, range_band_indices),
            shared_xrange([s2_10],  args.hist_pmin, args.hist_pmax, range_band_indices),
        ]
    else:
        shared = shared_xrange([k3a_10, sim_10, s2_10], args.hist_pmin, args.hist_pmax, range_band_indices)
        h_ranges = [shared, shared, shared]
        log.info("  shared hist x-range: %s", shared)

    sample_cap = args.sample_cap if args.sample_cap > 0 else None

    fig, axes = plt.subplots(3, 3, figsize=(args.figwidth, args.figwidth * 0.95))

    # Row 1
    # Sim·S2 은 표시용으로만 4× nearest 업샘플 — 값은 10m 집계 그대로, 4×4 픽셀 블록 구조가 화면에 드러남
    sim_disp = upsample_nearest(sim_10, factor=4)
    s2_disp = upsample_nearest(s2_10, factor=4)
    sim_zoom_disp = upsample_nearest(sim_zoom, factor=4)
    s2_zoom_disp = upsample_nearest(s2_zoom, factor=4)

    imshow_panel(axes[0, 0], k3a_25, vv_a_full, "(a) Kompsat-3A 2.5 m")
    imshow_panel(axes[0, 1], sim_disp, vv_b_full, "(b) Simulation (rad + MTF) 10 m")
    imshow_panel(axes[0, 2], s2_disp, vv_b_full, "(c) Sentinel-2 10 m")
    # Row 2 — Row 1 의 vmin/vmax 그대로 재사용
    imshow_panel(axes[1, 0], k3a_zoom, vv_a_full, "(d) K3A zoom")
    imshow_panel(axes[1, 1], sim_zoom_disp, vv_b_full, "(e) Sim zoom (10 m blocks)")
    imshow_panel(axes[1, 2], s2_zoom_disp,  vv_b_full, "(f) S2 zoom (10 m blocks)")
    # Row 3
    hist_panel(axes[2, 0], k3a_10, "(g) K3A histogram (DN, 10 m)",
               args.bins, h_ranges[0], args.logy, args.fill_alpha, sample_cap)
    hist_panel(axes[2, 1], sim_10, "(h) Sim histogram (10 m)",
               args.bins, h_ranges[1], args.logy, args.fill_alpha, sample_cap)
    hist_panel(axes[2, 2], s2_10, "(i) S2 histogram (10 m)",
               args.bins, h_ranges[2], args.logy, args.fill_alpha, sample_cap)
    for c in range(3):
        axes[2, c].set_xlabel("Pixel value")
    axes[2, 0].set_ylabel("Pixel count" + (" (log)" if args.logy else ""))

    fig.suptitle(f"Simulation overview (rad + MTF) — pair label {args.label}", fontsize=13, y=1.0)
    fig.tight_layout()

    out_path = args.out
    if out_path is None:
        out_dir = resolve_path("output_dir") / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"radsim_overview_label{args.label}.png"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("저장: %s", out_path)


if __name__ == "__main__":
    main()
