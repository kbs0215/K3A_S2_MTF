"""fig_radsim_histogram.py - 방사 모사 결과 히스토그램 (3 열 x 4 밴드 오버레이).

[K3A 원본 10m] | [방사 모사 10m] | [Sentinel-2 원본 10m]

각 열에 4밴드(B,G,R,NIR) 히스토그램을 색깔별로 오버레이.
방사 전용 TIF (rad_sim/*_simulated_2p5m.tif) 는 MTF 후 정리되었을 수 있어,
대신 *_radsim_coeffs.json 의 (slope, intercept) 를 K3A 2.5m grid 에 즉시 적용해 rad-sim 분포를 도출.
K3A 원본·방사모사는 4×4 mean 으로 10m 로 다운샘플 (S2 격자와 일치).
열별로 단위가 달라 (K3A: raw DN / rad-sim·S2: S2 reflectance 스케일) x 축은 열별 독립.

사용:
  python module4_visualization/scripts/fig_radsim_histogram.py --label 10
  python module4_visualization/scripts/fig_radsim_histogram.py --label 10 --bins 100 --logy
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from module4_visualization.src import aggregate_4x4_mean, locate_pair
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


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def load_radsim_coeffs(label: int) -> Dict[str, Tuple[float, float]]:
    """*_radsim_coeffs.json 또는 *_radsim.json 에서 밴드별 (slope, intercept) 로드."""
    rad_dir = resolve_path("rad_sim_dir")
    candidates = sorted(rad_dir.glob(f"{label}_K3A_*_radsim_coeffs.json"))
    if not candidates:
        candidates = sorted(rad_dir.glob(f"{label}_K3A_*_radsim.json"))
    if not candidates:
        raise FileNotFoundError(f"label={label} rad-sim coeffs JSON 없음 in {rad_dir}")
    payload = json.loads(candidates[0].read_text(encoding="utf-8"))
    coeffs: Dict[str, Tuple[float, float]] = {}
    for entry in payload.get("band_coefficients", []):
        coeffs[entry["band_name"]] = (float(entry["slope"]), float(entry["intercept"]))
    missing = [b for b in BAND_ORDER if b not in coeffs]
    if missing:
        raise ValueError(f"label={label} coeffs 누락 밴드: {missing} (source={candidates[0].name})")
    return coeffs


def apply_radsim(k3a_2p5m: np.ndarray, coeffs: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """K3A 2.5m (4,H,W) 에 밴드별 (slope, intercept) 선형 변환 적용. nodata(0) 보존, 음수 0 clip."""
    out = np.zeros_like(k3a_2p5m, dtype=np.float32)
    for c, band in enumerate(BAND_ORDER):
        slope, intercept = coeffs[band]
        sim = slope * k3a_2p5m[c].astype(np.float32) + intercept
        sim = np.clip(sim, 0, None)
        sim[k3a_2p5m[c] == 0] = 0
        out[c] = sim
    return out


def read_k3a_4band(k3a_grid: Dict[str, Path]) -> np.ndarray:
    """K3A grid 4밴드를 (4,H,W) 로 읽음. 밴드 순서는 BAND_ORDER 와 동일."""
    arrs: List[np.ndarray] = []
    for band in BAND_ORDER:
        suffix = K3A_SUFFIX_FOR_BAND[band]
        with rasterio.open(k3a_grid[suffix]) as src:
            arrs.append(src.read(1))
    return np.stack(arrs, axis=0)


def read_s2_4band(s2_grid: Dict[str, Path]) -> np.ndarray:
    arrs: List[np.ndarray] = []
    for band in BAND_ORDER:
        code = S2_CODE_FOR_BAND[band]
        with rasterio.open(s2_grid[code]) as src:
            arrs.append(src.read(1))
    return np.stack(arrs, axis=0)


def _shared_xrange(
    stacks: List[np.ndarray],
    pmin: float,
    pmax: float,
    band_indices: Optional[List[int]] = None,
) -> Tuple[float, float]:
    """여러 스택에 걸쳐 valid 픽셀(!=0) 합쳐 percentile 로 공유 x 축 범위 결정.

    band_indices=None 이면 모든 밴드 사용. NIR 의 우측 tail 이 x 축을 끌고 가는 것을
    피하려면 [0,1,2] (B/G/R) 만 넘김 → NIR 은 plot 시 우측이 살짝 잘릴 수 있음.
    """
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


def _plot_column(
    ax: plt.Axes,
    stack_10m: np.ndarray,
    title: str,
    bins: int,
    xrange: Tuple[float, float],
    logy: bool,
    sample_cap: Optional[int],
    fill_alpha: float,
) -> None:
    rng = np.random.default_rng(0)
    for c, band in enumerate(BAND_ORDER):
        ch = stack_10m[c].astype(np.float32).ravel()
        valid = ch[ch != 0.0]
        if valid.size == 0:
            continue
        if sample_cap is not None and valid.size > sample_cap:
            valid = rng.choice(valid, sample_cap, replace=False)
        color = BAND_COLORS[band]
        ax.hist(
            valid,
            bins=bins,
            range=xrange,
            histtype="stepfilled",
            color=color,
            alpha=fill_alpha,
            label=band,
        )
        ax.hist(
            valid,
            bins=bins,
            range=xrange,
            histtype="step",
            color=color,
            linewidth=1.2,
        )
    ax.set_title(title, fontsize=11)
    ax.set_xlim(*xrange)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")


def main() -> None:
    parser = argparse.ArgumentParser(description="방사 모사 히스토그램 (3 열, 밴드 오버레이)")
    parser.add_argument("--label", type=int, required=True, help="페어 label (예: 10)")
    parser.add_argument("--bins", type=int, default=80, help="히스토그램 빈 수 (기본 80)")
    parser.add_argument("--pmin", type=float, default=1.0, help="x 축 lower percentile (기본 1)")
    parser.add_argument("--pmax", type=float, default=99.0, help="x 축 upper percentile (기본 99)")
    parser.add_argument("--logy", action="store_true", help="y 축 로그 스케일")
    parser.add_argument("--fill-alpha", type=float, default=0.3, help="히스토그램 채움 투명도 (기본 0.3)")
    parser.add_argument("--per-column-range", action="store_true", help="x 축을 열별 독립 (기본: 3 열 공유)")
    parser.add_argument("--include-nir-range", action="store_true", help="x 축 범위 계산에 NIR 포함 (기본: B/G/R 만 → NIR 우측 살짝 잘릴 수 있음)")
    parser.add_argument("--sample-cap", type=int, default=2_000_000, help="밴드별 샘플링 상한 (기본 2M, 0 이면 전체)")
    parser.add_argument("--out", type=Path, default=None, help="출력 PNG 경로")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--figwidth", type=float, default=15.0)
    args = parser.parse_args()

    setup_logging()
    log = logging.getLogger("fig_radsim_histogram")

    pair = locate_pair(args.label)
    coeffs = load_radsim_coeffs(args.label)
    log.info("[label=%d] rad-sim coeffs: %s",
             args.label, {b: (round(s, 4), round(i, 2)) for b, (s, i) in coeffs.items()})

    log.info("  K3A 4밴드 로드")
    k3a_25 = read_k3a_4band(pair.k3a_grid)
    log.info("  rad-sim 4밴드 = K3A * slope + intercept (밴드별)")
    sim_25 = apply_radsim(k3a_25, coeffs)
    log.info("  S2 4밴드 로드")
    s2_10 = read_s2_4band(pair.s2_grid)

    log.info("  K3A/rad-sim 4×4 mean 다운샘플 (2.5m -> 10m)")
    k3a_10 = aggregate_4x4_mean(k3a_25)
    sim_10 = aggregate_4x4_mean(sim_25)
    log.info("  shapes — K3A 10m:%s sim 10m:%s S2 10m:%s", k3a_10.shape, sim_10.shape, s2_10.shape)

    sample_cap = args.sample_cap if args.sample_cap > 0 else None

    range_band_indices = None if args.include_nir_range else [0, 1, 2]  # NIR=3 제외
    if args.per_column_range:
        ranges = [
            _shared_xrange([k3a_10], args.pmin, args.pmax, range_band_indices),
            _shared_xrange([sim_10], args.pmin, args.pmax, range_band_indices),
            _shared_xrange([s2_10],  args.pmin, args.pmax, range_band_indices),
        ]
    else:
        shared = _shared_xrange([k3a_10, sim_10, s2_10], args.pmin, args.pmax, range_band_indices)
        ranges = [shared, shared, shared]
        log.info("  shared x-range (B/G/R%s): %s",
                 "+NIR" if args.include_nir_range else "", shared)

    fig, axes = plt.subplots(1, 3, figsize=(args.figwidth, args.figwidth / 3.0 * 0.85))
    _plot_column(axes[0], k3a_10, "(a) Kompsat-3A original (DN)\n10 m (4×4 mean)",
                 args.bins, ranges[0], args.logy, sample_cap, args.fill_alpha)
    _plot_column(axes[1], sim_10, "(b) Radiometric simulation\n10 m (4×4 mean)",
                 args.bins, ranges[1], args.logy, sample_cap, args.fill_alpha)
    _plot_column(axes[2], s2_10, "(c) Sentinel-2 original\n10 m",
                 args.bins, ranges[2], args.logy, sample_cap, args.fill_alpha)
    for ax in axes:
        ax.set_xlabel("Pixel value")
    axes[0].set_ylabel("Pixel count" + (" (log)" if args.logy else ""))
    fig.suptitle(f"Radiometric simulation histogram — pair label {args.label}", fontsize=12, y=1.02)
    fig.tight_layout()

    out_path = args.out
    if out_path is None:
        out_dir = resolve_path("output_dir") / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"radsim_histogram_label{args.label}.png"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("저장: %s", out_path)


if __name__ == "__main__":
    main()
