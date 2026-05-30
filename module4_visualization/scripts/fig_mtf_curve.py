"""fig_mtf_curve.py - 상대적 PSF (sigma) vs robust MSE 곡선 (단일 패널, 4밴드 오버레이).

MTF 모사 시퀀스(`module3_simulation.src.mtf.estimate_relative_psf_per_band`) 를 그대로 재현해
밴드별로 수렴한 (cutoff, 패치) 를 알아낸 뒤, 같은 설정에서 σ grid 를 추가로 스캔해 곡선을 만든다.
- 입력: K3A grid 2.5m (4밴드) + rad-sim coeffs JSON → 방사 모사 K3A 2.5m → MTF 패치 매칭
- 곡선: σ ∈ [sigma_min, sigma_max] step σ_grid_step, robust MSE (cutoff 백분위 컷)
- 마커: ★ = scan grid 상의 최소점, 점선 vertical = mtfsim.json 에 저장된 best_sigma

사용:
  python module4_visualization/scripts/fig_mtf_curve.py --label 10
  python module4_visualization/scripts/fig_mtf_curve.py --label 10 --sigma-step 0.1
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
from scipy.ndimage import gaussian_filter, shift
from scipy.optimize import minimize_scalar
from skimage.registration import phase_cross_correlation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from module3_simulation.src.mtf import (
    aggregate_4x4_mean,
    find_best_patch,
    normalized_gaussian_filter,
)
from module4_visualization.src import locate_pair
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

CUTOFFS = [95, 90, 85, 80, 75, 70, 60, 50, 40, 30]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def load_radsim_coeffs(label: int) -> Dict[str, Tuple[float, float]]:
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
        raise ValueError(f"label={label} coeffs 누락 밴드: {missing}")
    return coeffs


def load_mtfsim_summary(label: int) -> Dict[str, float]:
    """*_mtfsim.json 에서 밴드별 saved best_sigma 로드 (없으면 빈 dict)."""
    mtf_dir = resolve_path("mtf_sim_dir")
    candidates = sorted(mtf_dir.glob(f"{label}_K3A_*_mtfsim.json"))
    if not candidates:
        return {}
    payload = json.loads(candidates[0].read_text(encoding="utf-8"))
    saved: Dict[str, float] = {}
    for entry in payload.get("mtf_results", []):
        saved[entry["band_name"]] = float(entry["best_sigma"])
    return saved


def apply_radsim(k3a_2p5m: np.ndarray, coeffs: Dict[str, Tuple[float, float]]) -> np.ndarray:
    out = np.zeros_like(k3a_2p5m, dtype=np.float32)
    for c, band in enumerate(BAND_ORDER):
        slope, intercept = coeffs[band]
        sim = slope * k3a_2p5m[c].astype(np.float32) + intercept
        sim = np.clip(sim, 0, None)
        sim[k3a_2p5m[c] == 0] = 0
        out[c] = sim
    return out


def read_k3a_4band(k3a_grid: Dict[str, Path]) -> np.ndarray:
    arrs: List[np.ndarray] = []
    for band in BAND_ORDER:
        with rasterio.open(k3a_grid[K3A_SUFFIX_FOR_BAND[band]]) as src:
            arrs.append(src.read(1))
    return np.stack(arrs, axis=0)


def read_s2_4band(s2_grid: Dict[str, Path]) -> np.ndarray:
    arrs: List[np.ndarray] = []
    for band in BAND_ORDER:
        with rasterio.open(s2_grid[S2_CODE_FOR_BAND[band]]) as src:
            arrs.append(src.read(1))
    return np.stack(arrs, axis=0)


def _build_objective(k_patch: np.ndarray, valid_k_mask: np.ndarray, s2_patch: np.ndarray,
                     is_perfect: bool, cutoff: float, nodata_value: float = 0.0):
    """mtf.py 의 objective 와 동일 (cutoff 고정)."""
    def objective(sigma: float) -> float:
        if is_perfect:
            blurred_2p5m = gaussian_filter(k_patch.astype(np.float32), sigma=sigma)
        else:
            blurred_2p5m = normalized_gaussian_filter(k_patch, valid_k_mask, sigma)
            blurred_2p5m[~valid_k_mask] = nodata_value
        blurred_10m = aggregate_4x4_mean(blurred_2p5m)
        h = min(blurred_10m.shape[0], s2_patch.shape[0])
        w = min(blurred_10m.shape[1], s2_patch.shape[1])
        b_crop = blurred_10m[:h, :w]
        s_crop = s2_patch[:h, :w]
        valid = (b_crop != nodata_value) & (s_crop != nodata_value)
        if valid.sum() == 0:
            return 1e9
        sq = (b_crop[valid] - s_crop[valid]) ** 2
        p_val = np.percentile(sq, cutoff)
        return float(np.mean(sq[sq <= p_val]))
    return objective


def _run_optimization(k_patch: np.ndarray, valid_k_mask: np.ndarray, s2_patch: np.ndarray,
                      is_perfect: bool, sigma_min: float, sigma_max: float,
                      band_name: str, log: logging.Logger) -> Tuple[float, float, int, bool]:
    """mtf.py::run_optimization 와 동일. (best_sigma, min_mse, used_cutoff, hit_boundary)."""
    best_sig = 0.0
    min_m = float("inf")
    used_c = 95
    hit_b = True
    for cutoff in CUTOFFS:
        objective = _build_objective(k_patch, valid_k_mask, s2_patch, is_perfect, cutoff)
        res = minimize_scalar(objective, bounds=(sigma_min, sigma_max),
                              method="bounded", options={"xatol": 0.05, "maxiter": 50})
        best_sig = float(res.x)
        min_m = float(res.fun)
        used_c = cutoff
        if (best_sig > sigma_min + 0.05) and (best_sig < sigma_max - 0.05):
            hit_b = False
            break
        else:
            log.warning("[%s] sigma=%.2f 경계값 수렴. cutoff=%d → 다음 단계로 재탐색",
                        band_name, best_sig, cutoff)
    return best_sig, min_m, used_c, hit_b


def trace_band_curve(
    k3a_band_2p5m: np.ndarray,
    s2_band_10m: np.ndarray,
    band_name: str,
    sigma_min: float,
    sigma_max: float,
    sigma_step: float,
    log: logging.Logger,
    nodata_value: float = 0.0,
    patch_size_10m: int = 500,
) -> Dict[str, object]:
    """mtf.py::estimate_relative_psf_per_band 의 시퀀스를 재현 후, 수렴한 (cutoff, 패치) 에서
    σ grid 스캔으로 곡선을 만든다."""
    y10, x10, is_perfect = find_best_patch(k3a_band_2p5m, s2_band_10m, nodata_value, patch_size_10m)
    ph = min(patch_size_10m, s2_band_10m.shape[0] - y10)
    pw = min(patch_size_10m, s2_band_10m.shape[1] - x10)
    s2_patch = s2_band_10m[y10:y10 + ph, x10:x10 + pw]
    y25, x25 = y10 * 4, x10 * 4
    ph25, pw25 = ph * 4, pw * 4
    k3a_patch = k3a_band_2p5m[y25:y25 + ph25, x25:x25 + pw25]
    valid_mask = (k3a_patch != nodata_value)
    log.info("[%s] patch is_perfect=%s pos=(y=%d,x=%d) shape=%s",
             band_name, is_perfect, y10, x10, s2_patch.shape)

    best_sig, min_m, used_c, hit_b = _run_optimization(
        k3a_patch, valid_mask, s2_patch, is_perfect, sigma_min, sigma_max, band_name, log,
    )
    used_patch = k3a_patch
    used_mask = valid_mask
    shifted = False

    if hit_b:
        log.warning("[%s] 모든 cutoff 실패 → phase correlation shift 시도", band_name)
        k_10m = aggregate_4x4_mean(k3a_patch)
        h = min(k_10m.shape[0], s2_patch.shape[0])
        w = min(k_10m.shape[1], s2_patch.shape[1])
        sh, _err, _ph = phase_cross_correlation(s2_patch[:h, :w], k_10m[:h, :w], upsample_factor=10)
        shy_25, shx_25 = sh[0] * 4, sh[1] * 4
        log.info("[%s] shift detected (2.5m): y=%.2f x=%.2f", band_name, shy_25, shx_25)
        k3a_shifted = shift(k3a_patch.astype(np.float32), shift=(shy_25, shx_25),
                            order=1, mode="constant", cval=nodata_value)
        valid_shifted = shift(valid_mask.astype(float), shift=(shy_25, shx_25),
                              order=0, mode="constant", cval=0).astype(bool)
        best_sig, min_m, used_c, hit_b = _run_optimization(
            k3a_shifted, valid_shifted, s2_patch, is_perfect, sigma_min, sigma_max, band_name, log,
        )
        used_patch = k3a_shifted
        used_mask = valid_shifted
        shifted = True

    log.info("[%s] 수렴: sigma=%.3f mse=%.3f cutoff=%d shifted=%s hit_b=%s",
             band_name, best_sig, min_m, used_c, shifted, hit_b)

    sigmas = np.arange(sigma_min, sigma_max + 0.5 * sigma_step, sigma_step, dtype=np.float64)
    objective = _build_objective(used_patch, used_mask, s2_patch, is_perfect, used_c)
    log.info("[%s] σ grid scan: %d points (cutoff=%d, shifted=%s)",
             band_name, sigmas.size, used_c, shifted)
    mses = np.array([objective(float(s)) for s in sigmas], dtype=np.float64)

    grid_min_idx = int(np.argmin(mses))
    return {
        "band_name": band_name,
        "sigmas": sigmas,
        "mses": mses,
        "converged_sigma": float(best_sig),
        "converged_mse": float(min_m),
        "grid_min_sigma": float(sigmas[grid_min_idx]),
        "grid_min_mse": float(mses[grid_min_idx]),
        "used_cutoff": int(used_c),
        "shifted": bool(shifted),
        "hit_boundary": bool(hit_b),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MTF σ vs MSE 곡선 (1 패널, 4밴드 오버레이)")
    parser.add_argument("--label", type=int, required=True)
    parser.add_argument("--sigma-min", type=float, default=0.05)
    parser.add_argument("--sigma-max", type=float, default=9.0)
    parser.add_argument("--sigma-step", type=float, default=0.1, help="σ grid step (기본 0.1, 더 촘촘히는 0.05)")
    parser.add_argument("--y-mode", choices=["normalized", "absolute-log"], default="normalized",
                        help="y 축: normalized=MSE/min_MSE 밴드별 정규화 + linear (기본), absolute-log=절대 MSE + log")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--figwidth", type=float, default=8.0)
    parser.add_argument("--ylim-max", type=float, default=3.0,
                        help="normalized y축 상한 (기본 3.0 = local min 의 3배). 곡선 시작부 급변 잘라내고 수렴 부근 확대.")
    args = parser.parse_args()

    setup_logging()
    log = logging.getLogger("fig_mtf_curve")

    pair = locate_pair(args.label)
    coeffs = load_radsim_coeffs(args.label)
    saved = load_mtfsim_summary(args.label)

    log.info("[label=%d] K3A grid 4밴드 로드 + rad-sim 적용", args.label)
    k3a_25 = read_k3a_4band(pair.k3a_grid)
    sim_25 = apply_radsim(k3a_25, coeffs)
    log.info("[label=%d] S2 grid 4밴드 로드", args.label)
    s2_10 = read_s2_4band(pair.s2_grid)

    curves: List[Dict[str, object]] = []
    for c, band in enumerate(BAND_ORDER):
        log.info("=== %s ===", band)
        curve = trace_band_curve(
            sim_25[c], s2_10[c], band,
            sigma_min=args.sigma_min, sigma_max=args.sigma_max,
            sigma_step=args.sigma_step, log=log,
        )
        if band in saved:
            curve["saved_sigma"] = saved[band]
        curves.append(curve)

    xlim_hi = 7.0
    fig, ax = plt.subplots(1, 1, figsize=(args.figwidth, args.figwidth * 0.7))
    axins = ax.inset_axes([0.55, 0.45, 0.42, 0.42]) if args.y_mode == "normalized" else None
    local_sigma_list: List[float] = []
    for curve in curves:
        band = curve["band_name"]
        color = BAND_COLORS[band]
        sigmas = curve["sigmas"]
        mses = curve["mses"]
        local_idx = int(np.argmin(np.where(sigmas <= xlim_hi, mses, np.inf)))
        local_sigma = float(sigmas[local_idx])
        local_mse = float(mses[local_idx])
        local_sigma_list.append(local_sigma)
        if args.y_mode == "normalized":
            y = mses / max(local_mse, 1e-12)
            y_marker = 1.0
        else:
            y = mses
            y_marker = local_mse
        ax.plot(sigmas, y, color=color, linewidth=1.6,
                label=f"{band} (σ*={local_sigma:.2f}, cutoff={curve['used_cutoff']}%)")
        ax.plot(local_sigma, y_marker,
                marker="o", markersize=8, color=color,
                markeredgecolor="black", markeredgewidth=0.6, zorder=5)
        if "saved_sigma" in curve:
            ax.axvline(curve["saved_sigma"], color=color, linestyle="--", linewidth=0.9, alpha=0.5)
        if axins is not None:
            axins.plot(sigmas, y, color=color, linewidth=1.4)
            axins.plot(local_sigma, y_marker, marker="o", markersize=6, color=color,
                       markeredgecolor="black", markeredgewidth=0.5, zorder=5)

    if args.y_mode == "absolute-log":
        ax.set_yscale("log")
        ax.set_ylabel("Robust MSE (log scale)")
    else:
        ax.set_ylabel("Robust MSE / min MSE  (per band)")
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_ylim(0.95, args.ylim_max)
    ax.set_xlabel("Relative PSF σ (pixels @ 2.5 m)")
    ax.set_xlim(args.sigma_min, xlim_hi)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="upper left" if args.y_mode == "normalized" else "upper right")
    ax.set_title(f"MTF σ search curve — pair label {args.label}\n"
                 f"● = local minimum (σ ≤ {xlim_hi:.0f}), dashed = saved best_sigma (mtfsim.json)", fontsize=11)

    if axins is not None and local_sigma_list:
        s_lo = max(args.sigma_min, min(local_sigma_list) - 1.0)
        s_hi = min(xlim_hi, max(local_sigma_list) + 1.0)
        axins.set_xlim(s_lo, s_hi)
        axins.set_ylim(0.99, 1.10)
        axins.axhline(1.0, color="gray", linestyle=":", linewidth=0.7, alpha=0.6)
        axins.grid(True, alpha=0.3)
        axins.tick_params(labelsize=8)
        axins.set_title("zoom: convergence", fontsize=9)
        ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.5)

    fig.tight_layout()

    out_path = args.out
    if out_path is None:
        out_dir = resolve_path("output_dir") / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"mtf_curve_label{args.label}.png"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("저장: %s", out_path)


if __name__ == "__main__":
    main()
