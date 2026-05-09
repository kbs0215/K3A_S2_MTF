"""mtf.py - 상대적 PSF 추정 기반 MTF 모사 모듈

고해상도 영상(K3A)에 가우시안 점확산함수(PSF) 커널을 적용하여 다운샘플링한 뒤,
저해상도 영상(S2)과의 평균제곱오차(MSE)를 최소화하는 최적의 sigma를 탐색합니다.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, shift
from scipy.optimize import minimize_scalar
from skimage.registration import phase_cross_correlation

logger = logging.getLogger(__name__)


@dataclass
class MTFResult:
    """단일 밴드 MTF 모사 최적화 결과"""
    band_name: str
    best_sigma: float
    min_mse: float
    simulated_band_2p5m: np.ndarray


def aggregate_4x4_mean(arr: np.ndarray) -> np.ndarray:
    """K3A 2.5m → 10m 평균 다운샘플 (permissive).
    run_simulation.py 에 있는 것과 동일한 로직입니다.
    """
    if arr.ndim == 2:
        h, w = arr.shape
        h2 = (h // 4) * 4
        w2 = (w // 4) * 4
        arr = arr[:h2, :w2]
        
        blocked = arr.reshape(h // 4, 4, w // 4, 4).astype(np.float64)
        valid_blocked = (arr != 0).reshape(h // 4, 4, w // 4, 4)
        sum_valid = (blocked * valid_blocked).sum(axis=(1, 3))
        count_valid = valid_blocked.sum(axis=(1, 3))
        means = np.where(count_valid > 0, sum_valid / np.maximum(count_valid, 1), 0.0)
        return means.astype(arr.dtype)
    else:
        n, h, w = arr.shape
        h2 = (h // 4) * 4
        w2 = (w // 4) * 4
        arr = arr[:, :h2, :w2]
        
        blocked = arr.reshape(n, h // 4, 4, w // 4, 4).astype(np.float64)
        valid_blocked = (arr != 0).reshape(n, h // 4, 4, w // 4, 4)
        sum_valid = (blocked * valid_blocked).sum(axis=(2, 4))
        count_valid = valid_blocked.sum(axis=(2, 4))
        means = np.where(count_valid > 0, sum_valid / np.maximum(count_valid, 1), 0.0)
        return means.astype(arr.dtype)


def find_best_patch(k3a_band_2p5m: np.ndarray, s2_band_10m: np.ndarray, nodata_value: float = 0.0, patch_size_10m: int = 500) -> Tuple[int, int, bool]:
    """100% 유효하며 엣지(대비)가 가장 뚜렷한 패치의 좌상단 좌표를 찾습니다."""
    h_10m, w_10m = s2_band_10m.shape
    stride = 250
    max_var = -1
    best_y, best_x = -1, -1
    
    for y in range(0, max(1, h_10m - patch_size_10m + 1), stride):
        for x in range(0, max(1, w_10m - patch_size_10m + 1), stride):
            y_end = min(h_10m, y + patch_size_10m)
            x_end = min(w_10m, x + patch_size_10m)
            s2_patch = s2_band_10m[y:y_end, x:x_end]
            
            y_2p5m, x_2p5m = y * 4, x * 4
            y_end_2p5m, x_end_2p5m = y_end * 4, x_end * 4
            k3a_patch = k3a_band_2p5m[y_2p5m:y_end_2p5m, x_2p5m:x_end_2p5m]
            
            # 크기가 모자라지 않고, 100% 유효한지 확인
            if s2_patch.size == patch_size_10m * patch_size_10m:
                if (s2_patch != nodata_value).all() and (k3a_patch != nodata_value).all():
                    var = np.var(s2_patch)
                    if var > max_var:
                        max_var = var
                        best_y, best_x = y, x
                        
    if max_var != -1:
        return best_y, best_x, True
        
    # 100% 유효 패치가 없으면 정중앙 반환
    ch_10m, cw_10m = h_10m // 2, w_10m // 2
    best_y = max(0, ch_10m - patch_size_10m // 2)
    best_x = max(0, cw_10m - patch_size_10m // 2)
    return best_y, best_x, False


def normalized_gaussian_filter(arr: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Method 2: 정규화된 가우시안 컨볼루션 (NoData 블리딩을 수학적으로 차단)"""
    V = arr.astype(np.float32)
    V[~mask] = 0.0
    W = mask.astype(np.float32)
    
    blurred_V = gaussian_filter(V, sigma=sigma)
    blurred_W = gaussian_filter(W, sigma=sigma)
    
    out = np.zeros_like(blurred_V)
    valid = blurred_W > 1e-4
    out[valid] = blurred_V[valid] / blurred_W[valid]
    return out


def estimate_relative_psf_per_band(
    k3a_band_2p5m: np.ndarray,
    s2_band_10m: np.ndarray,
    band_name: str,
    sigma_min: float,
    sigma_max: float,
    nodata_value: float = 0.0,
    patch_size_10m: int = 500
) -> MTFResult:
    # 1. 최고 품질 패치 탐색 (Method 1)
    y_10m, x_10m, is_perfect = find_best_patch(k3a_band_2p5m, s2_band_10m, nodata_value, patch_size_10m)
    
    ph = min(patch_size_10m, s2_band_10m.shape[0] - y_10m)
    pw = min(patch_size_10m, s2_band_10m.shape[1] - x_10m)
    s2_patch = s2_band_10m[y_10m:y_10m+ph, x_10m:x_10m+pw]
    
    y_2p5m, x_2p5m = y_10m * 4, x_10m * 4
    ph_2p5m, pw_2p5m = ph * 4, pw * 4
    k3a_patch = k3a_band_2p5m[y_2p5m:y_2p5m+ph_2p5m, x_2p5m:x_2p5m+pw_2p5m]
    
    valid_k3a_patch_mask = (k3a_patch != nodata_value)

    if is_perfect:
        logger.info("[%s] 100%% 유효한 최고 대비 패치 발견 (Method 1 적용)", band_name)
    else:
        logger.warning("[%s] 100%% 유효한 패치 없음. 정규화된 컨볼루션(Method 2) 대체 적용", band_name)

    cutoffs = [95, 90, 85, 80, 75, 70, 60, 50, 40, 30]  # 이상치 제거율
    
    def run_optimization(k_patch, valid_k_mask):
        best_sig = 0.0
        min_m = float('inf')
        used_c = 95
        hit_b = True

        for cutoff in cutoffs:
            def objective(sigma):
                if is_perfect:
                    blurred_patch_2p5m = gaussian_filter(k_patch.astype(np.float32), sigma=sigma)
                else:
                    blurred_patch_2p5m = normalized_gaussian_filter(k_patch, valid_k_mask, sigma)
                    blurred_patch_2p5m[~valid_k_mask] = nodata_value
                
                blurred_patch_10m = aggregate_4x4_mean(blurred_patch_2p5m)
                
                h = min(blurred_patch_10m.shape[0], s2_patch.shape[0])
                w = min(blurred_patch_10m.shape[1], s2_patch.shape[1])
                b_crop = blurred_patch_10m[:h, :w]
                s_crop = s2_patch[:h, :w]
                
                valid_mask = (b_crop != nodata_value) & (s_crop != nodata_value)
                if valid_mask.sum() == 0:
                    return 1e9
                    
                squared_errors = (b_crop[valid_mask] - s_crop[valid_mask])**2
                p_val = np.percentile(squared_errors, cutoff)
                robust_mse = np.mean(squared_errors[squared_errors <= p_val])
                
                return robust_mse

            res = minimize_scalar(
                objective,
                bounds=(sigma_min, sigma_max),
                method='bounded',
                options={'xatol': 0.05, 'maxiter': 50}
            )
            
            best_sig = res.x
            min_m = res.fun
            used_c = cutoff
            
            if (best_sig > sigma_min + 0.05) and (best_sig < sigma_max - 0.05):
                hit_b = False
                break 
            else:
                outlier_percent = 100 - cutoff
                logger.warning("[%s] sigma=%.2f 경계값 수렴. 이상치 제거율을 %d%%로 높여 재탐색합니다.", 
                               band_name, best_sig, outlier_percent + 5)
                               
        return best_sig, min_m, used_c, hit_b

    # 1차 최적화 시도
    best_sigma, min_mse, used_cutoff, hit_boundary = run_optimization(k3a_patch, valid_k3a_patch_mask)

    # 2. 위상 정합(Phase Correlation) 폴백 로직
    if hit_boundary:
        max_outlier = 100 - cutoffs[-1]
        logger.warning("[%s] 모든 이상치 제거율(최대 %d%%) 실패. 위상정합(Phase Correlation)을 통한 미세 정합 시도...", band_name, max_outlier)
        
        # S2와 K3A(10m 다운샘플링) 간의 픽셀 어긋남(Shift) 계산
        k_10m = aggregate_4x4_mean(k3a_patch)
        h_crop = min(k_10m.shape[0], s2_patch.shape[0])
        w_crop = min(k_10m.shape[1], s2_patch.shape[1])
        k_crop = k_10m[:h_crop, :w_crop]
        s_crop = s2_patch[:h_crop, :w_crop]
        
        # S2를 기준으로 K3A가 얼마나 이동해야 하는지 서브픽셀 단위로 탐색
        shift_10m, error, diffphase = phase_cross_correlation(s_crop, k_crop, upsample_factor=10)
        
        shift_y_2p5m = shift_10m[0] * 4
        shift_x_2p5m = shift_10m[1] * 4
        
        logger.info("[%s] 위상정합 결과: Y=%.2f, X=%.2f 픽셀(2.5m 기준) 이동 감지", band_name, shift_y_2p5m, shift_x_2p5m)
        
        # 감지된 어긋남만큼 K3A 패치를 물리적으로 이동 (정합 수행)
        k3a_patch_shifted = shift(k3a_patch.astype(np.float32), shift=(shift_y_2p5m, shift_x_2p5m), order=1, mode='constant', cval=nodata_value)
        valid_shifted = shift(valid_k3a_patch_mask.astype(float), shift=(shift_y_2p5m, shift_x_2p5m), order=0, mode='constant', cval=0).astype(bool)
        
        # 이동된 패치로 재탐색
        logger.info("[%s] 패치 이동 후 최적화 재시도...", band_name)
        best_sig_new, min_m_new, used_c_new, hit_b_new = run_optimization(k3a_patch_shifted, valid_shifted)
        
        if not hit_b_new:
            logger.info("[%s] 🎉 위상정합 후 최적화 성공! sigma=%.2f", band_name, best_sig_new)
        else:
            logger.error("[%s] ❌ 위상정합 후에도 최적화 실패. 경계값 유지.", band_name)
            
        best_sigma = best_sig_new
        min_mse = min_m_new
        used_cutoff = used_c_new

    outlier_percent = 100 - used_cutoff
    logger.info("[%s] 최종 최적 sigma 탐색 완료: sigma=%.2f, MSE=%.4f (Patch: %dx%d, 이상치 제거: %d%%)", 
                band_name, best_sigma, min_mse, pw_2p5m, ph_2p5m, outlier_percent)

    # 4. 찾은 최적 sigma를 원본 K3A 전체 영상에 딱 1번만 적용 (전체 영상은 항상 NoData를 포함하므로 Method 2 사용)
    valid_k3a_mask_2p5m = (k3a_band_2p5m != nodata_value)
    best_simulated_2p5m = normalized_gaussian_filter(k3a_band_2p5m, valid_k3a_mask_2p5m, best_sigma)
    best_simulated_2p5m[~valid_k3a_mask_2p5m] = nodata_value
    out_simulated = np.clip(best_simulated_2p5m, 0, 65535).astype(k3a_band_2p5m.dtype)

    return MTFResult(
        band_name=band_name,
        best_sigma=best_sigma,
        min_mse=min_mse,
        simulated_band_2p5m=out_simulated
    )


def simulate_mtf_all_bands(
    img_k3a_2p5m: np.ndarray,
    img_s2_10m: np.ndarray,
    band_names: List[str],
    sigma_min: float = 0.05,
    sigma_max: float = 9.0,
    sigma_step: float = 0.01,  # 호환성을 위해 남겨두나 최적화 모드에서는 무시됨
    nodata_value: float = 0.0
) -> Tuple[np.ndarray, List[MTFResult]]:
    """다중 밴드 위성 영상에 대해 밴드별로 최적의 PSF를 찾아 MTF 모사를 수행합니다.
    (최적화 알고리즘 기반 10배 속도 향상 적용됨)
    """
    if img_k3a_2p5m.shape[0] != img_s2_10m.shape[0] or img_k3a_2p5m.shape[0] != len(band_names):
        raise ValueError("밴드 수 불일치")

    simulated_k3a = np.zeros_like(img_k3a_2p5m)
    results = []
    
    logger.info("MTF 모사 시작 (최적화 모드): sigma 탐색 범위 %.2f ~ %.2f", sigma_min, sigma_max)
    
    for i, band_name in enumerate(band_names):
        res = estimate_relative_psf_per_band(
            img_k3a_2p5m[i],
            img_s2_10m[i],
            band_name,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            nodata_value=nodata_value
        )
        simulated_k3a[i] = res.simulated_band_2p5m
        results.append(res)
        
    return simulated_k3a, results
