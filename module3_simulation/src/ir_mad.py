"""ir_mad.py - IR-MAD (Iteratively Reweighted MAD) PIF 탐지

두 시기/센서 영상 간 정준상관분석(CCA) 을 가중치를 갱신하며 반복적으로
수행해 불변 픽셀(PIFs, Pseudo-Invariant Features) 을 자동 탐지한다.
χ² CDF 보수로 픽셀별 불변 확률을 산출하고, 임계값 이상이면 PIF 로 분류.

참고: Nielsen, A.A. (2007). The Regularized Iteratively Reweighted MAD
Method for Change Detection in Multi- and Hyperspectral Data.
IEEE Trans. Image Processing, 16(2), 463-478.
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy import linalg, stats

logger = logging.getLogger(__name__)


@dataclass
class IRMADResult:
    """IR-MAD 알고리즘 결과"""
    mad_variates: np.ndarray             # MAD 변량 (bands, H, W)
    chi2: np.ndarray                     # 카이제곱 통계량 (H, W)
    no_change_prob: np.ndarray           # 불변 확률 (H, W), 0~1
    canonical_correlations: np.ndarray   # 정준 상관 계수
    iterations: int                      # 수렴까지 반복 횟수
    pif_mask: np.ndarray                 # PIF 마스크 (H, W), True=PIF


def _weighted_covariance(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """가중 공분산 행렬 (Σxx, Σyy, Σxy) 을 계산합니다.

    Args:
        x, y:    (bands, pixels)
        weights: (pixels,)
    """
    w_sum = weights.sum()

    x_mean = (x * weights[np.newaxis, :]).sum(axis=1) / w_sum
    y_mean = (y * weights[np.newaxis, :]).sum(axis=1) / w_sum

    x_centered = x - x_mean[:, np.newaxis]
    y_centered = y - y_mean[:, np.newaxis]

    wx = x_centered * np.sqrt(weights)[np.newaxis, :]
    wy = y_centered * np.sqrt(weights)[np.newaxis, :]

    sigma_xx = (wx @ wx.T) / w_sum
    sigma_yy = (wy @ wy.T) / w_sum
    sigma_xy = (wx @ wy.T) / w_sum

    return sigma_xx, sigma_yy, sigma_xy


def ir_mad(
    img_reference: np.ndarray,
    img_target: np.ndarray,
    max_iter: int = 50,
    epsilon: float = 1e-3,
    pif_threshold: float = 0.95,
    nodata_value: float | None = 0.0,
    min_valid_pixels: int = 1000,
) -> IRMADResult:
    """IR-MAD (Iteratively Reweighted MAD) 알고리즘을 수행합니다.

    Args:
        img_reference: 기준 영상 (bands, H, W) — S2
        img_target:    대상 영상 (bands, H, W) — K3A
        max_iter:      최대 반복 횟수
        epsilon:       수렴 판정 임계값 (정준상관계수 변화량)
        pif_threshold: PIF 판정 확률 임계값 (기본 0.95)
        nodata_value:  NoData 값 (해당 픽셀은 제외)
        min_valid_pixels: 유효 픽셀이 이 값 미만이면 ValueError 로 조기 종료.
            CCA 계산 단계에서 NaN 으로 폭발하기 전에 명확한 메시지로 끊는 가드.
    """
    n_bands, height, width = img_reference.shape
    n_pixels = height * width

    if img_target.shape[0] != n_bands:
        raise ValueError(
            f"밴드 수 불일치: reference={n_bands}, target={img_target.shape[0]}"
        )

    x = img_reference.reshape(n_bands, n_pixels).astype(np.float64)
    y = img_target.reshape(n_bands, n_pixels).astype(np.float64)

    if nodata_value is not None:
        valid_mask = np.all(x != nodata_value, axis=0) & np.all(y != nodata_value, axis=0)
    else:
        valid_mask = np.ones(n_pixels, dtype=bool)

    n_valid = int(valid_mask.sum())
    logger.info(
        "IR-MAD 시작: %d밴드, %dx%d, 유효 픽셀=%d/%d",
        n_bands, height, width, n_valid, n_pixels
    )

    if n_valid < min_valid_pixels:
        raise ValueError(
            f"IR-MAD 유효 픽셀 부족 (n_valid={n_valid} < min={min_valid_pixels}). "
            f"K3A·S2 grid 가 정합 후 거의 NoData 거나 겹치는 영역이 너무 작습니다. "
            f"입력 페어의 grid 파일을 확인하거나 다른 S2 SAFE 와 매칭을 시도하세요."
        )

    x_valid = x[:, valid_mask]
    y_valid = y[:, valid_mask]

    weights = np.ones(n_valid)
    old_rho = np.zeros(n_bands)

    for iteration in range(1, max_iter + 1):
        # 1. 가중 공분산
        sigma_xx, sigma_yy, sigma_xy = _weighted_covariance(x_valid, y_valid, weights)

        # 수치 안정화
        reg = 1e-10 * np.eye(n_bands)
        sigma_xx += reg
        sigma_yy += reg

        # 2. CCA: A = Σxy Σyy⁻¹ Σyx, B = Σxx
        sigma_yy_inv = linalg.inv(sigma_yy)
        A = sigma_xy @ sigma_yy_inv @ sigma_xy.T
        B = sigma_xx
        eigenvalues, eigenvectors_x = linalg.eigh(A, B)

        # 작은 상관 → 큰 상관 순 정렬 (MAD 는 상관이 작은 쪽이 변화 신호)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors_x = eigenvectors_x[:, idx]

        rho = np.sqrt(np.clip(eigenvalues, 0, 1))

        eigenvectors_y = sigma_yy_inv @ sigma_xy.T @ eigenvectors_x
        for i in range(n_bands):
            norm = np.sqrt(eigenvectors_y[:, i] @ sigma_yy @ eigenvectors_y[:, i])
            if norm > 1e-10:
                eigenvectors_y[:, i] /= norm

        # 3. MAD = aᵀX − bᵀY (정준 변량의 차)
        w_sum = weights.sum()
        x_mean = (x_valid * weights[np.newaxis, :]).sum(axis=1) / w_sum
        y_mean = (y_valid * weights[np.newaxis, :]).sum(axis=1) / w_sum
        u = eigenvectors_x.T @ (x_valid - x_mean[:, np.newaxis])
        v = eigenvectors_y.T @ (y_valid - y_mean[:, np.newaxis])
        mad = u - v

        # 4. χ² = Σ(MADᵢ² / σ²ᵢ),  σ²ᵢ = 2(1 − ρᵢ)
        sigma2_mad = np.maximum(2 * (1 - rho), 1e-10)
        chi2_valid = np.sum(mad ** 2 / sigma2_mad[:, np.newaxis], axis=0)

        # 5. 불변 확률 (χ² CDF 보수)
        no_change_prob_valid = 1 - stats.chi2.cdf(chi2_valid, df=n_bands)

        # 6. 가중치 갱신 = 불변 확률
        weights = no_change_prob_valid

        # 7. 수렴 판정
        delta_rho = np.max(np.abs(rho - old_rho))
        logger.info("  반복 %d: rho=%s, Δrho=%.6f",
                    iteration, np.round(rho, 4), delta_rho)
        if delta_rho < epsilon:
            logger.info("수렴 완료 (반복 %d회)", iteration)
            break
        old_rho = rho.copy()
    else:
        logger.warning("최대 반복(%d)에 도달, 수렴하지 않음", max_iter)

    # 결과를 전체 이미지 크기로 복원
    mad_full = np.zeros((n_bands, n_pixels), dtype=np.float64)
    chi2_full = np.zeros(n_pixels, dtype=np.float64)
    prob_full = np.zeros(n_pixels, dtype=np.float64)
    mad_full[:, valid_mask] = mad
    chi2_full[valid_mask] = chi2_valid
    prob_full[valid_mask] = no_change_prob_valid

    pif_mask = prob_full >= pif_threshold
    n_pifs = pif_mask.sum()
    pif_ratio = n_pifs / n_valid * 100 if n_valid > 0 else 0
    logger.info("IR-MAD 완료: PIF=%d개 (%.1f%%), rho=%s",
                n_pifs, pif_ratio, np.round(rho, 4))

    return IRMADResult(
        mad_variates=mad_full.reshape(n_bands, height, width),
        chi2=chi2_full.reshape(height, width),
        no_change_prob=prob_full.reshape(height, width),
        canonical_correlations=rho,
        iterations=iteration,
        pif_mask=pif_mask.reshape(height, width),
    )
