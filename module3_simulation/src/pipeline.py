"""pipeline.py - K3A→S2 방사모사 통합 파이프라인

IR-MAD (PIF 탐지) → 선형회귀 (fit) → K3A 적용 (apply) 을 한 번에 실행.
ir_mad.py + linear_norm.py 의 함수를 묶기만 한다.
"""

import logging
from dataclasses import dataclass

import numpy as np

from .ir_mad import IRMADResult, ir_mad
from .linear_norm import (
    BandNormCoeffs,
    apply_linear_normalization,
    fit_linear_normalization,
)

logger = logging.getLogger(__name__)


@dataclass
class RadSimulationResult:
    """방사모사 결과"""
    simulated_image: np.ndarray              # 정규화된 K3A (bands, H, W)
    band_coefficients: list[BandNormCoeffs]  # 밴드별 회귀 계수
    ir_mad_result: IRMADResult               # IR-MAD 결과 (진단용)
    band_names: list[str]                    # 밴드 이름 순서


def radiometric_simulation(
    img_s2: np.ndarray,
    img_k3a: np.ndarray,
    band_names: list[str] | None = None,
    mad_max_iter: int = 50,
    mad_epsilon: float = 1e-3,
    pif_threshold: float = 0.95,
    nodata_value: float | None = 0.0,
) -> RadSimulationResult:
    """IR-MAD 기반 선형 방사모사 (PIF 탐지 → fit → apply) 통합 실행.

    img_s2/img_k3a 는 동일 해상도/동일 grid 여야 합니다.
    호출 측에서 K3A → 10m 다운샘플 후 넘기는 것을 권장합니다.
    """
    n_bands = img_s2.shape[0]
    if band_names is None:
        band_names = [f"Band_{i+1}" for i in range(n_bands)]

    logger.info("=" * 60)
    logger.info("IR-MAD 기반 선형 방사모사 시작")
    logger.info("  S2 shape: %s, K3A shape: %s", img_s2.shape, img_k3a.shape)
    logger.info("  밴드: %s", band_names)
    logger.info("=" * 60)

    logger.info("[Step 1/3] IR-MAD 수행 중...")
    mad_result = ir_mad(
        img_reference=img_s2,
        img_target=img_k3a,
        max_iter=mad_max_iter,
        epsilon=mad_epsilon,
        pif_threshold=pif_threshold,
        nodata_value=nodata_value,
    )

    logger.info("[Step 2/3] PIF 기반 선형 회귀 학습 중...")
    coefficients = fit_linear_normalization(
        reference=img_s2,
        target=img_k3a,
        pif_mask=mad_result.pif_mask,
        band_names=band_names,
    )

    logger.info("[Step 3/3] 정규화 적용 중...")
    simulated = apply_linear_normalization(img_k3a, coefficients)

    logger.info("=" * 60)
    logger.info("방사모사 완료!")
    for c in coefficients:
        logger.info("  %s: a=%.6f, b=%.4f, R²=%.4f",
                    c.band_name, c.slope, c.intercept, c.r_squared)
    logger.info("=" * 60)

    return RadSimulationResult(
        simulated_image=simulated,
        band_coefficients=coefficients,
        ir_mad_result=mad_result,
        band_names=band_names,
    )


def print_normalization_report(result: RadSimulationResult) -> None:
    """방사모사 결과를 보기 좋게 출력합니다."""
    print("\n" + "=" * 70)
    print("IR-MAD 기반 선형 방사모사 결과")
    print("=" * 70)
    print(f"IR-MAD 반복 횟수: {result.ir_mad_result.iterations}")
    print(f"정준 상관 계수: {np.round(result.ir_mad_result.canonical_correlations, 4)}")
    print(f"PIF 픽셀 수: {result.ir_mad_result.pif_mask.sum()}")
    print()
    print(f"{'밴드':<10} {'기울기(a)':>12} {'절편(b)':>12} {'R²':>10} {'RMSE':>10}")
    print("-" * 56)
    for c in result.band_coefficients:
        print(f"{c.band_name:<10} {c.slope:>12.6f} {c.intercept:>12.4f} "
              f"{c.r_squared:>10.4f} {c.rmse:>10.4f}")
    print("=" * 70)
