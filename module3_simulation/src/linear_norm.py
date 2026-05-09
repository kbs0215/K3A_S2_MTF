"""linear_norm.py - PIF 기반 밴드별 선형 정규화

S2 = a × K3A + b 회귀를 PIF 픽셀에서 fit 하고 (`fit_linear_normalization`),
학습된 (a, b) 를 K3A 영상 전체에 적용 (`apply_linear_normalization`).

PIF 마스크는 ir_mad.py 의 IR-MAD 결과로부터 받아서 사용한다.
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


@dataclass
class BandNormCoeffs:
    """밴드별 선형 정규화 계수"""
    band_name: str
    slope: float       # 기울기 (a)
    intercept: float   # 절편 (b)
    r_squared: float   # 결정 계수 (R²)
    rmse: float        # RMSE
    n_pifs: int        # 사용된 PIF 개수


def fit_linear_normalization(
    reference: np.ndarray,
    target: np.ndarray,
    pif_mask: np.ndarray,
    band_names: list[str] | None = None,
    min_pifs: int = 100,
) -> list[BandNormCoeffs]:
    """PIF 픽셀에서 밴드별 선형회귀 (S2 = a × K3A + b) 계수를 학습합니다.

    Args:
        reference: 기준 영상 (bands, H, W) — S2
        target:    대상 영상 (bands, H, W) — K3A
        pif_mask:  PIF 마스크 (H, W), True=PIF
        band_names: 밴드 이름 리스트
        min_pifs:  최소 PIF 개수 (미달 시 경고)
    """
    n_bands = reference.shape[0]
    if band_names is None:
        band_names = [f"Band_{i+1}" for i in range(n_bands)]

    n_pifs = pif_mask.sum()
    if n_pifs < min_pifs:
        logger.warning(
            "PIF 개수(%d)가 최소 기준(%d)보다 적습니다. 결과 신뢰도가 낮을 수 있습니다.",
            n_pifs, min_pifs
        )

    coefficients: list[BandNormCoeffs] = []

    for i in range(n_bands):
        ref_pifs = reference[i][pif_mask].astype(np.float64)
        tgt_pifs = target[i][pif_mask].astype(np.float64)

        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(
            tgt_pifs, ref_pifs
        )

        predicted = slope * tgt_pifs + intercept
        rmse = np.sqrt(np.mean((ref_pifs - predicted) ** 2))

        coeff = BandNormCoeffs(
            band_name=band_names[i],
            slope=slope,
            intercept=intercept,
            r_squared=r_value ** 2,
            rmse=rmse,
            n_pifs=int(n_pifs),
        )
        coefficients.append(coeff)

        logger.info(
            "  %s: S2 = %.6f × K3A + %.4f | R²=%.4f | RMSE=%.4f | PIFs=%d",
            coeff.band_name, slope, intercept, coeff.r_squared, rmse, n_pifs
        )

    return coefficients


def apply_linear_normalization(
    target: np.ndarray,
    coefficients: list[BandNormCoeffs],
) -> np.ndarray:
    """학습된 (a, b) 를 target 영상 전체에 적용합니다.

    출력은 float64. 음수는 0 으로 clip. uint16 캐스팅은 호출 측에서 수행.
    """
    n_bands = target.shape[0]
    if len(coefficients) != n_bands:
        raise ValueError(
            f"계수 개수({len(coefficients)})와 밴드 수({n_bands}) 불일치"
        )

    simulated = np.zeros_like(target, dtype=np.float64)
    for i, coeff in enumerate(coefficients):
        simulated[i] = coeff.slope * target[i].astype(np.float64) + coeff.intercept
        simulated[i] = np.clip(simulated[i], 0, None)

    logger.info("선형 정규화 적용 완료: %d밴드", n_bands)
    return simulated
