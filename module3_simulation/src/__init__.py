"""module3_simulation.src - K3A→S2 방사·MTF 모사 모듈

기능별 분할 (module2 패턴 따름):
- ir_mad.py:      IR-MAD (Iteratively Reweighted MAD) PIF 탐지
- linear_norm.py: PIF 기반 밴드별 선형 정규화 (fit / apply)
- pipeline.py:    IR-MAD + 선형 정규화 통합 파이프라인 (radiometric_simulation)

GDAL/PROJ 환경 설정은 shared.utils 에서 공통으로 처리 (import 시 side effect).
MTF 모사 (가우시안 필터 기반) 는 추후 본 모듈 안에 별도 파일로 추가 예정.
"""

from shared.utils import proj_env  # noqa: F401  GDAL 환경 초기화 (import 시 side effect)
from .ir_mad import IRMADResult, ir_mad
from .linear_norm import (
    BandNormCoeffs,
    apply_linear_normalization,
    fit_linear_normalization,
)
from .pipeline import (
    RadSimulationResult,
    print_normalization_report,
    radiometric_simulation,
)
from .mtf import (
    MTFResult,
    estimate_relative_psf_per_band,
    simulate_mtf_all_bands,
)

__all__ = [
    "ir_mad",
    "fit_linear_normalization",
    "apply_linear_normalization",
    "radiometric_simulation",
    "print_normalization_report",
    "IRMADResult",
    "BandNormCoeffs",
    "RadSimulationResult",
    "MTFResult",
    "estimate_relative_psf_per_band",
    "simulate_mtf_all_bands",
]
