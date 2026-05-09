"""module2_coregistration.src - K3A↔S2 공간 정합 모듈

기능별 분할:
- rpc.py:      K3A _rpc.txt 사이드카 탐색/파싱
- ortho.py:    K3A RPC + DEM 정사보정 (gdal.Warp)
- grid.py:     가상격자 생성 + K3A/S2 격자 재투영
- pipeline.py: 위 단계를 묶은 통합 파이프라인 (run_coregistration_pipeline)

GDAL/PROJ 환경 설정 + paths 로딩은 shared.utils 에서 공통으로 처리.
"""

from shared.utils import proj_env  # noqa: F401  GDAL 환경 초기화 (import 시 side effect)
from .grid import create_virtual_grid, reproject_k3a_to_grid, resample_s2_to_grid
from .ortho import orthorectify_k3a_with_rpc
from .pipeline import run_coregistration_pipeline

__all__ = [
    "run_coregistration_pipeline",
    "orthorectify_k3a_with_rpc",
    "create_virtual_grid",
    "reproject_k3a_to_grid",
    "resample_s2_to_grid",
]
