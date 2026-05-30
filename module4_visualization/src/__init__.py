"""module4_visualization - 논문용 figure 생성 모듈

src/ 의 공개 API:
  - locate_pair: label 기준으로 K3A/S2/Simulated 경로 묶음 탐색
  - read_rgb_stack: 다중 grid tif 를 RGB 스택으로 읽기 (윈도우 옵션)
  - read_simulated_rgb: simulated_final_2p5m.tif 의 R/G/B 밴드 스택
  - percentile_stretch: per-panel 또는 shared 퍼센타일 스트레치
  - aggregate_4x4_mean: K3A 2.5m -> 10m permissive 다운샘플 (module3 와 동일 로직)
  - bilinear_4x_downsample: 단순 보간 (bilinear) 기반 4x 다운샘플

GDAL/PROJ env 초기화는 import 시점에 자동 수행 (shared.utils.proj_env side effect).
"""

import shared.utils.proj_env  # noqa: F401  (GDAL_DATA / PROJ_DATA 환경설정 side effect)

from .grid_io import (
    PairPaths,
    locate_pair,
    read_rgb_stack,
    read_simulated_rgb,
)
from .stretch import percentile_stretch
from .resample import aggregate_4x4_mean, bilinear_4x_downsample

__all__ = [
    "PairPaths",
    "locate_pair",
    "read_rgb_stack",
    "read_simulated_rgb",
    "percentile_stretch",
    "aggregate_4x4_mean",
    "bilinear_4x_downsample",
]
