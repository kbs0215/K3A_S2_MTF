"""shared.utils - 모듈 1~4 공용 유틸리티

- proj_env.py: GDAL/PROJ 환경 설정 (import 시 side effect) + PROJ_DATA 노출
- paths.py:    PROJECT_ROOT + config/paths.json 로딩 헬퍼
"""

from .paths import PROJECT_ROOT, load_paths, resolve_path
from .proj_env import PROJ_DATA

__all__ = ["PROJECT_ROOT", "PROJ_DATA", "load_paths", "resolve_path"]
