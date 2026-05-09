"""proj_env.py - GDAL/PROJ 환경 설정 (모든 모듈 공용)

이 모듈을 import 하면:
  - GDAL_PAM_ENABLED=NO  (.aux.xml PAM 사이드카 무시 → 한글 메시지 source 차단)
  - CPL_DEBUG=OFF        (디버그 로그 비활성화)
  - osgeo.gdal.UseExceptions() 활성화

그리고 PROJ_DATA 가 모듈 변수로 노출됩니다 — system PROJ (PostgreSQL/PostGIS
등 다른 설치) 가 잡혀 proj.db schema 버전 불일치(예: MINOR=2 vs ≥6 기대) 가
생기는 경우, rasterio 호출을 `rasterio.env.Env(PROJ_LIB=PROJ_DATA)` 로 감싸
rasterio 번들 PROJ 또는 conda PROJ 를 강제할 수 있도록.

CLAUDE.md §7.1(4)(5) 의 GDAL JP2/한글 메시지 이슈도 동일 환경변수로 완화됨.
"""

import os

# rasterio/osgeo import 보다 먼저 환경변수 설정해야 효과 있음
os.environ.setdefault("GDAL_PAM_ENABLED", "NO")
os.environ.setdefault("CPL_DEBUG", "OFF")

from pathlib import Path
from typing import Optional

import rasterio
from osgeo import gdal

gdal.UseExceptions()


def _find_proj_data() -> Optional[str]:
    """rasterio 번들 proj_data → CONDA_PREFIX 순으로 올바른 proj.db 경로를 반환합니다."""
    bundled = Path(rasterio.__file__).parent / "proj_data"
    if (bundled / "proj.db").exists():
        return str(bundled)
    prefix = os.environ.get("CONDA_PREFIX", "")
    if prefix:
        for p in [Path(prefix) / "Library" / "share" / "proj",
                  Path(prefix) / "share" / "proj"]:
            if (p / "proj.db").exists():
                return str(p)
    return None


PROJ_DATA: Optional[str] = _find_proj_data()
"""rasterio 번들 또는 conda 의 proj.db 가 들어 있는 디렉토리. 못 찾으면 None.

사용 예:
    import rasterio
    from shared.utils.proj_env import PROJ_DATA

    env_kwargs = {"PROJ_LIB": PROJ_DATA} if PROJ_DATA else {}
    with rasterio.env.Env(**env_kwargs):
        with rasterio.open(path) as src:
            ...
"""
