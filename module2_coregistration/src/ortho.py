"""ortho.py - K3A 영상 RPC + DEM 정사보정

K3A 의 RPC 사이드카(_rpc.txt 또는 _M_rpc.txt) 를 GDAL 가 자동 인식할 수
있도록 stem 매칭 이름으로 정리한 뒤, gdal.Warp 으로 DEM 기반 정사보정을
수행합니다. RPC_DEM 은 transformerOptions 로 전달해야 실제 DEM 이 적용됩니다
(rasterio **kwargs 경로로 넘기면 GDAL 가 거부 → Z=0 으로 동작).
"""

import logging
import shutil
from pathlib import Path
from typing import List, Optional

import rasterio
from osgeo import gdal

from .rpc import _find_rpc_file

logger = logging.getLogger(__name__)


def orthorectify_k3a_with_rpc(
    k3a_tif_path: str | Path,
    out_path: str | Path,
    dem_path: Optional[str | Path] = None,
    target_crs: str = "EPSG:32652",
    target_resolution: float = 2.5,
) -> Path:
    """K3A 영상을 RPC + DEM 으로 정사보정합니다.

    가상격자에 직접 투영하는 것이 아니라, K3A 자체를 목표 CRS/해상도로
    먼저 정사보정합니다. 이후 이 결과물의 실제 범위를 읽어
    가상격자를 생성하는 데 사용합니다.

    Args:
        k3a_tif_path: 원본 K3A TIF 경로.
        out_path: 정사보정 결과 TIF 경로.
        dem_path: DEM GeoTIFF 파일 경로 (없으면 Z=0 으로 가정).
        target_crs: 목표 좌표계 (기본 UTM 52N).
        target_resolution: 출력 해상도 (기본 2.5m).

    Returns:
        저장된 정사보정 TIF 파일의 Path.
    """
    k3a_tif_path = Path(k3a_tif_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # RPC 존재 확인 — 원본 TIF 내장 RPC 가 없으면 _rpc.txt 를 외부 sidecar 로 인식시키기 위해
    # GDAL 가 자동 탐지하는 동일 디렉토리에 위치해 있는지 검증한다.
    with rasterio.open(k3a_tif_path) as src:
        has_internal_rpc = bool(src.rpcs)

    if not has_internal_rpc:
        rpc_file = _find_rpc_file(k3a_tif_path)
        if rpc_file is None:
            logger.error("RPC 정보를 찾을 수 없어 정사보정 불가: %s", k3a_tif_path)
            raise ValueError("RPC parameters not found.")

        # GDAL 은 {tif_stem}_rpc.txt 만 자동 인식한다.
        # K3A 멀티스펙트럼 통합 RPC(_M_rpc.txt) 는 stem 이 안 맞아 인식 못하므로,
        # 밴드별 stem-매칭 사이드카 이름으로 복사해둔다(파일이 없을 때만).
        # K3A 의 _M_rpc.txt 는 B/G/R/N 4개 밴드가 동일 RPC 를 공유.
        expected_sidecar = k3a_tif_path.with_name(k3a_tif_path.stem + "_rpc.txt")
        if rpc_file != expected_sidecar and not expected_sidecar.exists():
            shutil.copy2(rpc_file, expected_sidecar)
            logger.info("RPC sidecar 복사 (GDAL 자동 인식용): %s -> %s",
                        rpc_file.name, expected_sidecar.name)
        logger.info("_rpc.txt 외부 RPC 로드: %s", rpc_file.name)

    # GDAL Warp transformer options (RPC_DEM 은 transformerOptions 로 전달)
    transformer_options: List[str] = []
    if dem_path:
        logger.info("DEM 적용: %s", dem_path)
        transformer_options.append(f"RPC_DEM={Path(dem_path).as_posix()}")
        transformer_options.append("RPC_DEMINTERPOLATION=bilinear")
    else:
        logger.warning("DEM 미적용 – Z=0 으로 정사보정합니다.")

    warp_opts = gdal.WarpOptions(
        format="GTiff",
        dstSRS=target_crs,
        xRes=target_resolution,
        yRes=target_resolution,
        rpc=True,
        transformerOptions=transformer_options,
        resampleAlg="bilinear",
        multithread=True,
        creationOptions=[
            "TILED=YES",
            "BLOCKXSIZE=256",
            "BLOCKYSIZE=256",
            "COMPRESS=LZW",
        ],
    )

    ds = gdal.Warp(str(out_path), str(k3a_tif_path), options=warp_opts)
    if ds is None:
        raise RuntimeError(f"gdal.Warp 실패: {k3a_tif_path}")
    ds = None  # flush & close

    logger.info("K3A 정사보정 완료: %s", out_path)
    return out_path
