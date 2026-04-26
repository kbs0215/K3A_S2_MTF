"""
raster_io.py - GeoTIFF 읽기/쓰기 공통 유틸리티

K3A 및 S2 위성영상의 래스터 데이터를 읽고 쓰는 공통 함수를 제공합니다.
rasterio 라이브러리를 기반으로 하며, 메타데이터(CRS, Transform, NoData 등)를
함께 관리합니다.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rasterio
from rasterio.transform import Affine

logger = logging.getLogger(__name__)


def read_geotiff(filepath: str | Path) -> dict[str, Any]:
    """
    GeoTIFF 파일을 읽어 배열과 메타데이터를 반환합니다.

    Args:
        filepath: GeoTIFF 파일 경로

    Returns:
        dict with keys:
            - 'data': numpy.ndarray (bands, height, width)
            - 'profile': rasterio profile (crs, transform, dtype 등)
            - 'bounds': 영상의 지리적 범위 (BoundingBox)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filepath}")

    with rasterio.open(filepath) as src:
        data = src.read()
        profile = dict(src.profile)
        bounds = src.bounds

    logger.info(
        "GeoTIFF 로딩 완료: %s | shape=%s | dtype=%s | CRS=%s",
        filepath.name, data.shape, data.dtype, profile.get("crs")
    )
    return {"data": data, "profile": profile, "bounds": bounds}


def write_geotiff(
    filepath: str | Path,
    data: np.ndarray,
    profile: dict[str, Any],
    overwrite: bool = False
) -> Path:
    """
    numpy 배열을 GeoTIFF 파일로 저장합니다.

    Args:
        filepath: 저장할 파일 경로
        data: numpy.ndarray (bands, height, width) 또는 (height, width)
        profile: rasterio profile dict
        overwrite: True면 기존 파일 덮어쓰기

    Returns:
        저장된 파일의 Path 객체
    """
    filepath = Path(filepath)
    if filepath.exists() and not overwrite:
        raise FileExistsError(f"파일이 이미 존재합니다: {filepath}. overwrite=True로 설정하세요.")

    filepath.parent.mkdir(parents=True, exist_ok=True)

    # 2D 배열이면 3D로 변환 (1, H, W)
    if data.ndim == 2:
        data = data[np.newaxis, :, :]

    write_profile = profile.copy()
    write_profile.update({
        "count": data.shape[0],
        "height": data.shape[1],
        "width": data.shape[2],
        "dtype": data.dtype.name,
    })

    with rasterio.open(filepath, "w", **write_profile) as dst:
        dst.write(data)

    logger.info(
        "GeoTIFF 저장 완료: %s | shape=%s | dtype=%s",
        filepath.name, data.shape, data.dtype
    )
    return filepath


def get_bounds_as_polygon(bounds: rasterio.coords.BoundingBox) -> list[tuple[float, float]]:
    """
    rasterio BoundingBox를 (lon, lat) 좌표 리스트로 변환합니다.
    Copernicus API 등에서 WKT/GeoJSON 폴리곤 생성에 사용합니다.

    Args:
        bounds: rasterio BoundingBox (left, bottom, right, top)

    Returns:
        [(lon, lat), ...] 형태의 폴리곤 좌표 리스트 (5개 점, 닫힌 폴리곤)
    """
    return [
        (bounds.left, bounds.top),      # 좌상
        (bounds.right, bounds.top),     # 우상
        (bounds.right, bounds.bottom),  # 우하
        (bounds.left, bounds.bottom),   # 좌하
        (bounds.left, bounds.top),      # 닫기
    ]
