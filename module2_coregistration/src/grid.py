"""grid.py - 공통 가상격자 생성 및 K3A/S2 격자 재투영

가상격자의 origin = K3A∩S2 교집합 bbox 의 좌상단 (minx, maxy) — 별도 snap
없이 그대로 사용. 그 origin 에서부터:
  - S2  격자: s2_resolution(10m)   간격으로 생성
  - K3A 격자: k3a_resolution(2.5m) 간격으로 생성 (S2 셀의 정수배 ratio 만큼 더 촘촘)
두 격자는 동일 origin 과 동일 outer rectangle 을 공유한다 (S2 셀 1개 = K3A 셀 4×4).
픽셀 수는 ceil 로 올려 교집합을 완전히 덮으며, 우/하단의 ≤s2_resolution 만큼은
교집합 밖으로 살짝 넘어갈 수 있고 그 영역은 nodata 로 채워진다.

K3A 와 S2 모두 자기 격자에 bilinear 로 보간되어 들어간다 (K3A → 2.5m 격자,
S2 → 10m 격자). reproject_k3a_to_grid 와 resample_s2_to_grid 는 입력 소스만
다른 두 reproject 호출이지만, 호출 단계에서의 의미(이미 정사보정된 K3A vs
원본 S2) 가 달라 별도 함수로 분리해 둔다.
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict

import rasterio
from rasterio.warp import Resampling, reproject

logger = logging.getLogger(__name__)


def create_virtual_grid(
    ortho_k3a_path: str | Path,
    s2_bounds: tuple[float, float, float, float],
    k3a_resolution: float = 2.5,
    s2_resolution: float = 10.0,
    crs: str = "EPSG:32652",
) -> Dict[str, Dict[str, Any]]:
    """공통 가상 격자 두 개 (K3A 2.5m + S2 10m) 를 생성합니다.

    K3A∩S2 교집합 bbox 의 좌상단 (minx, maxy) 을 그대로 가상격자의 origin 으로
    사용 (별도 snap 없음). 그 origin 에서 s2_resolution(10m) 간격으로 S2 격자,
    k3a_resolution(2.5m) 간격으로 K3A 격자를 만듭니다. 두 격자는 origin 과
    outer rect 을 공유하며 (S2 셀 1개 = K3A 셀 4×4), 픽셀 수는 ceil 로 올려
    교집합을 완전히 덮습니다. 우/하단 모서리에 ≤s2_resolution 의 extra 영역이
    포함될 수 있고 그곳은 K3A·S2 데이터 없으면 nodata.

    Args:
        ortho_k3a_path: 정사보정된 K3A GeoTIFF 경로.
        s2_bounds: S2 (minx, miny, maxx, maxy) — 동일 CRS 기준.
        k3a_resolution: K3A 격자 픽셀 해상도 (기본 2.5m).
        s2_resolution: S2 격자 픽셀 해상도 (기본 10m, S2 10m 밴드의 native 해상도).
            k3a_resolution 의 양의 정수배여야 함 (10/2.5=4 → S2 1셀 = K3A 4×4셀).
        crs: 목표 좌표계 (기본 UTM Zone 52N).

    Returns:
        {"k3a_profile": <2.5m profile>, "s2_profile": <10m profile>}
        두 프로필은 동일한 outer rect 과 origin (minx, maxy) 를 공유.
    """
    ratio = s2_resolution / k3a_resolution
    if ratio <= 0 or not math.isclose(ratio, round(ratio), rel_tol=1e-9):
        raise ValueError(
            f"s2_resolution ({s2_resolution}) 은 k3a_resolution ({k3a_resolution}) 의 "
            f"양의 정수배여야 합니다. 현재 ratio={ratio}"
        )
    ratio_int = int(round(ratio))

    with rasterio.open(ortho_k3a_path) as src:
        k3a_bounds = (src.bounds.left, src.bounds.bottom,
                      src.bounds.right, src.bounds.top)
        logger.info("정사보정된 K3A 범위: %s (CRS: %s)", k3a_bounds, src.crs)

    minx = max(k3a_bounds[0], s2_bounds[0])
    miny = max(k3a_bounds[1], s2_bounds[1])
    maxx = min(k3a_bounds[2], s2_bounds[2])
    maxy = min(k3a_bounds[3], s2_bounds[3])

    if minx >= maxx or miny >= maxy:
        raise ValueError(
            f"두 영상(K3A, S2) 간에 겹치는 영역이 없습니다.\n"
            f"  K3A bounds: {k3a_bounds}\n"
            f"  S2  bounds: {s2_bounds}"
        )

    # 가상격자 origin = 교집합의 좌상단 (snap 없음).
    # 그 origin 에서 S2 10m / K3A 2.5m 간격으로 ceil 만큼 셀을 깔아 교집합 전체를 덮음.
    # K3A 픽셀 수 = S2 픽셀 수 × ratio (=4) → 두 격자가 정확히 동일 outer rect 공유.
    width_s2 = int(math.ceil((maxx - minx) / s2_resolution))
    height_s2 = int(math.ceil((maxy - miny) / s2_resolution))
    width_k3a = width_s2 * ratio_int
    height_k3a = height_s2 * ratio_int

    transform_k3a = rasterio.transform.from_origin(minx, maxy, k3a_resolution, k3a_resolution)
    transform_s2 = rasterio.transform.from_origin(minx, maxy, s2_resolution, s2_resolution)

    base_profile = {
        "driver": "GTiff",
        "dtype": "uint16",
        "nodata": 0,
        "count": 1,
        "crs": rasterio.crs.CRS.from_string(crs),
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "lzw",
    }

    k3a_profile = {**base_profile, "width": width_k3a, "height": height_k3a, "transform": transform_k3a}
    s2_profile = {**base_profile, "width": width_s2, "height": height_s2, "transform": transform_s2}

    rect_maxx = minx + width_s2 * s2_resolution
    rect_miny = maxy - height_s2 * s2_resolution
    logger.info("공통 가상격자 생성 완료 (origin = 교집합 좌상단, snap 없음)")
    logger.info("  - origin: (%.3f, %.3f)  교집합 우/하단: (%.3f, %.3f)  격자 우/하단: (%.3f, %.3f)",
                minx, maxy, maxx, miny, rect_maxx, rect_miny)
    logger.info("  - K3A: %dx%d @ %.2fm", width_k3a, height_k3a, k3a_resolution)
    logger.info("  - S2 : %dx%d @ %.2fm", width_s2, height_s2, s2_resolution)
    return {"k3a_profile": k3a_profile, "s2_profile": s2_profile}


def reproject_k3a_to_grid(
    ortho_k3a_path: str | Path,
    grid_profile: Dict[str, Any],
    out_path: str | Path,
) -> Path:
    """정사보정된 K3A 를 가상 격자에 맞춰 재투영합니다.

    이미 정사보정이 완료되었기 때문에 RPC 는 사용하지 않습니다.
    Bilinear 리샘플링으로 가상 격자에 정렬합니다.

    Args:
        ortho_k3a_path: 정사보정된 K3A TIF 경로.
        grid_profile: create_virtual_grid() 가 반환한 dict 의 "k3a_profile" 항목.
        out_path: 저장할 결과 TIF 경로.

    Returns:
        저장된 K3A 가상격자 TIF 파일의 Path.
    """
    ortho_k3a_path = Path(ortho_k3a_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(ortho_k3a_path) as src:
        kwargs = grid_profile.copy()
        kwargs.update({
            "count": src.count,
            "dtype": src.dtypes[0],
        })

        with rasterio.open(out_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=grid_profile["transform"],
                    dst_crs=grid_profile["crs"],
                    resampling=Resampling.bilinear,
                    src_nodata=0,
                    dst_nodata=0,
                )

    logger.info("K3A → 가상격자 재투영 완료: %s", out_path)
    return out_path


def resample_s2_to_grid(
    s2_tif_path: str | Path,
    grid_profile: Dict[str, Any],
    out_path: str | Path,
) -> Path:
    """S2 영상을 10m 가상 격자에 bilinear 로 재투영합니다.

    가상격자 origin 이 교집합 좌상단이라 S2 native 10m 격자와 0~10m 만큼
    어긋날 수 있고, 그 어긋남이 bilinear 로 보간되어 출력 픽셀에 반영됩니다.
    (k3a_profile 을 넘기면 2.5m 격자에 oversample 도 가능)

    Args:
        s2_tif_path: S2 GeoTIFF 경로.
        grid_profile: create_virtual_grid() 가 반환한 dict 의 "s2_profile" 항목.
        out_path: 저장할 결과 TIF 경로.

    Returns:
        저장된 S2 리샘플링 TIF 파일의 Path.
    """
    s2_tif_path = Path(s2_tif_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(s2_tif_path) as src:
        kwargs = grid_profile.copy()
        kwargs.update({
            "count": src.count,
            "dtype": src.dtypes[0],  # S2 uint16 유지
        })

        with rasterio.open(out_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=grid_profile["transform"],
                    dst_crs=grid_profile["crs"],
                    resampling=Resampling.bilinear,
                    src_nodata=0,
                    dst_nodata=0,
                )

    logger.info("S2 리샘플링(Bilinear) 완료: %s", out_path)
    return out_path
