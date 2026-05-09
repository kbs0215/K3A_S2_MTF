"""
dem_download.py - COP-DEM GLO-30 타일 다운로드 모듈

K3A 씬 BBOX를 기준으로 필요한 1°×1° 타일을 계산하고
AWS S3 공개 버킷에서 직접 다운로드합니다 (인증 불필요).
타일이 여러 장이면 rasterio로 병합하여 단일 GeoTIFF를 반환합니다.
"""

import json
import logging
import math
from pathlib import Path
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

_S3_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _PROJECT_ROOT / "config" / "paths.json"


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

def _get_dem_raw_dir() -> Path:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        paths = json.load(f)
    return _PROJECT_ROOT / paths.get("dem_raw_dir", "data/raw/dem")


# ──────────────────────────────────────────────
# 타일 계산
# ──────────────────────────────────────────────

def _tile_name(lat: int, lon: int) -> str:
    """COP-DEM GLO-30 표준 타일 이름을 반환합니다."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"


def _tiles_for_bbox(
    bbox: tuple[float, float, float, float],
) -> List[tuple[int, int]]:
    """BBOX에 걸치는 모든 1°×1° 타일의 (lat, lon) 목록을 반환합니다."""
    min_lon, min_lat, max_lon, max_lat = bbox
    tiles = []
    for lat in range(math.floor(min_lat), math.ceil(max_lat)):
        for lon in range(math.floor(min_lon), math.ceil(max_lon)):
            tiles.append((lat, lon))
    return tiles


# ──────────────────────────────────────────────
# 단일 타일 다운로드
# ──────────────────────────────────────────────

_MIN_TILE_BYTES = 1_000_000  # 정상 COP-DEM 타일은 최소 수십 MB


def download_dem_tile(
    lat: int,
    lon: int,
    output_dir: Path,
) -> Optional[Path]:
    """AWS S3에서 단일 COP-DEM GLO-30 타일을 다운로드합니다.

    Returns:
        저장된 GeoTIFF Path, 해당 타일이 없으면(바다 등) None.
    """
    name = _tile_name(lat, lon)
    out_path = output_dir / f"{name}.tif"

    if out_path.exists():
        if out_path.stat().st_size >= _MIN_TILE_BYTES:
            logger.info("타일 이미 존재, 스킵: %s", out_path.name)
            return out_path
        logger.warning("불완전한 타일 감지 (%.0f bytes), 재다운로드: %s",
                       out_path.stat().st_size, out_path.name)
        out_path.unlink()

    url = f"{_S3_BASE}/{name}/{name}.tif"
    logger.info("DEM 타일 다운로드: %s", name)

    try:
        response = requests.get(url, stream=True, timeout=300)
        if response.status_code == 404:
            logger.warning("타일 없음 (바다 또는 무데이터 구역): %s", name)
            return None
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error("타일 다운로드 실패: %s – %s", name, e)
        raise

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    logger.info("타일 저장 완료: %s (%.1f MB)", out_path.name, out_path.stat().st_size / 1e6)
    return out_path


# ──────────────────────────────────────────────
# 병합
# ──────────────────────────────────────────────

def _merge_tiles(tile_paths: List[Path], output_path: Path) -> Path:
    """여러 DEM 타일을 하나의 GeoTIFF로 병합합니다."""
    import rasterio
    from rasterio.merge import merge

    datasets = [rasterio.open(p) for p in tile_paths]
    mosaic, transform = merge(datasets)
    profile = datasets[0].profile.copy()
    profile.update(
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        transform=transform,
    )
    for ds in datasets:
        ds.close()

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mosaic)

    logger.info("DEM 병합 완료: %s", output_path.name)
    return output_path


# ──────────────────────────────────────────────
# 파이프라인용 일체형 함수
# ──────────────────────────────────────────────

def download_dem_for_k3a(
    k3a_bbox: tuple[float, float, float, float],
    output_dir: Optional[str | Path] = None,
    scene_name: Optional[str] = None,
) -> Optional[Path]:
    """K3A 씬 BBOX에 맞는 COP-DEM GLO-30 타일을 다운로드합니다.

    타일이 여러 장이면 자동으로 병합하여 단일 GeoTIFF를 반환합니다.

    Args:
        k3a_bbox: (min_lon, min_lat, max_lon, max_lat).
        output_dir: 저장 디렉토리. None이면 config에서 자동 로딩.
        scene_name: K3A 씬 이름. 제공 시 출력 파일명에 포함됩니다.

    Returns:
        단일 DEM GeoTIFF Path, 실패 시 None.
    """
    import shutil

    if output_dir is None:
        output_dir = _get_dem_raw_dir()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tiles = _tiles_for_bbox(k3a_bbox)
    logger.info("K3A DEM: bbox=%s → 타일 %d개 %s", k3a_bbox, len(tiles), tiles)

    # scene_name이 있으면 최종 출력 경로를 먼저 확인 (캐시)
    if scene_name:
        output_path = output_dir / f"dem_{scene_name}.tif"
        if output_path.exists() and output_path.stat().st_size >= _MIN_TILE_BYTES:
            logger.info("DEM 이미 존재, 스킵: %s", output_path.name)
            return output_path

    tile_paths: List[Path] = []
    for lat, lon in tiles:
        path = download_dem_tile(lat, lon, output_dir)
        if path:
            tile_paths.append(path)

    if not tile_paths:
        logger.warning("다운로드된 DEM 타일이 없습니다.")
        return None

    if scene_name:
        output_path = output_dir / f"dem_{scene_name}.tif"
        if len(tile_paths) == 1:
            shutil.copy2(tile_paths[0], output_path)
            logger.info("DEM 저장: %s", output_path.name)
            return output_path
        return _merge_tiles(tile_paths, output_path)

    # scene_name 없을 때 기존 방식
    if len(tile_paths) == 1:
        return tile_paths[0]

    min_lon, min_lat = k3a_bbox[0], k3a_bbox[1]
    merged_path = output_dir / f"dem_merged_{min_lat:.3f}_{min_lon:.3f}.tif"
    if merged_path.exists() and merged_path.stat().st_size >= _MIN_TILE_BYTES:
        logger.info("병합 파일 이미 존재, 스킵: %s", merged_path.name)
        return merged_path

    return _merge_tiles(tile_paths, merged_path)


# ──────────────────────────────────────────────
# CLI 실행부
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import os
    import sys

    # PostgreSQL PostGIS의 구 버전 PROJ가 환경변수를 오염시키는 문제 방지
    try:
        import pyproj
        os.environ["PROJ_DATA"] = pyproj.datadir.get_data_dir()
    except Exception:
        pass

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT))

    from module1_data_download.src.k3a_loader import load_k3a_scene

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    main_logger = logging.getLogger("dem_download.main")

    parser = argparse.ArgumentParser(
        description="K3A 전체 씬 기준 COP-DEM GLO-30 타일 다운로드",
    )
    parser.add_argument(
        "--k3a-dir",
        default="data/interim/k3a_extracted",
        help="K3A 씬 디렉토리 경로 (기본값: data/interim/k3a_extracted)",
    )
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="필요한 타일 목록만 출력하고 다운로드는 건너뜁니다",
    )
    args = parser.parse_args()

    k3a_base = PROJECT_ROOT / args.k3a_dir
    if not k3a_base.exists():
        main_logger.error("K3A 디렉토리가 존재하지 않습니다: %s", k3a_base)
        sys.exit(1)

    scene_dirs = sorted([
        d for d in k3a_base.iterdir()
        if d.is_dir() and d.name.startswith("K3A_")
    ])

    if not scene_dirs:
        main_logger.error("K3A 씬을 찾을 수 없습니다: %s", k3a_base)
        sys.exit(1)

    main_logger.info("=" * 60)
    main_logger.info("K3A 씬 %d개 발견 → COP-DEM 다운로드 시작", len(scene_dirs))
    main_logger.info("=" * 60)

    downloaded = 0
    skipped = 0
    failed = 0

    for idx, scene_dir in enumerate(scene_dirs, 1):
        main_logger.info("")
        main_logger.info("─" * 60)
        main_logger.info("[%d/%d] %s", idx, len(scene_dirs), scene_dir.name)
        main_logger.info("─" * 60)

        try:
            scene = load_k3a_scene(scene_dir)
        except Exception as e:
            main_logger.error("  씬 로딩 실패: %s", e)
            failed += 1
            continue

        if scene.bounds is None:
            main_logger.warning("  좌표 정보 없음 → 스킵")
            skipped += 1
            continue

        bbox = scene.bounds.to_bbox()
        tiles = _tiles_for_bbox(bbox)
        main_logger.info("  BBOX: %s", bbox)
        main_logger.info("  필요 타일 (%d개):", len(tiles))
        for lat, lon in tiles:
            main_logger.info("    %s", _tile_name(lat, lon))

        if args.search_only:
            continue

        try:
            dem_path = download_dem_for_k3a(bbox, scene_name=scene_dir.name)
            if dem_path:
                main_logger.info("  저장 완료: %s", dem_path)
                downloaded += 1
            else:
                main_logger.warning("  DEM 없음 (바다 구역일 수 있음)")
                skipped += 1
        except Exception as e:
            main_logger.error("  다운로드 실패: %s", e)
            failed += 1

    print(f"\n{'═'*60}")
    print(f"  COP-DEM 다운로드 전체 요약")
    print(f"{'═'*60}")
    print(f"  전체 씬        : {len(scene_dirs)}개")
    print(f"  다운로드 완료  : {downloaded}개")
    print(f"  스킵           : {skipped}개")
    print(f"  실패           : {failed}개")
    print(f"{'═'*60}")
