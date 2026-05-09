"""
k3a_loader.py - KOMPSAT-3A 데이터 로딩 모듈

K3A L1R 데이터의 zip 압축 해제, 밴드별 GeoTIFF 로딩,
메타데이터(Aux.xml) 파싱, 좌표 범위 추출을 수행합니다.
data/raw/k3a/ 디렉토리의 zip 파일을 처리합니다.
"""

import json
import logging
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from module1_data_download.src.raster_io import read_geotiff

logger = logging.getLogger(__name__)

# K3A 밴드 접미사 매핑
K3A_BAND_SUFFIXES: dict[str, str] = {
    "Blue": "_B.tif",
    "Green": "_G.tif",
    "Red": "_R.tif",
    "NIR": "_N.tif",
    "PAN": "_P.tif",
}


@dataclass
class K3ABounds:
    """K3A 영상의 지리적 범위 (경위도)"""
    top_left: tuple[float, float]       # (lat, lon)
    top_right: tuple[float, float]
    bottom_left: tuple[float, float]
    bottom_right: tuple[float, float]

    def to_bbox(self) -> tuple[float, float, float, float]:
        """(min_lon, min_lat, max_lon, max_lat) 형태의 bbox 반환"""
        lats = [self.top_left[0], self.top_right[0],
                self.bottom_left[0], self.bottom_right[0]]
        lons = [self.top_left[1], self.top_right[1],
                self.bottom_left[1], self.bottom_right[1]]
        return (min(lons), min(lats), max(lons), max(lats))

    def to_wkt_polygon(self) -> str:
        """WKT POLYGON 문자열 반환 (Copernicus API 검색용)"""
        bbox = self.to_bbox()
        return (
            f"POLYGON(("
            f"{bbox[0]} {bbox[1]},"
            f"{bbox[2]} {bbox[1]},"
            f"{bbox[2]} {bbox[3]},"
            f"{bbox[0]} {bbox[3]},"
            f"{bbox[0]} {bbox[1]}"
            f"))"
        )


@dataclass
class K3AScene:
    """K3A 단일 씬(Scene) 데이터를 담는 클래스"""
    scene_id: str
    extract_dir: Path
    bounds: Optional[K3ABounds] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    band_files: dict[str, Path] = field(default_factory=dict)

    def get_acquisition_date(self) -> Optional[str]:
        """씬 ID에서 촬영 날짜를 추출합니다 (YYYYMMDD)"""
        # K3A_20170223044142_10582_00056713_L1R → 20170223
        parts = self.scene_id.split("_")
        if len(parts) >= 2:
            return parts[1][:8]
        return None


def extract_k3a_zip(
    zip_path: str | Path,
    extract_to: str | Path,
    overwrite: bool = False
) -> list[Path]:
    """
    K3A zip 파일을 압축 해제합니다.

    Args:
        zip_path: zip 파일 경로
        extract_to: 압축 해제 대상 디렉토리
        overwrite: True면 기존 파일 덮어쓰기

    Returns:
        압축 해제된 씬 디렉토리 목록
    """
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)

    if not zip_path.exists():
        raise FileNotFoundError(f"zip 파일을 찾을 수 없습니다: {zip_path}")

    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        # 씬 디렉토리 이름 추출 (최상위 폴더들)
        scene_dirs = set()
        for name in zf.namelist():
            top_dir = name.split("/")[0]
            if top_dir:
                scene_dirs.add(top_dir)

        # 이미 해제된 경우 스킵
        if not overwrite:
            existing = [extract_to / d for d in scene_dirs if (extract_to / d).exists()]
            if existing:
                logger.info(
                    "이미 압축 해제됨 (%d개 씬), 스킵합니다: %s",
                    len(existing), zip_path.name
                )
                return [extract_to / d for d in sorted(scene_dirs)]

        zf.extractall(extract_to)
        logger.info(
            "압축 해제 완료: %s → %s (%d개 씬)",
            zip_path.name, extract_to, len(scene_dirs)
        )

    return [extract_to / d for d in sorted(scene_dirs)]


def parse_coordinate_file(coord_file: Path) -> K3ABounds:
    """
    1RCoordinate.txt 파일을 파싱하여 영상 범위를 반환합니다.

    파일 형식:
        TOP_LEFT:37.610380,126.888221
        TOP_RIGHT:37.635614,127.042888
        BOTTOM_LEFT:37.493903,126.919001
        BOTTOM_RIGHT:37.519122,127.073425
    """
    coords: dict[str, tuple[float, float]] = {}
    text = coord_file.read_text(encoding="utf-8").strip()

    for line in text.splitlines():
        key, value = line.strip().split(":")
        lat_str, lon_str = value.split(",")
        coords[key] = (float(lat_str), float(lon_str))

    return K3ABounds(
        top_left=coords["TOP_LEFT"],
        top_right=coords["TOP_RIGHT"],
        bottom_left=coords["BOTTOM_LEFT"],
        bottom_right=coords["BOTTOM_RIGHT"],
    )


def parse_aux_xml(aux_file: Path) -> dict[str, Any]:
    """
    K3A Aux.xml 메타데이터 파일에서 주요 정보를 추출합니다.

    Returns:
        dict with keys like 'SatelliteID', 'AcquisitionDate', 'SunElevation' 등
    """
    tree = ET.parse(aux_file)
    root = tree.getroot()

    metadata: dict[str, Any] = {}

    # 주요 태그 추출 (K3A XML 구조에 맞게)
    tag_map = {
        "SatelliteID": str,
        "AcquisitionDate": str,
        "AcquisitionTime": str,
        "SunElevation": float,
        "SunAzimuth": float,
        "SatelliteElevation": float,
        "SatelliteAzimuth": float,
        "CloudCoverage": float,
    }

    for element in root.iter():
        tag_name = element.tag.split("}")[-1] if "}" in element.tag else element.tag
        if tag_name in tag_map and element.text:
            try:
                metadata[tag_name] = tag_map[tag_name](element.text.strip())
            except (ValueError, TypeError):
                metadata[tag_name] = element.text.strip()

    logger.info("Aux.xml 파싱 완료: %s", aux_file.name)
    return metadata


def discover_band_files(scene_dir: Path) -> dict[str, Path]:
    """
    씬 디렉토리에서 각 밴드 TIF 파일을 찾습니다.

    Returns:
        {'Blue': Path, 'Green': Path, ...} 형태의 딕셔너리
    """
    band_files: dict[str, Path] = {}

    for band_name, suffix in K3A_BAND_SUFFIXES.items():
        matches = list(scene_dir.glob(f"*{suffix}"))
        if matches:
            band_files[band_name] = matches[0]
        else:
            logger.warning("밴드 파일 없음: %s in %s", band_name, scene_dir.name)

    return band_files


def load_k3a_scene(scene_dir: str | Path) -> K3AScene:
    """
    K3A 씬 디렉토리에서 모든 정보를 로딩합니다.

    Args:
        scene_dir: 씬 디렉토리 경로

    Returns:
        K3AScene 객체 (밴드 파일 경로, 좌표 범위, 메타데이터 포함)
    """
    scene_dir = Path(scene_dir)
    if not scene_dir.is_dir():
        raise NotADirectoryError(f"씬 디렉토리가 아닙니다: {scene_dir}")

    scene_id = scene_dir.name
    scene = K3AScene(scene_id=scene_id, extract_dir=scene_dir)

    # 1. 좌표 파일 파싱
    coord_files = list(scene_dir.glob("*Coordinate.txt"))
    if coord_files:
        scene.bounds = parse_coordinate_file(coord_files[0])
        logger.info("좌표 범위: bbox=%s", scene.bounds.to_bbox())

    # 2. Aux.xml 메타데이터 파싱
    aux_files = list(scene_dir.glob("*Aux.xml"))
    if aux_files:
        scene.metadata = parse_aux_xml(aux_files[0])

    # 3. 밴드 파일 탐색
    scene.band_files = discover_band_files(scene_dir)
    logger.info(
        "K3A 씬 로딩 완료: %s | 밴드: %s",
        scene_id, list(scene.band_files.keys())
    )

    return scene


def load_k3a_band(scene: K3AScene, band_name: str) -> dict:
    """
    K3A 씬에서 특정 밴드의 래스터 데이터를 로딩합니다.

    Args:
        scene: K3AScene 객체
        band_name: 밴드 이름 ('Blue', 'Green', 'Red', 'NIR', 'PAN')

    Returns:
        raster_io.read_geotiff() 반환값 (data, profile, bounds)
    """
    if band_name not in scene.band_files:
        available = list(scene.band_files.keys())
        raise ValueError(
            f"밴드 '{band_name}'을 찾을 수 없습니다. 사용 가능: {available}"
        )

    return read_geotiff(scene.band_files[band_name])


def load_all_k3a_from_zip(
    zip_path: str | Path,
    extract_to: str | Path = "data/interim/k3a_extracted"
) -> list[K3AScene]:
    """
    K3A zip 파일에서 모든 씬을 압축 해제하고 로딩합니다.

    Args:
        zip_path: zip 파일 경로
        extract_to: 압축 해제 대상 디렉토리

    Returns:
        K3AScene 객체 리스트
    """
    scene_dirs = extract_k3a_zip(zip_path, extract_to)
    scenes = []

    for scene_dir in scene_dirs:
        try:
            scene = load_k3a_scene(scene_dir)
            scenes.append(scene)
        except Exception as e:
            logger.error("씬 로딩 실패: %s - %s", scene_dir.name, e)

    logger.info("총 %d개 씬 로딩 완료 from %s", len(scenes), Path(zip_path).name)
    return scenes

if __name__ == "__main__":
    import sys
    
    # 터미널에 진행 상황을 출력하기 위한 로깅 기본 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 실행 시 압축 파일 경로를 인자로 받습니다.
    if len(sys.argv) < 2:
        logger.warning("사용법: python -m src.data_access.k3a_loader <zip파일_경로>")
        logger.info("예시: python -m src.data_access.k3a_loader data/raw/K3A_sample.zip")
        # 테스트용으로 경로가 없을 때는 안내만 하고 정상 종료합니다.
        sys.exit(0)
        
    test_zip_path = sys.argv[1]
    
    try:
        logger.info(f"K3A 데이터 로딩을 시작합니다: {test_zip_path}")
        loaded_scenes = load_all_k3a_from_zip(test_zip_path)
        
        print("\n" + "="*40)
        print(f"로딩 결과 요약 (총 {len(loaded_scenes)}개 씬)")
        print("="*40)
        for i, sc in enumerate(loaded_scenes, 1):
            print(f"[{i}] 씬 ID: {sc.scene_id}")
            print(f"    촬영일자: {sc.get_acquisition_date()}")
            if sc.bounds:
                print(f"    좌표(BBOX): {sc.bounds.to_bbox()}")
            print(f"    확인된 밴드: {list(sc.band_files.keys())}\n")
            
    except Exception as e:
        logger.error(f"실행 중 오류가 발생했습니다: {e}")
