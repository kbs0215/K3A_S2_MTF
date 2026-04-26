"""
test_data_access.py - K3A 로딩 + S2 검색 통합 테스트

1. K3A zip 파일에서 씬 정보 로딩 (압축해제 없이 좌표만)
2. 좌표 기반 S2 영상 검색
"""

import logging
import sys
import zipfile
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def test_k3a_loading():
    """K3A zip에서 씬 로딩 테스트"""
    from src.data_access.k3a_loader import (
        extract_k3a_zip,
        load_k3a_scene,
    )

    zip_path = PROJECT_ROOT / "data" / "raw" / "k3a" / "1_Seoul.zip"
    extract_dir = PROJECT_ROOT / "data" / "interim" / "k3a_extracted"

    print("\n" + "=" * 60)
    print("1. K3A 데이터 로딩 테스트")
    print("=" * 60)

    # 압축 해제
    scene_dirs = extract_k3a_zip(zip_path, extract_dir)
    print(f"\n씬 개수: {len(scene_dirs)}")

    # 첫 번째 씬만 상세 로딩
    scene = load_k3a_scene(scene_dirs[0])

    print(f"\n[씬 정보]")
    print(f"  ID: {scene.scene_id}")
    print(f"  촬영일: {scene.get_acquisition_date()}")
    print(f"  밴드: {list(scene.band_files.keys())}")

    if scene.bounds:
        bbox = scene.bounds.to_bbox()
        print(f"  BBox: {bbox}")
        print(f"  WKT: {scene.bounds.to_wkt_polygon()[:80]}...")

    if scene.metadata:
        print(f"\n[메타데이터]")
        for k, v in scene.metadata.items():
            print(f"  {k}: {v}")

    return scene


def test_s2_search(scene):
    """S2 검색 테스트"""
    from src.data_access.s2_search import (
        search_s2_for_k3a_scene,
        print_search_results,
    )

    print("\n" + "=" * 60)
    print("2. Sentinel-2 검색 테스트")
    print("=" * 60)

    if not scene.bounds:
        print("ERROR: K3A 좌표 정보가 없습니다.")
        return

    bbox = scene.bounds.to_bbox()
    acq_date = scene.get_acquisition_date()

    print(f"\n검색 조건:")
    print(f"  K3A bbox: {bbox}")
    print(f"  K3A 촬영일: {acq_date}")
    print(f"  검색 범위: ±30일")
    print(f"  최대 구름량: 20%")

    results = search_s2_for_k3a_scene(
        k3a_bounds_bbox=bbox,
        k3a_acquisition_date=acq_date,
        date_range_days=30,
        max_cloud_cover=20.0,
    )

    print_search_results(results)
    return results


if __name__ == "__main__":
    scene = test_k3a_loading()
    results = test_s2_search(scene)
    print("\n테스트 완료!")
