"""
run_coregistration.py - K3A 전체 씬 → S2 공간 정합 자동화 파이프라인 (Batch)

전체 K3A 폴더 구조를 순회하며 자동으로 S2 영상과 매칭 및 정합하는 배치 파이프라인.
코페르니쿠스 API 접속 없이 로컬(data/raw/s2)에 다운로드된 S2 파일들 중에서
날짜가 가장 가까운 영상을 찾아 자동으로 매칭합니다.

사용법:
    python scripts/run_coregistration.py
    python scripts/run_coregistration.py --bands Green Red
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import rasterio
from rasterio.warp import transform_bounds

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from module1_data_download.src.k3a_loader import load_k3a_scene
from module1_data_download.src.dem_download import download_dem_for_k3a
from module2_coregistration.src import run_coregistration_pipeline

# config/paths.json 로드 (PROJECT_ROOT 기준 상대경로 → 절대경로 해석)
_PATHS_CONFIG = json.loads((PROJECT_ROOT / "config" / "paths.json").read_text(encoding="utf-8"))

def _path_from_config(key: str) -> Path:
    return (PROJECT_ROOT / _PATHS_CONFIG[key]).resolve()

# K3A 밴드별 S2 매칭 밴드 (10m 해상도 밴드 위주)
S2_BAND_MAP = {
    "Blue": "B02",
    "Green": "B03",
    "Red": "B04",
    "NIR": "B08",
}

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

def find_dem_tif(dem_dir: Path) -> Path | None:
    if dem_dir is None:
        return None
    for ext in ("*.tif", "*.tiff", "*.TIF"):
        matches = list(dem_dir.rglob(ext))
        if matches:
            return matches[0]
    return None

def find_s2_band_jp2(s2_safe_dir: Path, s2_band_name: str) -> Path | None:
    """S2 .SAFE 디렉토리 내에서 특정 밴드(예: B02)의 .jp2 파일을 재귀적으로 찾습니다."""
    matches = list(s2_safe_dir.rglob(f"*_{s2_band_name}.jp2"))
    if not matches:
        matches = list(s2_safe_dir.rglob(f"*{s2_band_name}*.jp2"))
    return matches[0] if matches else None

def get_s2_date(s2_safe_name: str) -> datetime:
    """S2 폴더명에서 촬영 날짜를 추출합니다. (예: S2A_MSIL1C_20151021T...)"""
    try:
        date_str = s2_safe_name.split("_")[2][:8]
        return datetime.strptime(date_str, "%Y%m%d")
    except Exception:
        return datetime(2020, 1, 1)


def get_s2_bounds_wgs84(s2_safe_dir: Path, cache: dict) -> Optional[tuple]:
    """S2 SAFE 디렉토리의 footprint bbox 를 WGS84(lon/lat) 로 반환합니다.

    SAFE 안의 B02 jp2 한 장만 열어서 native CRS bounds 를 읽고 WGS84 로 변환.
    같은 SAFE 가 여러 K3A 씬에서 반복 조회되므로 cache dict 에 결과를 저장.
    실패하면 None.
    """
    if s2_safe_dir in cache:
        return cache[s2_safe_dir]

    jp2 = find_s2_band_jp2(s2_safe_dir, "B02")
    if jp2 is None:
        cache[s2_safe_dir] = None
        return None

    try:
        with rasterio.open(jp2) as src:
            wgs = transform_bounds(src.crs, "EPSG:4326",
                                   src.bounds.left, src.bounds.bottom,
                                   src.bounds.right, src.bounds.top,
                                   densify_pts=21)
        cache[s2_safe_dir] = wgs
        return wgs
    except Exception as e:
        logging.getLogger("batch_pipeline").warning(
            "  S2 bounds 읽기 실패 (%s): %s", s2_safe_dir.name, e
        )
        cache[s2_safe_dir] = None
        return None


def bbox_intersects(a: tuple, b: tuple) -> bool:
    """두 bbox (minx, miny, maxx, maxy) 가 겹치면 True."""
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def bbox_contains(outer: tuple, inner: tuple) -> bool:
    """outer bbox 가 inner bbox 를 완전히 포함하면 True (경계 포함).

    K3A↔S2 매칭 시 S2 한 타일이 K3A 전체를 덮어야 정합 가능 — K3A 가 MGRS
    타일 경계에 걸치면 한 S2 만으로는 부족하므로 그런 후보는 제외해야 한다.
    """
    return (outer[0] <= inner[0] and outer[1] <= inner[1]
            and outer[2] >= inner[2] and outer[3] >= inner[3])

def main():
    parser = argparse.ArgumentParser(
        description="K3A 전체 씬 → S2 공간 정합 자동화 파이프라인 (Batch - Offline Match)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--k3a-dir", default=None,
        help=f"K3A 씬들이 있는 루트 디렉토리 (기본: config/paths.json의 k3a_extracted_dir = {_PATHS_CONFIG['k3a_extracted_dir']})",
    )
    parser.add_argument(
        "--s2-dir", default=None,
        help=f"S2 .SAFE 파일들이 있는 로컬 디렉토리 (기본: config/paths.json의 s2_raw_dir = {_PATHS_CONFIG['s2_raw_dir']})",
    )
    parser.add_argument(
        "--bands", nargs="+", default=["Blue", "Green", "Red", "NIR"],
        help="처리할 K3A 밴드 리스트 (기본: Blue Green Red NIR)",
    )
    parser.add_argument(
        "--crs", default="EPSG:32652",
        help="목표 좌표계 (기본: EPSG:32652, UTM 52N)",
    )
    parser.add_argument(
        "--resolution", type=float, default=2.5,
        help="가상격자 해상도 (기본: 2.5m)",
    )
    parser.add_argument(
        "--skip-dem", action="store_true",
        help="DEM 다운로드 건너뛰기 (Z=0으로 정사보정)",
    )
    parser.add_argument(
        "--max-date-diff", type=int, default=30,
        help="K3A-S2 촬영일 차이 최대 허용일수 (기본: 30일). 초과하는 후보는 제외.",
    )

    args = parser.parse_args()
    setup_logging()
    logger = logging.getLogger("batch_pipeline")

    # 인자가 주어지면 그대로 쓰고(절대/상대 모두 허용 — 상대경로는 CWD 기준),
    # 비어 있으면 config/paths.json + PROJECT_ROOT 로 해석
    k3a_base_dir = Path(args.k3a_dir).resolve() if args.k3a_dir else _path_from_config("k3a_extracted_dir")
    s2_base_dir = Path(args.s2_dir).resolve() if args.s2_dir else _path_from_config("s2_raw_dir")

    if not k3a_base_dir.exists():
        logger.error("K3A 루트 디렉토리를 찾을 수 없습니다: %s", k3a_base_dir)
        sys.exit(1)
        
    s2_safes = list(s2_base_dir.glob("*.SAFE"))
    if not s2_safes:
        logger.error("S2 디렉토리에 .SAFE 폴더가 없습니다: %s", s2_base_dir)
        sys.exit(1)
    
    for b in args.bands:
        if b not in S2_BAND_MAP:
            logger.error("지원하지 않는 밴드입니다: %s. 가능한 밴드: %s", b, list(S2_BAND_MAP.keys()))
            sys.exit(1)

    scene_dirs = sorted([d for d in k3a_base_dir.iterdir() if d.is_dir() and d.name.startswith("K3A_")])
    if not scene_dirs:
        logger.error("K3A 씬 디렉토리를 찾을 수 없습니다: %s", k3a_base_dir)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("배치 정합 파이프라인 시작 (로컬 매칭): 총 %d개 K3A 씬 대기", len(scene_dirs))
    logger.info("대상 밴드: %s", ", ".join(args.bands))
    logger.info("보유 S2 영상 수: %d개", len(s2_safes))
    logger.info("=" * 60)

    success_cnt = 0
    skip_cnt = 0
    fail_cnt = 0

    # 같은 S2 SAFE 의 bounds 를 K3A 씬별로 반복해서 읽지 않도록 캐싱
    s2_bounds_cache: dict = {}

    for idx, scene_dir in enumerate(scene_dirs, 1):
        logger.info("\n[%d/%d] 씬 처리 시작: %s", idx, len(scene_dirs), scene_dir.name)
        
        try:
            # 1. K3A 씬 로드
            scene = load_k3a_scene(scene_dir)
            if scene.bounds is None:
                logger.warning("  좌표 정보 없음 -> 스킵")
                skip_cnt += 1
                continue
                
            bbox = scene.bounds.to_bbox()
            acq_date_str = scene.get_acquisition_date() or "20200101"
            acq_date = datetime.strptime(acq_date_str, "%Y%m%d")

            # 2. 오프라인 S2 매칭
            #    (a) S2 한 타일이 K3A 전체를 완전히 포함하는 SAFE 만 1차 필터
            #        (단순 교차로 통과시키면 K3A 일부가 S2 밖에 있는 페어가 만들어짐.
            #         K3A 가 MGRS 타일 경계에 걸친 케이스는 다중 타일 모자이크가 필요하나
            #         현재는 미구현 → 그런 씬은 skip 으로 처리.)
            #    (b) 촬영일 차이가 max_date_diff 이내인 후보만 2차 필터
            #    (c) 남은 후보를 날짜 차이 오름차순으로 정렬
            s2_containing = []
            s2_partial_only = []  # 진단용: 일부만 겹치는 후보 수 카운트
            for s2_path in s2_safes:
                s2_bbox = get_s2_bounds_wgs84(s2_path, s2_bounds_cache)
                if s2_bbox is None:
                    continue
                if bbox_contains(s2_bbox, bbox):
                    s2_containing.append(s2_path)
                elif bbox_intersects(bbox, s2_bbox):
                    s2_partial_only.append(s2_path)

            if not s2_containing:
                if s2_partial_only:
                    logger.warning(
                        "  K3A 전체를 덮는 S2 타일이 없습니다 (부분 겹침 %d개 존재 → "
                        "K3A 가 MGRS 타일 경계에 걸침, 다중 타일 모자이크 필요) -> 스킵",
                        len(s2_partial_only),
                    )
                else:
                    logger.warning(
                        "  공간적으로 겹치는 S2 영상이 없습니다 -> 스킵 (K3A bbox=%s)", bbox,
                    )
                skip_cnt += 1
                continue

            s2_in_window = [
                p for p in s2_containing
                if abs((get_s2_date(p.name) - acq_date).days) <= args.max_date_diff
            ]

            if not s2_in_window:
                logger.warning(
                    "  K3A 를 완전히 덮는 S2 후보 %d개 모두 날짜 차이 %d일 초과 -> 스킵",
                    len(s2_containing), args.max_date_diff,
                )
                skip_cnt += 1
                continue

            logger.info(
                "  매칭 후보: %d개 (전체 %d개 중 K3A 전체 포함 %d개, 그중 ±%d일 이내)",
                len(s2_in_window), len(s2_safes), len(s2_containing), args.max_date_diff,
            )
            s2_candidates = sorted(
                s2_in_window,
                key=lambda p: abs((get_s2_date(p.name) - acq_date).days),
            )
            
            # 3. DEM 확보 (하나의 씬에 대해 DEM은 공통으로 사용)
            dem_tif = None
            if args.skip_dem:
                logger.info("  DEM 다운로드 스킵 (Z=0)")
            else:
                dem_tif = download_dem_for_k3a(bbox, scene_name=scene_dir.name)
                if dem_tif:
                    logger.info("  DEM 로드 완료: %s", dem_tif.name)

            # 후보 S2들을 순회 (첫번째 후보에서 영역 겹치지 않으면 다음 후보 시도)
            scene_success = False
            
            for s2_local_path in s2_candidates:
                diff_days = abs((get_s2_date(s2_local_path.name) - acq_date).days)
                logger.info("  S2 후보 시도: %s (날짜 차이: %d일)", s2_local_path.name, diff_days)
                
                bands_success = 0
                intersection_failed = False
                
                for band in args.bands:
                    if band not in scene.band_files:
                        logger.warning("  [%s] K3A 밴드 파일 누락 -> 밴드 스킵", band)
                        continue
                        
                    k3a_tif = scene.band_files[band]
                    s2_target_band = S2_BAND_MAP[band]

                    # S2 타겟 밴드 jp2 찾기
                    s2_jp2_path = find_s2_band_jp2(s2_local_path, s2_target_band)
                    if not s2_jp2_path:
                        logger.warning("  [%s] S2 폴더 내에서 밴드 %s(.jp2)를 찾을 수 없음 -> 밴드 스킵", band, s2_target_band)
                        continue

                    # 출력물 존재 여부 확인 (스킵 로직) — 페어 라벨(idx) 포함된 파일명으로 검사
                    pair_label = str(idx)
                    k3a_grid_expected = PROJECT_ROOT / "data/interim/grid" / f"{pair_label}_{k3a_tif.stem}_grid.tif"
                    s2_grid_expected = PROJECT_ROOT / "data/interim/grid" / f"{pair_label}_{s2_jp2_path.stem}_grid.tif"

                    if k3a_grid_expected.exists() and s2_grid_expected.exists():
                        logger.info("  [%s] 이미 정합된 결과물이 존재합니다 -> 스킵", band)
                        bands_success += 1
                        continue

                    # 파이프라인 실행
                    logger.info("  [%s] 공간 정합 파이프라인 실행 중... (페어 #%s)", band, pair_label)
                    try:
                        run_coregistration_pipeline(
                            k3a_tif_path=k3a_tif,
                            s2_tif_path=s2_jp2_path,
                            dem_path=dem_tif,
                            target_crs=args.crs,
                            resolution=args.resolution,
                            pair_label=pair_label,
                        )
                        bands_success += 1
                    except ValueError as ve:
                        if "겹치는 영역이 없습니다" in str(ve):
                            logger.warning("  [%s] 영역 불일치: S2 타일이 K3A 영역을 덮지 않습니다.", band)
                            intersection_failed = True
                            break # 현재 S2 후보로는 겹치지 않으므로 밴드 루프 중단하고 다음 S2 후보로 넘어감
                        else:
                            raise # 다른 ValueError는 그대로 발생시킴
                            
                if intersection_failed:
                    logger.info("  -> 다른 S2 후보를 시도합니다.\n")
                    continue # 다음 S2 후보 시도
                
                if bands_success > 0:
                    scene_success = True
                    logger.info("  씬 처리 완료! (성공 밴드 수: %d/%d)", bands_success, len(args.bands))
                    break # 성공했으므로 다음 S2 후보를 시도할 필요 없음
                else:
                    # 처리할 밴드가 아예 없었거나 모두 스킵된 경우
                    break 

            if scene_success:
                success_cnt += 1
            else:
                logger.warning("  사용 가능한 S2 영상을 찾지 못했거나 실패했습니다.")
                fail_cnt += 1
            
        except Exception as e:
            # 풀 traceback 을 함께 남겨야 어느 단계에서 났는지 식별 가능
            logger.exception("  처리 중 오류 발생 (%s): %s", type(e).__name__, e)
            fail_cnt += 1

    logger.info("=" * 60)
    logger.info("배치 정합 파이프라인 종료")
    logger.info("  성공(씬): %d", success_cnt)
    logger.info("  스킵(씬): %d", skip_cnt)
    logger.info("  실패(씬): %d", fail_cnt)
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
