"""
s2_download.py - Copernicus Data Space S2 검색·다운로드 통합 모듈

K3A 영상의 bbox를 기반으로 Sentinel-2 영상을 검색하고 다운로드합니다.
인증 정보는 .env 파일에서 로딩합니다.

사용법 (직접 실행 - 전체 K3A 씬 루프 처리):
    python -m src.data_access.s2_download
    python -m src.data_access.s2_download --search-only
    python -m src.data_access.s2_download --days 10 --cloud 10 --max 1
    python -m src.data_access.s2_download --k3a-dir data/interim/k3a_extracted
"""

import json
import logging
import os
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1"
DOWNLOAD_URL = "https://zipper.dataspace.copernicus.eu/odata/v1/Products"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

# ──────────────────────────────────────────────
# Config 유틸
# ──────────────────────────────────────────────
_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "paths.json"


def _load_paths() -> dict:
    """config/paths.json 에서 경로 설정을 읽어 반환합니다."""
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_s2_raw_dir() -> Path:
    """S2 원본 다운로드 디렉토리를 반환합니다."""
    paths = _load_paths()
    return Path(paths.get("s2_raw_dir", "data/raw/s2"))


# ──────────────────────────────────────────────
# 인증 (Authentication)
# ──────────────────────────────────────────────

def get_access_token() -> str:
    """
    .env 파일의 Copernicus 계정으로 액세스 토큰을 발급받습니다.

    Raises:
        EnvironmentError: .env에 인증 정보 없을 때
        requests.HTTPError: 인증 실패 시
    """
    load_dotenv()
    username = os.getenv("COPERNICUS_USERNAME")
    password = os.getenv("COPERNICUS_PASSWORD")

    if not username or not password:
        raise EnvironmentError(
            "COPERNICUS_USERNAME, COPERNICUS_PASSWORD가 .env에 없습니다. "
        )

    response = requests.post(
        TOKEN_URL,
        data={
            "client_id": "cdse-public",
            "grant_type": "password",
            "username": username,
            "password": password,
        },
        timeout=30,
    )
    response.raise_for_status()
    logger.info("Copernicus 액세스 토큰 발급 성공")
    return response.json()["access_token"]


# ──────────────────────────────────────────────
# 검색 (Search)
# ──────────────────────────────────────────────

@dataclass
class S2SearchResult:
    """Sentinel-2 검색 결과 단일 항목"""
    product_id: str
    name: str
    acquisition_date: str
    cloud_cover: float
    footprint: str
    processing_level: str
    online: bool
    size_mb: float


def search_s2_by_bbox(
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    max_cloud_cover: float = 10.0,
    processing_level: str = "S2MSI1C",
    max_results: int = 50,
) -> list[S2SearchResult]:
    """
    bbox 영역과 기간으로 Sentinel-2 영상을 검색합니다.

    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat)
        start_date: 시작일 (YYYY-MM-DD)
        end_date: 종료일 (YYYY-MM-DD)
        max_cloud_cover: 최대 구름량 (%)
        processing_level: 'S2MSI1C' (L1C) 또는 'S2MSI2A' (L2A)
        max_results: 최대 검색 결과 수

    Returns:
        S2SearchResult 리스트 (구름량 오름차순 정렬)
    """
    min_lon = round(bbox[0], 6)
    min_lat = round(bbox[1], 6)
    max_lon = round(bbox[2], 6)
    max_lat = round(bbox[3], 6)

    footprint_filter = (
        f"OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(("
        f"{min_lon} {min_lat},{max_lon} {min_lat},"
        f"{max_lon} {max_lat},{min_lon} {max_lat},"
        f"{min_lon} {min_lat}))')"
    )
    date_filter = (
        f"ContentDate/Start gt {start_date}T00:00:00.000Z and "
        f"ContentDate/Start lt {end_date}T23:59:59.999Z"
    )
    cloud_filter = (
        f"Attributes/OData.CSC.DoubleAttribute/any("
        f"att:att/Name eq 'cloudCover' and "
        f"att/OData.CSC.DoubleAttribute/Value le {max_cloud_cover})"
    )
    # Collection/Name 은 'SENTINEL-2' 이어야 함 (S2MSI1C 는 product type)
    collection_filter = "Collection/Name eq 'SENTINEL-2'"
    # product type 필터 (S2MSI1C = L1C, S2MSI2A = L2A)
    product_type_filter = (
        f"Attributes/OData.CSC.StringAttribute/any("
        f"att:att/Name eq 'productType' and "
        f"att/OData.CSC.StringAttribute/Value eq '{processing_level}')"
    )
    full_filter = " and ".join([
        collection_filter, footprint_filter, date_filter,
        cloud_filter, product_type_filter,
    ])

    params = {
        "$filter": full_filter,
        "$orderby": "ContentDate/Start desc",
        "$top": max_results,
        "$expand": "Attributes",
    }

    logger.info("S2 검색: bbox=%s, 기간=%s~%s, 구름≤%.1f%%",
                bbox, start_date, end_date, max_cloud_cover)

    response = requests.get(
        f"{CATALOGUE_URL}/Products", params=params, timeout=60
    )
    response.raise_for_status()

    products = response.json().get("value", [])
    results: list[S2SearchResult] = []

    for product in products:
        cloud = 0.0
        for attr in product.get("Attributes", []):
            if attr.get("Name") == "cloudCover":
                cloud = float(attr.get("Value", 0))
                break

        size_mb = product.get("ContentLength", 0) / (1024 * 1024)
        results.append(S2SearchResult(
            product_id=product["Id"],
            name=product["Name"],
            acquisition_date=product["ContentDate"]["Start"][:10],
            cloud_cover=cloud,
            footprint=product.get("Footprint", ""),
            processing_level=processing_level,
            online=product.get("Online", True),
            size_mb=round(size_mb, 1),
        ))

    results.sort(key=lambda r: r.cloud_cover)
    logger.info("S2 검색 결과: %d개 영상 발견", len(results))
    return results


def search_s2_for_k3a_scene(
    k3a_bounds_bbox: tuple[float, float, float, float],
    k3a_acquisition_date: str,
    date_range_days: int = 10,
    max_cloud_cover: float = 10.0,
    processing_level: str = "S2MSI1C",
    max_date_range_days: int = 30,
    min_coverage: float = 0.9,
) -> list[S2SearchResult]:
    """
    K3A 씬과 매칭되는 S2 영상을 검색합니다.
    K3A 촬영 날짜 ±date_range_days 범위 내에서 검색하고,
    결과가 없으면 ±max_date_range_days 까지 자동 확장합니다.

    Args:
        min_coverage: 최소 커버리지 비율 (0.0~1.0, 기본 0.9).
            1.0이면 K3A bbox 전체를 검색 영역으로 사용하고,
            0.9이면 bbox를 10% 축소하여 90%만 덮어도 매칭되도록 합니다.
    """
    # bbox 축소 (min_coverage 적용)
    min_lon, min_lat, max_lon, max_lat = k3a_bounds_bbox
    if min_coverage < 1.0:
        shrink = (1.0 - min_coverage) / 2.0
        lon_margin = (max_lon - min_lon) * shrink
        lat_margin = (max_lat - min_lat) * shrink
        search_bbox = (
            min_lon + lon_margin,
            min_lat + lat_margin,
            max_lon - lon_margin,
            max_lat - lat_margin,
        )
        logger.info("검색 bbox 축소 (%.0f%% 커버리지): %s → %s",
                     min_coverage * 100, k3a_bounds_bbox, search_bbox)
    else:
        search_bbox = k3a_bounds_bbox

    date_str = k3a_acquisition_date.replace("-", "")
    acq_date = datetime.strptime(date_str, "%Y%m%d")
    start = (acq_date - timedelta(days=date_range_days)).strftime("%Y-%m-%d")
    end = (acq_date + timedelta(days=date_range_days)).strftime("%Y-%m-%d")

    results = search_s2_by_bbox(
        bbox=search_bbox,
        start_date=start,
        end_date=end,
        max_cloud_cover=max_cloud_cover,
        processing_level=processing_level,
    )

    # 결과 없으면 날짜 범위 확장
    if not results and max_date_range_days > date_range_days:
        logger.info(
            "±%d일 내 S2 없음 → ±%d일로 확장 검색",
            date_range_days, max_date_range_days,
        )
        start = (acq_date - timedelta(days=max_date_range_days)).strftime("%Y-%m-%d")
        end = (acq_date + timedelta(days=max_date_range_days)).strftime("%Y-%m-%d")
        results = search_s2_by_bbox(
            bbox=search_bbox,
            start_date=start,
            end_date=end,
            max_cloud_cover=max_cloud_cover,
            processing_level=processing_level,
        )

    return results


def print_search_results(results: list[S2SearchResult]) -> None:
    """검색 결과를 보기 좋게 출력합니다."""
    if not results:
        print("검색 결과가 없습니다.")
        return
    print(f"\n{'='*80}")
    print(f"Sentinel-2 검색 결과: {len(results)}개")
    print(f"{'='*80}")
    for i, r in enumerate(results, 1):
        print(f"{i:>3} | {r.acquisition_date} | "
              f"구름 {r.cloud_cover:>5.1f}% | "
              f"{r.size_mb:>8.1f}MB | {r.name[:45]}")
    print(f"{'='*80}\n")


# ──────────────────────────────────────────────
# 다운로드 (Download)
# ──────────────────────────────────────────────

def download_s2_product(
    result: S2SearchResult,
    output_dir: Optional[str | Path] = None,
    extract: bool = True,
    token: Optional[str] = None,
) -> Path:
    """
    S2 제품을 다운로드합니다.

    Args:
        result: S2SearchResult 객체
        output_dir: 저장 디렉토리. None 이면 config 에서 자동 로딩.
        extract: True면 zip 압축 해제
        token: 액세스 토큰 (없으면 자동 발급)

    Returns:
        다운로드/압축해제된 디렉토리 Path
    """
    if output_dir is None:
        output_dir = _get_s2_raw_dir()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 이미 존재하는지 확인 (S2 다운로드 폴더는 항상 .SAFE 확장자를 포함함)
    safe_name = result.name.replace(".SAFE", "")
    extract_dir = output_dir / result.name
    if extract_dir.exists():
        logger.info("이미 다운로드됨, 스킵: %s", result.name)
        return extract_dir

    # 토큰 발급
    if token is None:
        token = get_access_token()

    # 다운로드
    zip_path = output_dir / f"{safe_name}.zip"
    url = f"{DOWNLOAD_URL}({result.product_id})/$value"
    headers = {"Authorization": f"Bearer {token}"}

    logger.info("다운로드 시작: %s (%.1f MB)", result.name, result.size_mb)

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=300)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        next_log_pct = 25  # 25% 단위로 로그 출력

        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=65536):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    if pct >= next_log_pct:
                        logger.info("  ↳ 다운로드 %d%% (%.0f/%.0f MB)",
                                    int(next_log_pct), downloaded / 1e6, total_size / 1e6)
                        next_log_pct += 25

        logger.info("다운로드 완료: %s", zip_path.name)

    except requests.RequestException as e:
        logger.error("다운로드 실패: %s - %s", result.name, e)
        if zip_path.exists():
            zip_path.unlink()
        raise

    # 압축 해제
    if extract:
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(output_dir)
            logger.info("압축 해제 완료: %s", safe_name)
            zip_path.unlink()  # zip 삭제하여 디스크 절약
        except zipfile.BadZipFile as e:
            logger.error("압축 해제 실패: %s - %s", zip_path.name, e)
            raise

    return extract_dir


def download_best_s2(
    results: list[S2SearchResult],
    output_dir: Optional[str | Path] = None,
    max_downloads: int = 1,
) -> list[Path]:
    """
    검색 결과에서 구름량이 가장 적은 S2 영상을 다운로드합니다.

    Args:
        results: S2SearchResult 리스트 (이미 구름량 정렬됨)
        output_dir: 저장 디렉토리
        max_downloads: 최대 다운로드 수

    Returns:
        다운로드된 디렉토리 Path 리스트
    """
    if not results:
        logger.warning("다운로드할 검색 결과가 없습니다.")
        return []

    token = get_access_token()
    downloaded: list[Path] = []

    for result in results[:max_downloads]:
        try:
            path = download_s2_product(
                result, output_dir=output_dir, token=token
            )
            downloaded.append(path)
        except Exception as e:
            logger.error("다운로드 실패, 다음 제품 시도: %s", e)
            continue

    logger.info("총 %d개 S2 영상 다운로드 완료", len(downloaded))
    return downloaded


# ──────────────────────────────────────────────
# 파이프라인용 일체형 함수
# ──────────────────────────────────────────────

def download_s2_for_k3a(
    k3a_bbox: tuple[float, float, float, float],
    k3a_acquisition_date: str,
    output_dir: Optional[str | Path] = None,
    date_range_days: int = 10,
    max_cloud_cover: float = 10.0,
    max_downloads: int = 1,
) -> list[Path]:
    """K3A 씬에 맞춰 S2 영상을 검색하고 다운로드합니다.

    전체 파이프라인에서 호출되는 일체형(one-shot) 함수입니다.

    Args:
        k3a_bbox: K3A 경계 (min_lon, min_lat, max_lon, max_lat).
        k3a_acquisition_date: K3A 촬영일 (YYYY-MM-DD 또는 YYYYMMDD).
        output_dir: 저장 디렉토리. None 이면 config 에서 자동 로딩.
        date_range_days: 검색 날짜 범위 (±일).
        max_cloud_cover: 최대 구름량 (%).
        max_downloads: 다운로드할 최대 영상 수.

    Returns:
        다운로드된 디렉토리 Path 리스트.
    """
    logger.info("K3A 씬 기반 S2 검색: bbox=%s, date=%s", k3a_bbox, k3a_acquisition_date)

    results = search_s2_for_k3a_scene(
        k3a_bounds_bbox=k3a_bbox,
        k3a_acquisition_date=k3a_acquisition_date,
        date_range_days=date_range_days,
        max_cloud_cover=max_cloud_cover,
    )

    if not results:
        logger.warning("K3A 영역에 해당하는 S2 영상을 찾을 수 없습니다.")
        return []

    print_search_results(results)
    return download_best_s2(results, output_dir=output_dir, max_downloads=max_downloads)


# ──────────────────────────────────────────────
# CLI 실행부
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    # 프로젝트 루트를 sys.path 에 추가
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="K3A 전체 씬 기준 Sentinel-2 검색 및 다운로드",
    )
    parser.add_argument(
        "--k3a-dir", default=None,
        help="K3A 추출 씬들이 있는 디렉토리 (기본: data/interim/k3a_extracted)",
    )
    parser.add_argument(
        "--days", type=int, default=10,
        help="K3A 촬영일 기준 검색 범위 (±일, 기본: 10)",
    )
    parser.add_argument(
        "--cloud", type=float, default=10.0,
        help="최대 구름량 (%%, 기본: 10)",
    )
    parser.add_argument(
        "--max", type=int, default=1,
        help="씬당 다운로드할 최대 S2 영상 수 (기본: 1)",
    )
    parser.add_argument(
        "--search-only", action="store_true",
        help="검색만 수행하고 다운로드하지 않음",
    )

    args = parser.parse_args()
    main_logger = logging.getLogger("s2_download")

    from src.data_access.k3a_loader import load_k3a_scene

    # K3A 씬 디렉토리 자동 탐색
    if args.k3a_dir:
        k3a_base = Path(args.k3a_dir)
    else:
        k3a_base = Path("data/interim/k3a_extracted")

    if not k3a_base.exists():
        main_logger.error("K3A 디렉토리를 찾을 수 없습니다: %s", k3a_base)
        sys.exit(1)

    # 모든 씬 디렉토리 탐색 (K3A_* 패턴)
    scene_dirs = sorted([d for d in k3a_base.iterdir() if d.is_dir() and d.name.startswith("K3A_")])

    if not scene_dirs:
        main_logger.error("K3A 씬을 찾을 수 없습니다: %s", k3a_base)
        sys.exit(1)

    main_logger.info("=" * 60)
    main_logger.info("K3A 씬 %d개 발견 → S2 검색 시작", len(scene_dirs))
    main_logger.info("=" * 60)

    total_downloaded = 0
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
        acq_date = scene.get_acquisition_date() or "20200101"

        main_logger.info("  촬영일: %s | BBOX: %s", acq_date, bbox)

        # 검색
        try:
            results = search_s2_for_k3a_scene(
                k3a_bounds_bbox=bbox,
                k3a_acquisition_date=acq_date,
                date_range_days=args.days,
                max_cloud_cover=args.cloud,
            )
        except Exception as e:
            main_logger.error("  S2 검색 실패: %s", e)
            failed += 1
            continue

        if not results:
            main_logger.warning("  매칭되는 S2 영상 없음")
            skipped += 1
            continue

        print_search_results(results)

        if args.search_only:
            continue

        # 다운로드
        try:
            downloaded = download_best_s2(results, max_downloads=args.max)
            total_downloaded += len(downloaded)
        except Exception as e:
            main_logger.error("  다운로드 실패: %s", e)
            failed += 1

    # ── 최종 요약 ──
    print(f"\n{'═'*60}")
    print(f"  S2 다운로드 전체 요약")
    print(f"{'═'*60}")
    print(f"  전체 씬      : {len(scene_dirs)}개")
    print(f"  다운로드 완료 : {total_downloaded}개")
    print(f"  스킵 (매칭없음): {skipped}개")
    print(f"  실패          : {failed}개")
    print(f"{'═'*60}")

