"""
s2_search.py - Copernicus Data Space에서 Sentinel-2 영상 검색

K3A 영상의 bbox를 기반으로 중첩하는 S2 영상을 검색합니다.
인증 정보는 .env 파일에서 로딩합니다.
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"


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
            ".env.example 참고하여 .env 파일을 생성하세요."
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
    min_lon, min_lat, max_lon, max_lat = bbox

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
    collection_filter = f"Collection/Name eq '{processing_level}'"
    full_filter = " and ".join([
        footprint_filter, date_filter, cloud_filter, collection_filter
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
    date_range_days: int = 30,
    max_cloud_cover: float = 10.0,
    processing_level: str = "S2MSI1C",
) -> list[S2SearchResult]:
    """
    K3A 씬과 매칭되는 S2 영상을 검색합니다.
    K3A 촬영 날짜 ±date_range_days 범위 내에서 검색합니다.
    """
    date_str = k3a_acquisition_date.replace("-", "")
    acq_date = datetime.strptime(date_str, "%Y%m%d")
    start = (acq_date - timedelta(days=date_range_days)).strftime("%Y-%m-%d")
    end = (acq_date + timedelta(days=date_range_days)).strftime("%Y-%m-%d")

    return search_s2_by_bbox(
        bbox=k3a_bounds_bbox,
        start_date=start,
        end_date=end,
        max_cloud_cover=max_cloud_cover,
        processing_level=processing_level,
    )


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
