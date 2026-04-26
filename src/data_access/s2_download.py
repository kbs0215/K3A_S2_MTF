"""
s2_download.py - Sentinel-2 영상 다운로드

Copernicus Data Space API로부터 S2 영상을 다운로드합니다.
검색된 S2SearchResult의 product_id를 사용하여 zip 파일을 받고
data/raw/s2/ 디렉토리에 저장합니다.
"""

import logging
import zipfile
from pathlib import Path
from typing import Optional

import requests

from src.data_access.s2_search import S2SearchResult, get_access_token

logger = logging.getLogger(__name__)

DOWNLOAD_URL = "https://zipper.dataspace.copernicus.eu/odata/v1/Products"


def download_s2_product(
    result: S2SearchResult,
    output_dir: str | Path = "data/raw/s2",
    extract: bool = True,
    token: Optional[str] = None,
) -> Path:
    """
    S2 제품을 다운로드합니다.

    Args:
        result: S2SearchResult 객체
        output_dir: 저장 디렉토리
        extract: True면 zip 압축 해제
        token: 액세스 토큰 (없으면 자동 발급)

    Returns:
        다운로드/압축해제된 디렉토리 Path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 이미 존재하는지 확인
    safe_name = result.name.replace(".SAFE", "")
    extract_dir = output_dir / safe_name
    if extract_dir.exists():
        logger.info("이미 다운로드됨, 스킵: %s", safe_name)
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

        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    if downloaded % (10 * 1024 * 1024) < 8192:  # 10MB마다 로그
                        logger.info("다운로드 진행: %.1f%%", pct)

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
    output_dir: str | Path = "data/raw/s2",
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
