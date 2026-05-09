# Module 1. 데이터 다운로드 구현 계획

## 현재 구현 상태 (완료)
이 모듈의 핵심 스크립트는 `src/data_access/` 하위에 위치하며, 이미 대부분의 기능이 구현되어 안정화 단계에 있습니다.

## 주요 파일 구조 및 역할

- `src/data_access/k3a_loader.py`
  - `load_k3a_scene()`: K3A 압축 해제 디렉토리를 순회하여 `band_files` 맵핑과 메타데이터(`Aux.xml`)를 읽어오는 객체(K3AScene)를 반환.
  - `parse_k3a_aux_xml()`: XML 파싱을 통해 `bounds` (Polygon) 및 `acquisition_date` 반환.

- `src/data_access/s2_download.py`
  - S2 `.SAFE` 폴더 파싱 및 다운로드.
  - (현재는 API 통신이 아닌 로컬 폴더 내 오프라인 매칭 구조로 변경됨에 따라, 검색 로직은 배치 스크립트로 이관되어 보조적 역할 수행)

- `src/data_access/dem_download.py`
  - `download_dem_for_k3a()`: K3A Bounding Box 좌표를 EPSG:4326 기준으로 파싱하여, Copernicus Open Access Hub 등을 통해 DEM GeoTIFF 타일 다운로드.

- `src/data_access/raster_io.py`
  - `rasterio` 기반의 범용적인 GeoTIFF 읽기/쓰기 유틸리티.

## 향후 개선 포인트
- S2 `.SAFE` 폴더 구조 외에 `.zip` 상태에서의 직접 리딩 기능 추가 (디스크 공간 절약 목적).
- DEM 다운로드 API 장애 시 fallback 옵션(SRTM 30m 로컬 데이터 등) 지원.
