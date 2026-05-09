# Module 1. 데이터 다운로드 및 접근 (Data Access) 방법론

본 문서는 K3A_S2_MTF 프로젝트의 첫 번째 모듈인 **데이터 획득 및 접근 구조**에 대한 방법론을 정의합니다.

## 1. Kompsat-3A (K3A) 데이터 처리
- **원본 구조**: K3A 영상은 일반적으로 L1R 레벨의 `.zip` 파일 형태로 제공됩니다.
- **처리 방법**:
  - `zip` 아카이브 내에 존재하는 다중 분광(Blue, Green, Red, NIR) 및 PAN 밴드 이미지(`.tif`)와 메타데이터 파일(`Aux.xml`)을 동적으로 인식합니다.
  - 디스크 공간 최적화 및 접근 속도 향상을 위해 메모리 상에서 직접 읽거나, 필요 시 임시 폴더(`data/interim/k3a_extracted`)로 자동 압축 해제합니다.
  - `Aux.xml`을 파싱하여 촬영 시간, 태양 고도각/방위각, 지리적 경계(Bounding Box) 등의 필수 정보를 추출합니다.

## 2. Sentinel-2 (S2) 데이터 로딩 및 매칭
- **원본 구조**: ESA Copernicus에서 제공하는 `.SAFE` 디렉토리 구조 (내부에 `GRANULE/.../IMG_DATA/` 하위 `.jp2` 파일 존재).
- **매칭 방법 (Offline Matching)**:
  - 기존에는 K3A의 Bounding Box를 기반으로 Copernicus OData API에 실시간 검색 쿼리를 날려 다운로드하는 방식을 취했으나, 속도와 안정성 문제로 **로컬 매칭 방식**으로 변경하였습니다.
  - `data/raw/s2` 에 사전 다운로드된 `.SAFE` 파일들의 이름을 파싱하여 획득 날짜를 추출합니다.
  - K3A 영상의 촬영일과 가장 가까운 날짜의 S2 `.SAFE` 폴더를 우선 후보로 선정합니다.

## 3. DEM (Digital Elevation Model) 다운로드
- **목적**: K3A 영상의 기하왜곡을 보정하는 정사보정(Orthorectification) 단계에 고도 정보(Z축)를 제공.
- **획득 방법**:
  - K3A 영상의 Bounding Box(WGS84)를 활용하여 Copernicus DEM API (또는 SRTM 등 외부 API)를 호출하여 해당 영역의 DEM을 GeoTIFF 형태로 다운로드합니다.
  - 동일한 K3A 씬에 대한 다중 밴드 처리 시, 한 번만 다운로드하여 로컬(`data/raw/dem`)에 캐싱함으로써 네트워크 비용을 최소화합니다.
