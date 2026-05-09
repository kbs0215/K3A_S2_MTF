# Module 2. 공간 정합 구현 계획

## 현재 구현 상태 (완료)
`src/preprocessing/coregistration.py` 및 배치 스크립트 `scripts/run_coregistration.py`를 통해 파이프라인이 자동화되어 있습니다.

## 주요 파일 구조 및 역할

- `src/preprocessing/coregistration.py`
  - `orthorectify_k3a_with_rpc()`: RPC와 DEM을 인자로 받아 K3A를 UTM 좌표계로 정사보정.
  - `create_virtual_grid()`: 정사보정된 K3A와 S2의 Bounds를 교집합 연산하여 2.5m 픽셀 사이즈의 `grid_profile` 생성.
  - `reproject_k3a_to_grid()`, `resample_s2_to_grid()`: 생성된 격자를 기준으로 두 영상을 동일하게 투영 및 Bilinear 보간.
  - `run_coregistration_pipeline()`: 위 단계들을 하나로 묶는 파이프라인.

- `scripts/run_coregistration.py` (배치 처리 스크립트)
  - 코페르니쿠스 서버 API 접속 대기 문제를 해결하기 위해 **로컬 오프라인 매칭(Offline Matching)** 로직 적용.
  - K3A 씬의 폴더명(날짜)을 파싱한 후, `data/raw/s2`에 보관된 `.SAFE` 디렉토리 목록 중 날짜가 가장 가까운 영상을 자동 스캔.
  - 4개 밴드(Blue, Green, Red, NIR)를 순회하며 일괄(Batch) 처리.
  - 교집합 영역이 없어서 `ValueError` 발생 시, 자동으로 다음으로 날짜가 가까운 후보 S2 타일을 찾아 재도전하도록 견고하게 구현.

## 향후 개선 포인트
- SIFT 기반 Fine Co-registration(정밀 정합) 모듈(`src/preprocessing/fine_registration.py`) 추가 개발.
