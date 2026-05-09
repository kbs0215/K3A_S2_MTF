# Module 2. 공간 정합 (Coregistration) 방법론

K3A(다중분광, ~2.2m)와 S2(10m)는 공간 해상도와 촬영 기하가 다르기 때문에, 방사 모사(IR-MAD)를 수행하기 전 픽셀 단위의 정확한 1:1 공간 매칭이 필수적입니다.

## 1. K3A 정사보정 (Orthorectification)
- **목적**: 위성 촬영 각도와 지형의 굴곡에 의한 K3A 원본 영상의 기하왜곡 교정.
- **방식**: 
  - K3A 씬에 포함된 **RPC (Rational Polynomial Coefficients)** 정보와 Module 1에서 획득한 **DEM (고도 데이터)**를 결합하여 GDAL/rasterio의 Warping 모듈을 구동합니다.
  - 출력 좌표계는 지역 UTM(예: EPSG:32652)으로 투영됩니다.

## 2. 공통 가상 격자 (Virtual Grid) 생성
- **목적**: K3A·S2 두 영상을 동일한 공간 외곽선 위에 일관되게 투영하기 위한 공통 격자 정의. 두 영상의 격자가 **외곽 사각형(outer rectangle) 과 origin 을 공유**하도록 함.
- **방식**:
  - **공통 origin**: 정사보정된 K3A 의 UTM bbox 와 S2 bbox 의 교집합 좌상단 (minx, maxy). UTM 자연 격자에 별도 snap 하지 않음.
  - **두 해상도 동시 생성** (한 함수 호출 → `{"k3a_profile": ..., "s2_profile": ...}`):
    - **K3A 격자**: 2.5m 정확 픽셀 (K3A native ~2.2m 에 가까움).
    - **S2 격자**: 10m 정확 픽셀 (S2 10m 밴드 native 해상도).
    - 두 격자 origin·outer rect 동일. **S2 셀 1개 = K3A 셀 4×4** (10/2.5=4).
  - **outer rect 결정**: S2 픽셀 수 = `ceil((maxx − minx) / 10)`, K3A 픽셀 수 = S2 × 4. 우/하단 모서리에 ≤10m extra 영역이 생길 수 있으며 그 부분은 nodata=0.
- **트레이드오프**: outer rect 가 UTM 10m 자연 격자에 정렬되지 않으므로 S2 native 픽셀이 출력 격자에 0~10m 만큼 bilinear 보간 시프트됨. HLS 스타일 inward snap (UTM 10m 자연 격자 anchor) 옵션도 가능하나, 모서리 데이터 보존 우선으로 snap 없음 채택.

## 3. 해상도 일치를 위한 리샘플링 (Resampling)
- **K3A → 2.5m 격자**: bilinear 보간 (정사보정된 K3A 를 가상격자에 정렬).
- **S2 → 10m 격자**: bilinear 보간 (origin 미정렬에 의한 시프트 보정).
- **보간법 선정 이유**: 추후 Module 3 에서 진행될 **MTF 분석은 영상의 에지(Edge) 특성에 매우 민감**. Bicubic 은 에지 주변에서 링잉 아티팩트를 발생시킬 수 있어 부적합. Bilinear 가 원본 방사값과 에지를 상대적으로 정직하게 보존.

## 4. K3A↔S2 매칭 정책
- **공간 필터** (`run_coregistration.py::bbox_contains`): S2 한 타일이 K3A bbox 를 WGS84 에서 **완전히 포함** 해야 후보로 채택. 부분 겹침은 K3A 일부가 S2 밖이라 정합 결과가 K3A 의 일부만 담게 되므로 제외.
- **시간 필터**: K3A·S2 촬영일 차이 ≤ `--max-date-diff` (기본 30일).
- **정렬**: 두 조건을 만족하는 후보 중 날짜 차이가 가장 작은 SAFE 부터 순서대로 시도.
- **알려진 한계**: K3A 가 MGRS 타일 경계에 걸치면 단일 SAFE 로 K3A 전체를 못 덮음 → 현재는 skip 처리 (경고 메시지에 "다중 타일 모자이크 필요" 표시). 향후 같은 날짜 인접 SAFE 들을 `gdal.BuildVRT` 로 한 장처럼 묶어 입력하는 모자이크 단계 도입 예정.

## 5. 페어 라벨링 (Pair Labeling)
- 정합 출력 파일명 prefix: `{N}_{stem}_grid.tif`. 같은 N 을 K3A 격자와 S2 격자 두 출력에 부여하여 디렉토리에서 페어가 한눈에 보이도록.
- 번호 = K3A 씬 디렉토리 정렬 순서 (`enumerate(scene_dirs, 1)` 의 idx). 재실행 시 안정.
- 스킵된 씬 번호는 출력에 빈자리로 남아 어느 씬이 처리/실패했는지 단서 제공.

## 6. 정밀 공간 정합 (Fine Co-registration) — 선택적 심화 단계
- 대략적 정합 후에도 잔존하는 sub-pixel 단위 오차(shift, rotation 등) 를 교정하기 위해 **SIFT/ORB 기반 특징점 추출** + **RANSAC** 을 활용한 미세 픽셀 워핑을 추가 적용할 수 있음 (현재 미구현).
