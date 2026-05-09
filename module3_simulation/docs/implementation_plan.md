# Module 3. 방사 및 MTF 모사 구현 계획

## 향후 구현 예정 스크립트 구조

이 모듈은 아직 구현되지 않았으며, 다음 단계의 개발 타겟입니다.

- `src/radiometric/`
  - `ir_mad.py`: CCA 계산 알고리즘 및 확률 맵 추출 기능 구현.
  - `rad_simulation.py`: PIF 마스크 생성 및 `scikit-learn` 등을 활용한 선형 회귀(Linear Regression) 파라미터 획득, 그리고 정규화(Normalization) 적용 기능.

- `src/mtf/`
  - `edge_detection.py`: Canny 에지 및 Hough 변환을 이용한 최적의 에지 추출기.
  - `mtf_calculation.py`: LSF / ESF 도출 및 1D 푸리에 변환 모듈.
  - `mtf_filter.py`: 주파수 응답을 기반으로 공간 도메인 컨볼루션 커널 설계.

- `src/simulation/`
  - `simulate_k3a.py`: 방사 모사와 MTF 필터링을 파이프라인으로 연결하여 최종 `output/` 렌더링.

- `scripts/run_simulation.py`
  - 배치 파이프라인으로 모듈 1, 2에서 처리 완료된 파일들을 입력받아 백그라운드 연산을 수행하는 스크립트 작성.
