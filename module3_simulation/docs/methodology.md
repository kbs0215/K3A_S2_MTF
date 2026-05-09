# Module 3. 방사 및 MTF 모사 (Simulation) 방법론

공간적으로 완벽히 매칭된 두 영상에 대해 방사적 특성(Radiometric characteristics)과 광학적 선명도(MTF)를 모사하여, 최종적으로 K3A를 S2와 동일한 조건으로 시뮬레이션합니다.

## 1. 방사 모사 (Radiometric Simulation)

기존의 대기/조명 조건을 반영하는 복잡한 물리 기반 보정(DN→Radiance→TOA) 대신, 통계적 접근인 **IR-MAD 기반 선형 정규화**를 사용합니다.

### 1.1 IR-MAD (Iteratively Reweighted MAD) 기반 PIF 탐지
- **목적**: 계절, 대기, 조명 변화에도 반사도가 변하지 않은 **불변 픽셀(PIF, Pseudo-Invariant Features)**을 자동으로 추출.
- **방식**:
  - 두 영상(K3A, S2) 간에 가중치가 적용된 정준상관분석(CCA)을 반복적으로 수행합니다.
  - 카이제곱 분포를 기반으로 불변 확률값(Probability)을 산출하고, 임계값(예: `0.95` 이상)을 만족하는 픽셀들을 불변 영역(PIF) 마스크로 정의합니다.

### 1.2 선형 회귀 학습 및 정규화 (Linear Normalization)
- **목적**: 추출된 PIF를 기준으로 K3A 영상을 S2 영상의 방사 특성에 맞춤.
- **방식**:
  - 각 밴드별로 PIF에 해당하는 픽셀들을 추출해 선형 회귀식(`S2 = a * K3A + b`)을 Fitting 합니다.
  - 도출된 `Gain (a)`과 `Offset (b)` 파라미터를 K3A 전체 영상에 일괄 적용하여 방사 모사된 K3A를 생성합니다.

## 2. MTF (Modulation Transfer Function) 분석 및 필터링

방사 특성이 맞춰진 영상에 대하여, 광학계의 선명도 차이를 측정하고 반영합니다.

### 2.1 상대적 MTF 계산
- **Slanted-edge Method**: 영상 내 존재하는 자연적/인공적 직선 에지를 탐지(Hough Transform 등)하여 LSF(Line Spread Function)와 ESF(Edge Spread Function)를 도출합니다.
- 푸리에 변환을 통해 주파수 대역별 MTF 곡선을 계산하고, S2와 K3A 간의 MTF 차이(비율)를 구합니다.

### 2.2 시뮬레이션 영상 생성 (Point Spread Function 적용)
- 측정된 MTF 차이를 기반으로 공간 주파수 필터(Kernel 또는 PSF)를 설계합니다.
- 이 필터를 K3A (또는 역으로 S2) 영상에 컨볼루션(Convolution) 연산하여, 최종적으로 렌즈 블러 등 광학적 특성까지 완벽히 일치하는 모사 영상(Simulated K3A)을 완성합니다.
