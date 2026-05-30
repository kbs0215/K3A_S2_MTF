# Module 5 — SR Downstream Comparison Plan

본 모듈은 학습 완료된 SR 모델(`best_sr_model.pth`)이 **downstream 수체탐지(Sen1Floods11)** 성능에 실질적으로 기여하는지를 정량 비교한다.

---

## 1. 목적

> "내 Prithvi 기반 SR 모델로 superresolve한 입력이, 원본 10m S2 또는 단순 업샘플(bicubic)에 비해 downstream 물탐지 정확도를 더 끌어올리는가?"

SR 자체의 픽셀 metric(PSNR/SSIM)은 본 모듈 범위 밖. **task-driven evaluation**만 수행한다.

---

## 2. 비교군 (3-way)

| ID | 입력 | 해상도 | 비고 |
|---|---|---|---|
| **A** | real S2 L1C | 10m | 하한선 baseline (해상도 효과 0) |
| **B** | bicubic 4× | 2.5m | naive upsample (학습 없는 baseline) |
| **D** | Prithvi-SR (내 모델) | 2.5m | 평가 대상 |

GT 고해상도 라벨이 Sen1Floods11에 없으므로 상한선(real K3A HR)은 본 비교에 포함 불가.

**학습형 small SR baseline (C, 이전 FSRCNN 계획) 은 본 실행 환경에서 학습 불가하여 드롭.**
대안 non-learned baseline (Lanczos, sharpened bicubic 등) 은 bicubic 과 변별력이 작아 추가 가치 없음 → 3-way 로 단순화.

비교 메시지:
- A vs D : SR 해상도 효과 (10m → 2.5m → downsample 했을 때 정보 보존)
- B vs D : 학습형 SR 의 순수 기여 (동일 2.5m 입력에서 모델 vs 단순 보간)
- A vs B : bicubic 만으로도 10m 와 동등한지 검증

---

## 3. 파이프라인 채널 정책

전 파이프라인 **4채널 BGRN 통일**.

- SR 입력 (LR): S2 B02/B03/B04/B08 4밴드 (10m)
- SR 출력 (HR): 4밴드 (2.5m)
- U-Net 입력: 4밴드 (SR 출력 그대로)
- Prithvi encoder 입력은 학습 시 관습대로 **6채널 zero-pad** (마지막 2채널 = 0) 유지. SWIR 실제값 삽입 금지 (학습 분포 이탈 위험).

---

## 4. 데이터 — Sen1Floods11 subset

- 소스: `Sen1Floods11/HandLabeled/S2Hand` (L1C TOA reflectance × 10000)
- **3 events** 선정 (지역 다양성: India / Bolivia / USA 등에서 1개씩)
- 각 event 약 50 chips → **총 ~150 chips**, 512×512×4bands
- 라벨: `LabelHand/` binary water mask (10m)
- 다운로드 크기: ~500 MB
- L2A 변환(sen2cor) **수행하지 않음**. L1C↔L2A 도메인 갭은 limitation로 명시. 비교군 4종 모두 동일 L1C 입력이므로 비교 자체는 공정.

---

## 5. Downstream 모델 — segmentation_models_pytorch U-Net

학습 없이 **라이브러리 사전학습 weight 로드 후 inference만** 수행.

```python
import segmentation_models_pytorch as smp
import torch

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=4,
    classes=1,
)
# 선택적 안전장치: NIR 채널 초기화는 Red 채널 weight 복사 (long-wavelength similarity)
with torch.no_grad():
    model.encoder.conv1.weight[:, 3] = model.encoder.conv1.weight[:, 0]
```

**Prithvi Sen1Floods11 finetuned 모델은 사용하지 않음** — 평가자 백본이 SR encoder와 동일하면 "내 SR이 Prithvi가 좋아하는 입력을 만들었다"는 비판 여지가 생김. smp는 완전 독립 아키텍처로 evaluator 중립성 확보.

학습이 정 필요하면 추후 옵션. 본 PLAN은 zero-train.

---

## 6. 평가 프로토콜 — 해상도 정합

라벨이 10m, SR 출력이 2.5m이므로 **모든 비교는 10m grid에서** 수행:

- A(real S2 10m): U-Net 출력 10m → 라벨과 직접 비교
- B/C/D(2.5m): U-Net 출력 2.5m logit → **4×4 mean pool → 10m** → threshold → 라벨과 비교

라벨을 NN으로 2.5m upsample하여 2.5m에서 비교하는 방식은 라벨이 부풀어 IoU 부정확 → 금지.

---

## 7. Metric

### 7.1 메인 (U-Net based)

- **IoU** (water class)
- **F1**
- **Boundary IoU** — 라벨 경계로부터 ±1 픽셀(10m grid) zone에서만 IoU 계산. SR의 경계 sharpness 우위가 가장 잘 드러나는 metric.
- (서브) **Small CC IoU** — 라벨에서 connected component 면적 < 100 픽셀인 객체 한정. N이 작으면 보고에서 제외.

### 7.2 보조 (NDWI threshold)

학습·추론 비용 0. SR의 spectral 충실도 평가용.

- NDWI = (G − NIR) / (G + NIR), 입력 DN 스케일에서 그대로 계산
- Otsu auto-threshold → binary water mask
- 동일하게 10m grid에서 평가, 같은 metric 세트

NDWI 결과는 메인 메시지를 **spectral 측면에서 보강**: U-Net에서 우위 + NDWI에서 spectral 보존 = "내 SR이 공간 sharpening과 spectral fidelity를 동시에 달성".

---

## 8. 디렉토리 구조 (self-contained)

```
module5_SR_comparison/
├── best_sr_model.pth          # 이미 존재 (1.3 GB)
├── docs/
│   └── PLAN.md                # 본 문서
├── src/
│   ├── sr_inference.py        # bicubic / Prithvi-SR 함수
│   ├── prithvi_sr.py          # best_sr_model.pth 로드 + inference
│   ├── tile_stitch.py         # 큰 입력 sliding-window 추론
│   ├── flood_eval.py          # smp.Unet 로드 + 4ch 입력 inference
│   ├── ndwi.py                # NDWI + Otsu
│   └── metrics.py             # IoU / F1 / Boundary IoU / Small CC IoU
├── scripts/
│   ├── download_sen1floods.py # 3 events × 4 bands 다운로드
│   ├── run_sr_all.py          # 3 비교군 SR 출력 생성
│   ├── run_downstream.py      # U-Net + NDWI 추론 + metric 집계
│   └── make_figures.py        # 정성 figure + metric 표
├── data/
│   └── sen1floods11/          # HandLabeled subset
│       ├── S2Hand/            # 4-band L1C chips
│       └── LabelHand/         # binary water masks
└── output/
    ├── sr_outputs/            # {B,D}/{chip_id}.tif  (A 는 원본 직접 사용)
    ├── flood_masks/           # {A,B,D}/{unet,ndwi}/{chip_id}.tif
    ├── metrics.csv            # 비교군 × 평가방식 × metric
    └── figures/               # qualitative comparison
```

다른 모듈의 `data/` `data/output/` 트리는 건드리지 않음.

---

## 9. 예상 비용

- 디스크: ~1.5 GB (data 500 MB + SR 출력 2종(B,D) ~600 MB + masks/figures ~400 MB)
- 시간 (Windows `gongjong` env, GPU 가정):
  - Sen1Floods11 다운로드: ~30분
  - SR inference (B bicubic + D Prithvi-SR) × 150 chips: ~1시간
  - U-Net + NDWI 추론: ~30분
  - Metric 집계 + figure: ~1–2시간
  - **합계 ≈ 3–4시간**

---

## 10. 알려진 한계 (limitation 명시)

1. **L1C/L2A 도메인 갭**: SR 모델은 L2A reflectance×10000 stats로 학습. 본 평가는 L1C TOA. 3 비교군 모두 동일 L1C라 비교는 공정하나, 절대 성능은 L2A 입력보다 낮을 수 있음.
2. **SWIR 미활용**: Prithvi 6채널 중 마지막 2채널은 학습 시 zero-pad 관습 유지. SWIR가 물탐지에 유용함은 알려져 있으나 본 SR 모델 분포 외라 본 평가 범위 밖.
3. **라벨 해상도 한계**: Sen1Floods11 라벨이 10m → 2.5m SR의 미세 디테일(< 10m 구조물) 평가가 boundary IoU에 의존. 작은 수체 sub-metric은 데이터셋 라벨 특성(대규모 홍수 이벤트) 상 N이 작을 수 있음.
4. **단일 downstream task**: 결과의 일반화는 수체탐지에 한정. 분류·검출 등 다른 task로 확장은 후속 연구.

---

## 11. 실행 순서

1. `scripts/download_sen1floods.py` — 데이터 확보
2. `scripts/run_sr_all.py` — B/D 입력 2종 생성 (A 는 원본 직접 사용)
3. `scripts/run_downstream.py` — U-Net + NDWI 추론, metric.csv 작성
4. `scripts/make_figures.py` — qualitative 비교 + metric 표
