4/26

기존 로직에서 

L1C -> L2A 변환 추가 필요 

    이유 : 픽셀의 DN값이 대기 상태에 따라 달라지기 때문에 
    대기보정 전의 두 데이터를 사용해서 초해상화의 쌍을 구축하면
    많은 노이즈가 발생할 것으로 예상됨.


그리고 또 개선할 점이 뭐가 있지?

---

5/7

module2 batch 정합 파이프라인 (`run_coregistration.py`) 첫 가동.

## 환경 셋업
- conda env `gongjong` 재구축. `gdal + rasterio + libgdal-jp2openjpeg + openjpeg` 를 conda-forge 에서 한 번에 설치.
  - 처음에 pip install requirements.txt 로 시도 → GDAL 빌드 실패. Windows + conda 조합에서 GDAL 은 항상 conda-forge 통일이 정답.
  - JP2OpenJPEG 드라이버 등록만 되고 실제 jp2 open 시 DLL load 실패하는 함정 발견 → CLAUDE.md 7.1.(4) 에 정리.
- `requirements.txt` 에 `gdal>=3.6.0` 명시 + conda 권장 주석 추가.

## 코드 수정
- `config/paths.json` 에 `k3a_extracted_dir` 키 추가, `run_coregistration.py` 가 CWD 의존 없이 PROJECT_ROOT + paths.json 으로 절대경로 해석하도록 변경 → 어디서 실행해도 동작.
- `coregistration.py::orthorectify_k3a_with_rpc()` 에 `_M_rpc.txt → _<X>_rpc.txt` 자동 복사 로직 추가 (CLAUDE.md 7.1.(6)).
- `run_coregistration.py` 매칭 로직 개선:
  - K3A bbox vs S2 footprint 사전 교집합 필터 (B02 jp2 bounds 캐싱)
  - `--max-date-diff` (기본 30일) 윈도우 필터
  - fail / skip 카운트 분리 (가용 S2 없는 씬 = skip)
- `coregistration.py` 의 두 reproject 호출에 `src_nodata=0, dst_nodata=0` 명시 → "value 0 changed to 1" 경고 제거. NoData 정책은 옵션 1(`nodata=0`) 유지로 결정.
- batch 스크립트 except 처리를 `logger.exception` 으로 바꿔서 풀 traceback 출력.

## 데이터 현황
- K3A 35씬 중 31개가 `_M_rpc.txt` only (밴드별 RPC 없음). _M_rpc fix 적용 전엔 31개 전부 fail 이었음.
- 보유 S2 12개. 한반도 전역 K3A 35씬을 전부 매칭하기엔 부족 — 추가 다운로드 필요.

## 미해결 / 다음 작업
- 4/26 메모: L1C → L2A 변환 추가 (방사모사 입력으로 대기보정 영상이 더 적합).
- 정사보정 결과 위치 정확도 검증 (S2 매칭 또는 GCP 비교).
- S2 추가 다운로드해서 매칭 안 된 K3A 씬 커버리지 확장.
- 다운스트림(module3 ir_mad / rad_simulation) 에서 NoData 마스킹 정책 정합 — 현재 `data == 0` 으로 단순 처리 시 footprint 밖과 valid 0 이 같이 묶임을 인지.

---

5/8

module2 정합 모듈 대규모 재정비 + 가상 격자 알고리즘 변경.

## 모듈 분할 (`coregistration.py` → 5 files)
이전 단일 ~500줄 `coregistration.py` 를 기능별로 분리:
- `env.py` — GDAL/PROJ 환경 변수 + `_load_paths`, `_PROJ_DATA` (import 시 side effect 로 GDAL_PAM_ENABLED 등 설정).
- `rpc.py` — `_find_rpc_file`, `_parse_rpc_txt` (K3A RPC 사이드카 탐색/파싱).
- `ortho.py` — `orthorectify_k3a_with_rpc` (gdal.Warp + RPC + DEM, _M_rpc.txt 자동 복사 로직 포함).
- `grid.py` — `create_virtual_grid` + `reproject_k3a_to_grid` + `resample_s2_to_grid`.
- `pipeline.py` — `run_coregistration_pipeline` + `_run_pipeline` (오케스트레이터).
`__init__.py` 가 공개 API 재노출 + 패키지 import 시 GDAL env 초기화. 기존 `coregistration.py` 삭제, `run_coregistration.py` 와 `notebooks/02_test_coregistration.ipynb` 의 import 경로 갱신.

## 공통 가상 격자 알고리즘 변경 (단일 2.5m → K3A 2.5m + S2 10m)
- 이전: K3A 와 S2 둘 다 2.5m 가상 격자에 bilinear → S2 가 oversample 되어 의미 없는 4× 데이터 증식.
- 신규: 두 격자를 한 함수에서 동시 생성 (`create_virtual_grid` 가 `{"k3a_profile": ..., "s2_profile": ...}` dict 반환).
  - **공통 origin** = K3A∩S2 교집합 bbox 의 좌상단 (minx, maxy). UTM 자연 격자에 snap 안 함.
  - S2 격자: 10m 정확 픽셀 (`from_origin`), 픽셀 수 = `ceil((maxx-minx)/10)`.
  - K3A 격자: 2.5m 정확 픽셀, 픽셀 수 = S2 픽셀 수 × 4.
  - 두 격자 outer rect 정확히 동일 (S2 1셀 = K3A 4×4셀).
  - 우/하단 모서리에 ≤10m extra 영역 (K3A·S2 데이터 없으면 nodata=0).
- **트레이드오프**: outer rect 모서리가 UTM 10m 자연 격자에 정렬되지 않으므로 S2 native 픽셀이 bilinear 로 0~10m 시프트됨. UTM 10m anchor 옵션 (HLS 스타일 inward snap) 도 검토했으나 "모서리 데이터 보존 우선" 으로 snap 없음 채택.
- 함수 시그니처: `create_virtual_grid(ortho_k3a_path, s2_bounds, k3a_resolution=2.5, s2_resolution=10.0, crs)`.

## 페어 라벨링
- 한 K3A 씬 ↔ 한 S2 SAFE 의 정합 페어를 출력 파일명에 시각화: `{N}_{stem}_grid.tif`.
- 번호 = K3A 씬 디렉토리 정렬 순서 (`enumerate(scene_dirs, 1)` 의 idx). 결정적.
- 같은 N 을 K3A·S2 격자 두 출력에 동시 부여 → 디렉토리 ls 만으로 페어 매칭 식별.
- 변경 범위: `_run_pipeline` 에 `pair_label` 파라미터 추가, `run_coregistration.py` 가 idx 를 그대로 라벨로 전달, 스킵 로직도 라벨된 파일명 기준으로 갱신.

## 공간 필터 강화 (intersect → contains)
- 이전 (`bbox_intersects`): K3A bbox 와 조금이라도 겹치는 S2 SAFE 를 후보로 채택. 결과적으로 K3A 가 S2 한 타일을 벗어나면 K3A 일부가 S2 footprint 밖에 위치한 채로 정합되어 출력 격자가 K3A 일부만 담음.
- 신규 (`bbox_contains`): S2 한 타일이 K3A 전체를 완전히 포함해야 후보. 못 덮으면 그 K3A 씬은 skip.
  - 부분 겹침 후보가 있으면 "K3A 가 MGRS 타일 경계에 걸침, 다중 타일 모자이크 필요" 경고 후 skip.
- 다중 타일 S2 모자이크 (`gdal.BuildVRT` 로 같은 날짜 인접 SAFE 합치기) 알고리즘 설계는 완료, 구현은 다음 작업으로 연기.

## 미해결 / 다음 작업
- **K3A double bilinear 최적화**: 현재 K3A 는 (RPC ortho → 임의 2.5m UTM) → (reproject_k3a_to_grid → 가상격자 2.5m) 의 두 번 bilinear 를 거침. `gdal.Warp` 에 `outputBounds`, `xRes`, `yRes` 명시하면 RPC warp 1회로 가상격자에 직접 떨어뜨릴 수 있음 (가상격자 정의 전 cheap 한 ortho extent probe 필요).
- **S2 다중 타일 모자이크**: K3A 가 MGRS 경계에 걸치는 씬 처리. 같은 날짜 인접 SAFE 들을 `gdal.BuildVRT` 로 묶어 한 장의 S2 처럼 파이프라인 입력. 단일 UTM 존 → 다른 존 → polygon 정밀 검사 단계적 도입.
- (carryover) L1C → L2A 변환, 정사보정 위치 정확도 검증, NoData 마스킹 정책 정합.

---

2026-05-10

module3 방사·MTF 모사 모듈 1차 구현 완료 + 전 페어 배치 통과.

## 현재 코드 구조 (module1~3)
- `module1_data_download/src/`: `k3a_loader.py` · `s2_download.py` · `dem_download.py` · `raster_io.py` — Copernicus 실시간 검색에서 로컬 `.SAFE` 오프라인 매칭으로 전환 후 안정화.
- `module2_coregistration/src/`: `rpc.py` · `ortho.py` · `grid.py` · `pipeline.py` — 5/8 분할 + 공통 가상격자(K3A 2.5m + S2 10m) + bbox_contains 필터로 가동 중.
- `module3_simulation/src/`: `ir_mad.py` · `linear_norm.py` · `pipeline.py` · `mtf.py` — 신규.
- `shared/utils/`: `proj_env.py` (GDAL/PROJ env side effect) · `paths.py` (paths.json 로딩). 모든 모듈 `__init__.py` 에서 import 시 자동 초기화.

## module3 신규 구현
### 방사모사 (`run_simulation.py`)
- 입력: `data/interim/grid/{N}_*_grid.tif` 페어 (K3A 4밴드 2.5m + S2 4밴드 10m).
- 절차: K3A 4×4 mean → 10m 다운샘플 → IR-MAD 로 PIF 마스크 → 밴드별 `S2 = a·K3A + b` fit → 학습된 (a,b) 를 **K3A 2.5m 원본** 에 적용.
- `aggregate_4x4_mean` 은 permissive (블록 16픽셀 중 1개 valid 라도 평균에 포함). strict 마스크 시 NIR/가장자리에서 IR-MAD valid_mask 가 비는 문제 회피 — CLAUDE.md 7.3.(2) 에 정리.
- 페어별 사이드카 JSON (`*_radsim.json`) 에 PIF 수, canonical correlations, band coefficients(R², RMSE) 기록 + `_summary.json` 집계.

### MTF 모사 (`run_mtf_simulation.py` + `mtf.py`)
- 상대적 PSF 추정: K3A 2.5m 에 가우시안 σ 변화시키며 블러 → 4×4 mean → S2 10m 과 robust MSE 최소화.
- 다단계 fallback (CLAUDE.md 7.3.(3)): ① 100% valid + 최대 분산 패치 탐색 → ② percentile cutoff 단계적 완화로 σ 최적화 → ③ boundary 갇히면 `phase_cross_correlation` 으로 K3A 패치 sub-pixel shift 후 재최적화.
- 전체영상 적용은 `normalized_gaussian_filter` (NoData 블리딩 수학적 차단: `(V*mask) blurred / mask blurred`).

### 진단 (`triage_failures.py`)
- `*_radsim.json` 전수 읽어 status="fail" 페어 추출 → K3A bbox + 촬영일로 로컬 다른 S2 후보 탐색 → `try-local-alt` / `needs-download` / `manual` 분류.
- read-only. 실제 재처리는 사람이 결과 보고 module2/3 수동 재실행.

## 데이터 현황 (2026-05-10 기준)
| 단계 | 산출물 |
|---|---|
| K3A 원본 | 3개 zip (Seoul/Daejeon/Gimjae), 추출 35씬 |
| S2 SAFE | 12개 — 35씬 전수 커버에는 부족 |
| DEM 타일 | 44개 |
| `interim/ortho/` | 133 TIF (~33씬 × 4밴드) |
| `interim/grid/` | 256 TIF — 32 페어 × 8 (K3A 4 + S2 4) |
| `output/rad_sim/` | success **30** / skip 0 / fail **1** / 31 시도 |
| `output/mtf_sim/` | success **30** / skip 0 / fail 0 / 30 시도 |

## CLAUDE.md 정리
- 1장 프로젝트 구조: 모듈 트리 + Sentinel-2 검색 쿼리 주의 + 모듈화 규칙 보강.
- 7.2 batch 정합: bbox_contains, 공통 가상격자, 페어 라벨, 모자이크 미구현 노트 추가.
- **7.3 Module 3 신설**: IR-MAD 입력 정책, permissive 마스크, MTF σ fallback 순서, NoData 블리딩 차단 패턴, status JSON / triage 흐름.

## 미해결 / 다음 작업
- **rad_sim fail 1건 triage**: `triage_failures.py` 로 분류한 뒤 module2/3 재처리.
- **MTF 결과 검증**: 30개 페어의 σ 분포·R² 분포 일관성 확인. σ 가 sigma_max 근처에 몰리면 phase_correlation 단계가 자주 트리거된다는 신호.
- **module4_webapp**: `docs/` 만 있고 `src/` 없음. 본격 착수 전 입력 명세(어떤 산출물을 어떤 UI 로 보여줄지) 정의 필요.
- (carryover) L1C → L2A 대기보정, 정사보정 위치 정확도 검증, K3A double bilinear 1회화, S2 다중 타일 모자이크, S2 추가 다운로드.

---

2026-05-10 (저녁)

SR 학습용 페어 칩 추출 단계 추가.

## 신규: `module3_simulation/scripts/run_chip_extraction.py`
- **칩 사양**: HR 896×896 @2.5m + LR 224×224 @10m, 동일 footprint 2240×2240 m, 4:1 비율 (SR 학습 LR/HR 페어).
- **HR 소스**: `interim/grid/{label}_K3A_*_{B|G|R|N}_grid.tif` — 정사보정·공통격자만 적용된 K3A 원본 (방사·MTF 모사 안 됨, GT 역할).
- **LR 소스**: `output/mtf_sim/{label}_*_simulated_final_2p5m.tif` 를 `mtf.aggregate_4x4_mean` 으로 10m 다운샘플 — "S2 처럼 보이도록 모사된 K3A" 의 LR view.
- **타일링**: LR 격자 stride 150 (overlap 74/224 = 33.0%, 사용자 요구 "최대 1/3 중첩" 한계). HR stride 자동 `×4 = 600`.
- **valid 필터**: HR 칩 안 4밴드 모두 nonzero 비율 < 0.8 스킵.
- **출력**: `output/chips/{label}_{scene}/` 아래 `{base}_hr.tif`, `{base}_lr.tif`, `{base}_meta.json` (CRS, UTM bbox, transform, valid_ratio). 루트에 `_chip_summary.json`.
- **결정성**: 칩 ID = HR 픽셀 좌표 `y{y:05d}_x{x:05d}` → 재실행 시 동일 파일명, `--overwrite` 없으면 기존 칩 스킵.

## 부수 변경
- `config/paths.json` 에 `mtf_sim_dir`, `chips_dir` 키 추가.
- CLAUDE.md 7.3.(6) 에 칩 추출 사양·valid 필터·결정성 정리.

## 검증 / 다음 작업
- 윈도우 `gongjong` env 에서 `python module3_simulation/scripts/run_chip_extraction.py` 실행해서 30 페어 칩 생성 확인 필요 (WSL 쪽에는 GDAL/rasterio 없음).
- `_chip_summary.json` 으로 페어당 평균 칩 수 / 스킵률 확인 → valid_threshold 0.8 이 너무 빡빡하면 0.7 로 완화 검토.
- 칩 메타의 UTM bbox 가 같은 페어 내 HR/LR 에서 일치하는지 spot-check.

