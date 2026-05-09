# 프로젝트 코딩 가이드라인 (Rules)

# K3A_S2_MTF 프로젝트 코딩 가이드라인


# 프로젝트 핵심

**Kompsat-3A 영상을 Sentinel-2 영상에 모사**
1. 주어진 K3A 영상과 100% 겹치는 Sentinel-2 영상 검색
2. 방사모사
3. 상대적 MTF값 계산
4. 최종 2,3번이 완료된 Simulated K3A 영상 생성


## 1. 프로젝트 구조 규칙 (Project Structure Rules)

### 1.1. 모듈 레이아웃 (Module-based)
파이프라인 단계별로 독립 모듈을 두고, 각 모듈 안에서 `src/`(라이브러리) + `scripts/`(배치 진입점) + `docs/`(설계 문서) 를 동일하게 유지한다.

```
K3A_S2_MTF/
├── claude.md                      # 본 가이드라인
├── config/                        # paths.json / processing_params / sensor_specs
├── data/
│   ├── raw/        {k3a, s2, dem}   # 입력 원본
│   ├── interim/    {k3a_extracted, ortho, grid}  # 중간 산출물
│   └── output/     {rad_sim, mtf_sim}            # 최종 산출물
├── docs/development_log.md        # 날짜별 진행 로그
├── module1_data_download/
│   └── src/  k3a_loader.py · s2_download.py · dem_download.py · raster_io.py
├── module2_coregistration/
│   └── src/  rpc.py · ortho.py · grid.py · pipeline.py · __init__.py
│       scripts/run_coregistration.py
├── module3_simulation/
│   └── src/  ir_mad.py · linear_norm.py · pipeline.py · mtf.py · __init__.py
│       scripts/run_simulation.py · run_mtf_simulation.py · triage_failures.py
├── module4_webapp/                # 미구현 (docs 만)
└── shared/utils/  paths.py · proj_env.py   # 모든 모듈 공통 (GDAL/PROJ env, paths.json 로딩)
```

### 1.2. Sentinel-2 검색 쿼리 주의
Copernicus Data Space 에서 `Collection/Name eq 'S2MSI1C'` 는 잘못됨.
- Collection 이름: **SENTINEL-2** (고정)
- Product type: **S2MSI1C** (L1C) / **S2MSI2A** (L2A) — `Attributes` 로 필터.

### 1.3. 코드 모듈화 규칙
- 하나의 파일은 500줄 이하로 작성 (현재 module1 `s2_download.py` 585줄은 예외, 분할 검토).
- 주석: 파일 상단에 10줄 이내로 기능 요약.
- 타입 힌트: 모든 함수와 클래스에 필수 적용.
- 신규 모듈은 `module{N}_<name>/{src,scripts,docs}/` 구조 + `src/__init__.py` 에서 공개 API 재노출 + `shared.utils.proj_env` import 로 GDAL 환경 초기화 (side effect).

## 2. 언어 및 프레임워크 규칙

### 2.1. 백엔드 (Backend)
- **환경 변수**: `.env` 파일 사용, 절대 경로 저장 금지

### 2.3. 에이전트 (Agents)
- **프레임워크**: LangGraph 1.x
- **LLM**: OpenAI GPT-4o 또는 Claude Opus 4.5 사용
- **메모리**: `MessageMemory` 사용, 1000개 메시지 제한
- **도구**: `tool` 데코레이터 사용, 타입 힌트 필수

## 3. API 설계 규칙

### 3.1. RESTful API 명명 규칙
- **명사 사용**: `/users`, `/orders` 등
- **HTTP 메서드**: GET, POST, PUT, DELETE만 사용
- **버전 관리**: `/v1/` 접두사 사용
- **응답 형식**: JSON only, Content-Type: application/json

### 3.2. API 보안 규칙
- 모든 엔드포인트에 rate limiting 적용 (100회/분)
- SQL Injection 방지: 매개변수화된 쿼리 사용
- 입력값 검증: Joi 또는 Pydantic 사용
- CORS: 특정 도메인만 허용

## 4. 데이터베이스 규칙

### 4.1. 스키마 설계
-snake_case 사용
-모든 테이블에 soft delete 컬럼(`is_deleted`, `deleted_at`) 추가
-인덱스: 쿼리 100ms 초과 시 인덱스 추가
-데이터 타입: JSON 필드는 PostgreSQL JSONB 타입 사용

## 5. 성능 최적화 규칙

### 5.1. API 성능 규칙
- 응답 시간: 95 percentile 200ms 이하
- 페이징: limit 100, offset 0 기본값
- 캐싱: Redis 5분 캐싱 적용
- CDN: 모든 정적 파일은 CDN 사용

### 5.2. LLM 최적화 규칙
-temperature = 0.7 고정
-prompt 길이: 5000 토큰 이하
-비용 절감: 20k 토큰 초과 시 경고 로그

## 6. 보안 규칙

- 비밀번호: bcrypt 해시 사용
- 입력값 검증: 모두 필수
- 로깅: 보안 이벤트는 별도 DB에 저장

## 7. 알려진 이슈 / 주의사항 (Known Issues)

### 7.1. K3A 정사보정 (rasterio + GDAL RPC)

#### (1) `BLOCKXSIZE must be a multiple of 16`
- **증상**: 정사보정 출력 시 `GDAL signalled an error: err_no=5, msg='..._ortho.tif: BLOCKXSIZE must be a multiple of 16'` → `The height and width of TIFF dataset blocks must be multiples of 16`
- **원인**: K3A 원본 GeoTIFF는 stripped layout (또는 16의 배수가 아닌 blocksize)인데, `src.profile.copy()` 후 `tiled: True`만 켜면 원본의 비호환 blocksize가 그대로 따라감.
- **해결**: tiled 출력 시 `blockxsize`/`blockysize`를 16의 배수(권장 256)로 **명시**.
  ```python
  kwargs.update({
      "tiled": True,
      "blockxsize": 256,
      "blockysize": 256,
      "compress": "lzw",
  })
  ```

#### (2) `src_transform, gcps, rpcs, and src_geoloc_array are mutually exclusive`
- **증상**: `rasterio.warp.reproject(...)` 호출 시 위 메시지로 즉시 실패.
- **원인**: 최신 rasterio는 `src_transform`/`src_crs`와 `rpcs`를 **상호 배타적**으로 강제. RPC 정사보정 시에는 `src_transform`/`src_crs`를 같이 넘기면 안 됨.
- **해결**:
  - `reproject` 호출에서 `src_transform`, `src_crs` 제거하고 `rpcs=rpcs`만 전달.
  - `calculate_default_transform`도 K3A는 raw image (좌표계 없음)이므로 `src.bounds` 대신 `rpcs=rpcs` + `src_crs="EPSG:4326"`로 RPC 기반으로 출력 영역 계산.
  ```python
  reproject(
      source=rasterio.band(src, i),
      destination=rasterio.band(dst, i),
      rpcs=rpcs,
      dst_transform=transform,
      dst_crs=dst_crs,
      resampling=Resampling.bilinear,
      **rpc_options,
  )
  ```

#### (3) `CPLE_NotSupported in warp options does not support option RPC_DEM / RPC_DEMINTERPOLATION` (WARNING)
- **증상**: DEM을 `RPC_DEM=str(dem_path)` 형태로 `rasterio.warp.reproject(..., **kwargs)`에 넘기면 출력되는 경고. 처리는 계속되지만 **DEM이 RPC 변환에 실제로 적용되지 않음** (Z=0 정사보정으로 동작).
- **원인**: `RPC_DEM`, `RPC_DEMINTERPOLATION`은 GDAL `Warp options`이 아니라 **`Transformer options`** (RPC transformer에 전달되어야 함). rasterio는 `**kwargs`를 무조건 Warp options로 분류하므로 GDAL이 거부.
- **적용된 해결**: `osgeo.gdal.Warp(... , transformerOptions=[...])` 로 직접 전달. `module2_coregistration/src/ortho.py::orthorectify_k3a_with_rpc()` 가 이 방식을 사용한다.
  ```python
  from osgeo import gdal
  gdal.UseExceptions()

  transformer_options = []
  if dem_path:
      transformer_options.append(f"RPC_DEM={Path(dem_path).as_posix()}")
      transformer_options.append("RPC_DEMINTERPOLATION=bilinear")

  warp_opts = gdal.WarpOptions(
      format="GTiff",
      dstSRS=target_crs,           # e.g. "EPSG:32652"
      xRes=target_resolution, yRes=target_resolution,
      rpc=True,
      transformerOptions=transformer_options,
      resampleAlg="bilinear",
      multithread=True,
      creationOptions=["TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256", "COMPRESS=LZW"],
  )
  ds = gdal.Warp(str(out_path), str(k3a_tif_path), options=warp_opts)
  ds = None  # flush & close
  ```
- **주의**:
  - 외부 RPC 사이드카(`*_rpc.txt`)는 GDAL이 동일 디렉토리에서 자동 인식. 사이드카 파일이 원본 TIF 옆에 있어야 함.
  - 정사보정 결과의 위치 정확도는 GCP 또는 S2 매칭으로 별도 검증 권장.

#### (4) GDAL JP2OpenJPEG DLL 로드 실패 (conda-forge plugin ABI mismatch)
- **증상**: `ERROR 1: Can't load requested DLL: ...\gdal_JP2OpenJPEG.dll  127: 지정된 프로시저를 찾을 수 없습니다.`
- **원인**: conda-forge GDAL 3.9+ 부터 JP2 등 일부 드라이버가 별도 plugin 패키지로 분리됨(`libgdal-jp2openjpeg`). 본체 gdal 과 plugin DLL 의 ABI 가 안 맞으면 등록은 되지만 실제 파일 open 시 DLL load 단계에서 실패. `gdal.GetDriverByName('JP2OpenJPEG')` 가 OK 리턴해도 안심하면 안 됨.
- **해결**: `--force-reinstall` 만으로는 부족한 경우가 많음. **env 자체를 재구축**하고 본체+plugin+의존성을 **한 번에** conda-forge 에서 받기:
  ```powershell
  conda env remove -n <env>
  conda create -n <env> -c conda-forge python=3.11 gdal rasterio libgdal-jp2openjpeg openjpeg <기타>
  ```
- **검증**: `gdalinfo <S2 .jp2 절대경로>` 가 size/CRS/Corner Coordinates 를 정상 출력해야 함. PowerShell 멀티라인 입력은 path 중간에 공백/개행이 끼는 경우가 있으니 변수에 담아서 호출 권장(`$f="..."; gdalinfo $f`).

#### (5) rasterio `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc1` (한국어 윈도우)
- **증상**: `with rasterio.open(s2_jp2_path)` 호출 시 위 메시지로 폭발. 종종 `Exception ignored in: 'rasterio._env.log_error'` 가 동반.
- **원인**: GDAL 이 시스템 에러/경고 메시지를 한국어 윈도우 로케일(CP949) 로 emit. rasterio `_err.pyx` 가 utf-8 strict 디코드를 시도해서 폭발. **메시지는 증상일 뿐 진짜 원인은 다른 곳(예: #4 의 DLL load 실패)**.
- **해결**: 위 디코딩 자체를 손대지 말고, GDAL 이 한국어 메시지를 emit 하게 만든 **원인을 제거**할 것. 실무상 거의 항상 #4 와 함께 발생하므로 #4 fix 가 자동으로 #5 도 해소.
- **부수적 잡음 차단** (선택): 모듈 import 시점에
  ```python
  os.environ.setdefault("GDAL_PAM_ENABLED", "NO")  # .aux.xml PAM 사이드카 무시 → 한글 경고 source 하나 차단
  os.environ.setdefault("CPL_DEBUG", "OFF")
  ```

#### (6) K3A `_M_rpc.txt` only 씬에서 `Unable to compute a RPC based transformation`
- **증상**: 일부 K3A 씬에서 `gdal.Warp(..., rpc=True)` 가 `RuntimeError: Unable to compute a RPC based transformation between pixel/line and georeferenced coordinates for ..._B.tif` 로 실패.
- **원인**: GDAL 의 RPC 사이드카 자동 인식은 **TIF stem 과 정확히 매칭**되는 파일만 찾음(`{stem}_rpc.txt` 또는 `{stem}.RPB`). K3A 의 일부 씬은 밴드별 `_B_rpc.txt`/`_G_rpc.txt`/... 가 없고 멀티스펙트럼 통합 `_M_rpc.txt` 만 제공 → stem 불일치 → transformer 생성 실패. `_find_rpc_file()` 가 `_M_rpc.txt` 를 fallback 으로 *찾기는* 하지만 GDAL 에 명시적으로 전달하지는 않음.
- **적용된 해결**: `orthorectify_k3a_with_rpc()` 가 호출될 때 stem-매칭 사이드카가 없으면 `_M_rpc.txt` 를 그 이름으로 자동 복사. K3A 의 B/G/R/N 4개 multispectral 밴드는 통합 RPC 를 공유하므로 의미적으로 안전(첫 씬 검증: _B/_G/_R/_N_rpc.txt 모두 동일 바이트수). PAN(`_P_rpc.txt`) 는 별도 RPC.
  ```python
  expected_sidecar = k3a_tif_path.with_name(k3a_tif_path.stem + "_rpc.txt")
  if rpc_file != expected_sidecar and not expected_sidecar.exists():
      shutil.copy2(rpc_file, expected_sidecar)
  ```

#### (7) `Value 0 in source has been changed to 1 in destination to avoid being treated as NoData`
- **증상**: `rasterio.warp.reproject(...)` 출력 시 위 경고가 매 밴드마다 반복.
- **원인**: 출력 프로필의 `nodata=0` 과 source 의 valid 0 픽셀이 충돌. GDAL 이 자동 보호 차원에서 valid 0 → 1 로 bump.
- **적용된 정책**: K3A 12-bit DN 에서 1 DN 차이는 무시 가능 (≈0.024%) 하므로 `nodata=0` 유지. 경고만 끄기 위해 reproject 호출에 `src_nodata=0, dst_nodata=0` 명시. K3A footprint 안의 진짜 0 은 NoData 로 함께 묶임 — 다운스트림에서 `data == 0` 으로 마스킹할 때 footprint 밖 + 어두운 valid 픽셀이 한 덩어리.

### 7.2. Batch 정합 파이프라인 매칭 로직 (`module2_coregistration/scripts/run_coregistration.py`)

- **공간 사전 필터** (`bbox_contains`): S2 SAFE 한 타일이 K3A bbox 를 WGS84 에서 **완전히 포함** 해야 후보로 채택. 부분 겹침은 K3A 일부가 S2 밖이라 정합 결과가 K3A 의 일부만 담게 되므로 제외 (5/8 변경 — 이전 `bbox_intersects` 에서 강화). SAFE bounds 는 B02 jp2 의 native CRS bounds → WGS84 변환 후 dict 캐시(`s2_bounds_cache`) 로 재사용.
- **날짜 윈도우**: `--max-date-diff`(기본 30일) 초과하는 후보는 제외. S2 다운로드 시 ±30일 페어로 받았으므로 그 이상 시도는 무의미.
- **fail vs skip 구분**: 가용 S2 가 없는 씬은 `fail` 이 아닌 `skip` 으로 카운트. fail 은 코드/데이터 결함, skip 은 데이터 부족 — 통계 해석을 분리.
- **공통 가상격자**: `create_virtual_grid` 가 K3A 2.5m + S2 10m 두 프로필을 **한 번에** 생성. origin(좌상단) 공유, S2 1셀 = K3A 4×4. outer rect 가 UTM 10m 자연격자에 anchor 되지 않음(모서리 데이터 보존 우선). 출력 파일명 prefix `{N}_` 는 `enumerate(scene_dirs, 1)` 의 idx — 같은 N 을 K3A·S2 두 출력에 부여해 페어가 한눈에 보임.
- **다중 타일 모자이크**: K3A 가 MGRS 경계에 걸쳐 단일 SAFE 로 못 덮는 경우 현재 skip. `gdal.BuildVRT` 로 같은 날짜 인접 SAFE 모자이크 입력은 설계만 완료 (미구현).

### 7.3. Module 3 방사·MTF 모사 (`module3_simulation/`)

#### (1) IR-MAD 입력은 K3A 를 10m 로 다운샘플 후 S2 와 정합
- **방식**: `run_simulation.py::aggregate_4x4_mean` 으로 K3A 2.5m → 10m 평균 다운샘플 후 IR-MAD 수행. PIF 기반 선형회귀로 (a,b) 를 학습한 뒤, **학습된 (a,b) 를 K3A 2.5m 원본에 적용** 해 출력은 2.5m 유지.
- **이유**: IR-MAD 의 χ² CDF 는 두 영상이 동일 grid 일 것을 가정. 2.5m vs 10m 직접 입력은 차원 안 맞음.

#### (2) `aggregate_4x4_mean` 은 permissive 마스크
- 4×4 블록 안에서 valid 픽셀(`!= 0`) 만 평균. 16 픽셀 모두 0 인 블록만 NoData 유지.
- **strict 마스크 (16 픽셀 전부 valid 요구) 금지**: NIR 의 어두운 픽셀이나 폴리곤 가장자리에서 너무 많은 블록을 떨궈 IR-MAD valid_mask 가 비게 됨.
- 단, 이 정책은 7.1.(7) 의 `nodata=0` 정책과 결합되어 **footprint 밖 + valid 0 픽셀이 한 덩어리**로 마스킹됨에 유의 (다운스트림 통계 해석 시).

#### (3) MTF — 가우시안 PSF σ 최적화 + 다단계 fallback (`mtf.py::estimate_relative_psf_per_band`)
순서대로:
1. **패치 탐색** (`find_best_patch`): S2 10m 영상에서 100% valid 이고 분산(에지 대비) 최대인 500×500 패치 선택. K3A 패치는 그 4× 영역(2000×2000 @2.5m).
2. **σ 최적화** (`scipy.optimize.minimize_scalar bounded`): K3A 패치에 가우시안 블러 → 4×4 mean 다운샘플 → S2 패치와 robust MSE. percentile cutoff `[95,90,85,80,75,70,60,50,40,30]` 단계적 완화 — 매 cutoff 마다 σ 가 boundary(`sigma_min+0.05` 또는 `sigma_max−0.05`) 에 수렴하면 다음 cutoff 로 재시도.
3. **위상정합 fallback**: 모든 cutoff 가 boundary 에 갇히면 `skimage.registration.phase_cross_correlation` 으로 K3A↔S2 sub-pixel shift 측정 → K3A 패치를 물리적으로 shift 후 σ 재최적화. 이 경로는 정합 잔차가 σ 추정을 망가뜨리는 경우의 마지막 수단.
4. **전체영상 적용**: 찾은 σ 를 K3A 2.5m 전체에 1회 적용 — 이때는 항상 `normalized_gaussian_filter` (NoData 블리딩 차단).

#### (4) `normalized_gaussian_filter` — NoData 블리딩 차단
- 일반 `gaussian_filter` 는 NoData(0) 를 valid 0 으로 취급해 블러 결과가 가장자리에서 어두워짐.
- 패턴: `V*mask` 와 `mask` 를 각각 가우시안 블러 후 나눈다 (`blurred_V / blurred_W`). 분모가 0 에 가까운 픽셀(`blurred_W <= 1e-4`) 은 출력 0.
- 패치 최적화 단계에서 100% valid 패치가 발견된 경우(`is_perfect=True`) 는 일반 `gaussian_filter` 사용 — 정규화 비용 회피.

#### (5) Status 사이드카 JSON
- `run_simulation.py` / `run_mtf_simulation.py` 는 모든 페어에 대해 결과 JSON 을 항상 기록 (success/skip/fail 무관). 각 디렉토리에 `_summary.json` / `_mtf_summary.json` 으로 집계.
- `triage_failures.py` 는 `*_radsim.json` 을 모두 읽어 fail 페어를 분류 (try-local-alt / needs-download / manual). 기본 read-only — 실제 재처리는 사용자가 보고 module2/3 수동 재실행.

#### (6) 칩 추출 (`run_chip_extraction.py`) — SR 학습용 페어 칩
- **칩 사양**: HR 896×896 @2.5m + LR 224×224 @10m, 동일 footprint(2240×2240 m), 4:1 비율.
- **HR 소스**: `data/interim/grid/{label}_K3A_*_{B|G|R|N}_grid.tif` (정사보정 + 공통격자, **방사·MTF 모사 안 된 원본**).
- **LR 소스**: `data/output/mtf_sim/{label}_*_simulated_final_2p5m.tif` 를 `aggregate_4x4_mean` 으로 4×4 mean → 10m. 즉 "S2 처럼 보이도록 모사된 K3A" 의 LR view.
- **타일링**: LR 격자에서 `stride_lr=150` (overlap=74/224 ≈ 33.0%, "최대 1/3 중첩" 한계). HR stride 는 자동으로 `stride_lr × 4 = 600`.
- **valid 필터**: HR 칩 안에서 4밴드 모두 nonzero 인 픽셀 비율 < 0.8 이면 스킵. `nodata=0` 정책상 footprint 밖과 dark valid 0 이 함께 묶이는 점 인지 (7.1.(7)).
- **출력**: `data/output/chips/{label}_{scene}/{base}_{hr,lr}.tif` + `{base}_meta.json` (CRS, UTM bbox, transform, valid_ratio_hr 등). 디렉토리당 `_chip_summary.json` 집계.
- **결정성**: 칩 ID 는 HR 픽셀 좌표 `y{y:05d}_x{x:05d}` — 재실행 시 동일 칩이 같은 파일명. `--overwrite` 없으면 기존 칩 스킵.


