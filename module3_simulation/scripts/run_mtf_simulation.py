"""run_mtf_simulation.py - 방사모사가 완료된 영상을 기반으로 MTF 모사 배치 실행

흐름:
  1) data/output/rad_sim/ 의 *_radsim.json 을 읽어 성공한 방사모사 결과(K3A 2.5m) 확인
  2) 해당 K3A 2.5m TIF 와 S2 10m TIF (grid) 로드
  3) mtf.py의 상대적 PSF 추정 로직 수행 (가우시안 블러 -> MSE 최소화)
  4) 최종 방사 및 공간(MTF) 특성이 모사된 2.5m K3A 영상을 저장
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import List, Tuple

import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from module3_simulation.src.mtf import simulate_mtf_all_bands
from shared.utils.proj_env import PROJ_DATA

_PATHS_CONFIG = json.loads((PROJECT_ROOT / "config" / "paths.json").read_text(encoding="utf-8"))


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def load_band_stack(band_paths: dict, band_order: List[str]) -> Tuple[np.ndarray, dict]:
    arrs = []
    profile = None
    for band in band_order:
        path = band_paths.get(band)
        if not path:
            raise FileNotFoundError(f"밴드 누락: {band}")
        with rasterio.open(path) as src:
            arrs.append(src.read(1))
            if profile is None:
                profile = src.profile.copy()
    return np.stack(arrs, axis=0), profile


def write_tif(out_path: Path, data: np.ndarray, profile: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = profile.copy()
    profile.update(count=data.shape[0], dtype=data.dtype.name)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)


def _save_record(record: dict, out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="MTF 모사 배치 (상대적 PSF 추정)")
    parser.add_argument("--rad-dir", default=None, help="방사모사 결과 디렉토리")
    parser.add_argument("--out-dir", default=None, help="최종 결과 저장 디렉토리")
    parser.add_argument("--sigma-min", type=float, default=0.05, help="최소 sigma 탐색값")
    parser.add_argument("--sigma-max", type=float, default=9.0, help="최대 sigma 탐색값")
    parser.add_argument("--sigma-step", type=float, default=0.05, help="sigma 탐색 간격")
    parser.add_argument("--overwrite", action="store_true", help="기존 출력 덮어쓰기")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("mtf_sim_batch")

    rad_dir = (
        Path(args.rad_dir).resolve()
        if args.rad_dir
        else (PROJECT_ROOT / _PATHS_CONFIG["rad_sim_dir"]).resolve()
    )
    # out_dir은 mtf_sim_dir이라는 설정이 따로 없으므로 rad_sim_dir 옆에 생성
    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else (PROJECT_ROOT / _PATHS_CONFIG["output_dir"] / "mtf_sim").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if not rad_dir.exists():
        logger.error("방사모사 디렉토리를 찾을 수 없습니다: %s", rad_dir)
        sys.exit(1)

    json_files = list(rad_dir.glob("*_radsim.json"))
    logger.info("=" * 60)
    logger.info("MTF 모사 배치 시작: %d개 페어 탐색", len(json_files))
    logger.info("  rad_dir: %s", rad_dir)
    logger.info("  out_dir: %s", out_dir)
    logger.info("=" * 60)

    success, skip, fail = 0, 0, 0
    env_kwargs = {"PROJ_LIB": PROJ_DATA} if PROJ_DATA else {}

    with rasterio.env.Env(**env_kwargs):
        for json_path in sorted(json_files):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    rad_record = json.load(f)
                
                if rad_record.get("status") != "success":
                    continue
                
                label = rad_record["pair_label"]
                scene_stem = rad_record["scene_stem"]
                bands = rad_record["bands"]
                
                out_tif = out_dir / f"{label}_{scene_stem}_simulated_final_2p5m.tif"
                out_json = out_dir / f"{label}_{scene_stem}_mtfsim.json"

                if out_tif.exists() and out_json.exists() and not args.overwrite:
                    logger.info("[%s] 이미 성공 결과 존재 -> 스킵", label)
                    skip += 1
                    continue

                logger.info("\n[%s] MTF 모사 시작", label)

                # 1. K3A 방사모사 결과 로드 (2.5m)
                k3a_path = rad_dir / Path(rad_record["output_tif"]).name
                if not k3a_path.exists():
                    logger.error("  TIF 파일을 찾을 수 없습니다: %s", k3a_path)
                    fail += 1
                    continue

                with rasterio.open(k3a_path) as src:
                    k3a_2p5m = src.read()
                    k3a_profile = src.profile.copy()

                # 2. S2 원본 그리드 로드 (10m)
                s2_10m, _ = load_band_stack(rad_record["s2_grid_files"], bands)

                # 3. MTF 모사 수행
                simulated_2p5m, mtf_results = simulate_mtf_all_bands(
                    img_k3a_2p5m=k3a_2p5m,
                    img_s2_10m=s2_10m,
                    band_names=bands,
                    sigma_min=args.sigma_min,
                    sigma_max=args.sigma_max,
                    sigma_step=args.sigma_step,
                    nodata_value=0.0
                )

                # 4. 결과 저장
                write_tif(out_tif, simulated_2p5m, k3a_profile)

                record = {
                    "pair_label": label,
                    "scene_stem": scene_stem,
                    "status": "success",
                    "bands": bands,
                    "sigma_search_range": [args.sigma_min, args.sigma_max, args.sigma_step],
                    "mtf_results": [
                        {
                            "band_name": r.band_name,
                            "best_sigma": float(r.best_sigma),
                            "min_mse": float(r.min_mse)
                        } for r in mtf_results
                    ],
                    "input_rad_tif": str(k3a_path),
                    "output_tif": str(out_tif)
                }
                _save_record(record, out_json)
                logger.info("[%s] 성공: %s", label, out_tif.name)
                success += 1

            except Exception as e:
                logger.exception("[%s] 실패: %s", json_path.stem, e)
                fail += 1

    summary_path = out_dir / "_mtf_summary.json"
    _save_record({
        "success": success,
        "skip": skip,
        "fail": fail,
        "total": success + skip + fail,
    }, summary_path)

    logger.info("=" * 60)
    logger.info("배치 종료 — 성공 %d, 스킵 %d, 실패 %d", success, skip, fail)
    logger.info("summary: %s", summary_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
