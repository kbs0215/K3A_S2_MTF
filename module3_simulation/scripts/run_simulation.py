"""run_simulation.py - 페어 단위 IR-MAD 방사모사 배치

흐름:
  1) data/interim/grid/ 의 페어를 자동 탐지 ({label}_*_grid.tif)
  2) 페어별로 K3A 2.5m, S2 10m 다중밴드 stack 로드
  3) K3A 4×4 mean → 10m 다운샘플 (S2 와 동일 해상도)
  4) IR-MAD + PIF 선형회귀 fit (10m 에서)
  5) 학습된 (a, b) 를 K3A 2.5m 원본에 적용 → 2.5m simulated K3A
  6) data/output/ 에 GeoTIFF + 계수 JSON 사이드카 저장

MTF 모사는 module3 의 추후 단계에서 별도 처리.
"""

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from module3_simulation.src import (
    apply_linear_normalization,
    radiometric_simulation,
)
from shared.utils.proj_env import PROJ_DATA

_PATHS_CONFIG = json.loads((PROJECT_ROOT / "config" / "paths.json").read_text(encoding="utf-8"))


K3A_BAND_ORDER = ["Blue", "Green", "Red", "NIR"]

# K3A grid 파일명 suffix — module2/pipeline.py 의 출력 명명규칙
K3A_BAND_SUFFIX = {
    "Blue":  "_B_grid.tif",
    "Green": "_G_grid.tif",
    "Red":   "_R_grid.tif",
    "NIR":   "_N_grid.tif",
}
S2_BAND_SUFFIX = {
    "Blue":  "_B02_grid.tif",
    "Green": "_B03_grid.tif",
    "Red":   "_B04_grid.tif",
    "NIR":   "_B08_grid.tif",
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def discover_pairs(grid_dir: Path) -> Dict[str, Dict[str, Dict[str, Path]]]:
    """grid_dir 의 *_grid.tif 들을 페어/센서/밴드 기준으로 묶어 반환.

    파일명 규약 (module2/pipeline.py):
      K3A: "{label}_K3A_{scene}_{B|G|R|N}_grid.tif"
      S2 : "{label}_T{tile}_{date}_{B02|B03|B04|B08}_grid.tif"

    Returns:
        {
          "1": {"K3A": {"Blue": Path, "Green": Path, "Red": Path, "NIR": Path},
                "S2":  {"Blue": Path, ...}},
          "2": {...},
          ...
        }
    """
    pairs: Dict[str, Dict[str, Dict[str, Path]]] = defaultdict(
        lambda: {"K3A": {}, "S2": {}}
    )
    for tif in grid_dir.glob("*_grid.tif"):
        m = re.match(r"^(\d+)_", tif.name)
        if not m:
            continue
        label = m.group(1)
        if "_K3A_" in tif.name:
            for band, suf in K3A_BAND_SUFFIX.items():
                if tif.name.endswith(suf):
                    pairs[label]["K3A"][band] = tif
                    break
        else:
            for band, suf in S2_BAND_SUFFIX.items():
                if tif.name.endswith(suf):
                    pairs[label]["S2"][band] = tif
                    break
    return dict(pairs)


def load_band_stack(
    band_paths: Dict[str, Path], band_order: List[str]
) -> Tuple[np.ndarray, dict]:
    """주어진 밴드들을 (bands, H, W) 로 스택. 첫 밴드의 rasterio profile 도 반환."""
    arrs: List[np.ndarray] = []
    profile: dict | None = None
    for band in band_order:
        if band not in band_paths:
            raise FileNotFoundError(f"밴드 누락: {band}")
        with rasterio.open(band_paths[band]) as src:
            arrs.append(src.read(1))
            if profile is None:
                profile = src.profile.copy()
    return np.stack(arrs, axis=0), profile  # type: ignore[return-value]


def aggregate_4x4_mean(arr: np.ndarray) -> np.ndarray:
    """K3A 2.5m → 10m 평균 다운샘플 (permissive).

    각 4×4 블록 안에서 valid 픽셀(!=0) 만 평균. 16 픽셀 모두 0 인 블록만 0
    (NoData) 유지. 일부 valid + 일부 nodata 인 폴리곤 가장자리 블록도 살아남음.
    "16 픽셀 전부 valid 여야 한다" 식의 strict 마스크는 NIR 의 어두운 픽셀이나
    가장자리에서 너무 많은 블록을 떨궈 IR-MAD 의 valid_mask 가 비게 만드는
    문제가 있어 permissive 로 둠.

    H/W 가 4의 배수가 아니면 우/하단 잉여를 잘라낸다.
    """
    n, h, w = arr.shape
    h2 = (h // 4) * 4
    w2 = (w // 4) * 4
    if (h, w) != (h2, w2):
        arr = arr[:, :h2, :w2]
        n, h, w = arr.shape

    blocked = arr.reshape(n, h // 4, 4, w // 4, 4).astype(np.float64)
    valid_blocked = (arr != 0).reshape(n, h // 4, 4, w // 4, 4)

    sum_valid = (blocked * valid_blocked).sum(axis=(2, 4))
    count_valid = valid_blocked.sum(axis=(2, 4))

    means = np.where(count_valid > 0, sum_valid / np.maximum(count_valid, 1), 0.0)
    return means.astype(arr.dtype)


def write_tif(out_path: Path, data: np.ndarray, profile: dict) -> None:
    """data: (bands, H, W) — profile 기반 multi-band TIF 저장."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = profile.copy()
    profile.update(count=data.shape[0], dtype=data.dtype.name)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)


def derive_scene_stem(k3a_grid_filename: str) -> str:
    """'1_K3A_..._B_grid.tif' → 'K3A_...' 로 정리. 출력 파일명 prefix 용."""
    stem = Path(k3a_grid_filename).stem  # e.g. "1_K3A_..._B_grid"
    stem = re.sub(r"^\d+_", "", stem)
    stem = re.sub(r"_[BGRN]_grid$", "", stem)
    return stem


def _save_record(record: dict, out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)


def _process_pair(
    label: str,
    pair: Dict[str, Dict[str, Path]],
    out_dir: Path,
    bands: List[str],
    args,
    logger: logging.Logger,
) -> str:
    """페어 1개를 처리하고 status('success' / 'skip' / 'fail') 를 반환합니다.

    어떤 결과든 페어별 JSON ({label}_{scene}_radsim.json) 을 항상 기록 — 사후
    triage 가 모든 페어의 상태를 한 곳에서 읽을 수 있게.
    """
    logger.info("\n[페어 %s] 시작", label)

    # 어느 K3A 밴드 하나라도 있으면 scene_stem 추정 가능
    any_k3a = next(iter(pair["K3A"].values()), None)
    scene_stem = derive_scene_stem(any_k3a.name) if any_k3a else f"unknown_{label}"

    out_tif = out_dir / f"{label}_{scene_stem}_simulated_2p5m.tif"
    out_json = out_dir / f"{label}_{scene_stem}_radsim.json"

    record: dict = {
        "pair_label": label,
        "scene_stem": scene_stem,
        "bands": bands,
        "k3a_grid_files": {b: str(pair["K3A"][b]) for b in bands if b in pair["K3A"]},
        "s2_grid_files": {b: str(pair["S2"][b]) for b in bands if b in pair["S2"]},
    }

    missing_k3a = [b for b in bands if b not in pair["K3A"]]
    missing_s2 = [b for b in bands if b not in pair["S2"]]
    if missing_k3a or missing_s2:
        logger.warning(
            "  밴드 누락 (K3A=%s, S2=%s) → 스킵",
            missing_k3a, missing_s2,
        )
        record.update({
            "status": "skip",
            "reason_class": "MissingBands",
            "reason_detail": f"K3A missing={missing_k3a}, S2 missing={missing_s2}",
        })
        _save_record(record, out_json)
        return "skip"

    # 기존 성공 결과가 있으면 재처리 안 함 (--overwrite 로 강제 재실행 가능)
    if out_tif.exists() and out_json.exists() and not args.overwrite:
        try:
            with open(out_json, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if existing.get("status") == "success":
                logger.info("  이미 성공 결과 존재 → 스킵 (%s)", out_tif.name)
                return "skip"
        except Exception:
            pass  # 기존 JSON 손상 등 → 재처리

    try:
        k3a_2p5m, k3a_profile = load_band_stack(pair["K3A"], bands)  # (n, H4, W4)
        s2_10m, _ = load_band_stack(pair["S2"], bands)                # (n, H, W)

        logger.info(
            "  K3A shape (2.5m): %s | S2 shape (10m): %s",
            k3a_2p5m.shape, s2_10m.shape,
        )

        k3a_10m = aggregate_4x4_mean(k3a_2p5m)

        # K3A grid 와 S2 grid 는 ratio=4 로 outer rect 을 공유하도록 설계되어 있지만
        # 부동소수 ceil 누락 등의 이유로 1~2 픽셀 어긋날 수 있음 → 공통 부분만 사용.
        if k3a_10m.shape[1:] != s2_10m.shape[1:]:
            logger.warning(
                "  K3A 10m %s ≠ S2 10m %s → 공통 부분만 사용",
                k3a_10m.shape, s2_10m.shape,
            )
            h = min(k3a_10m.shape[1], s2_10m.shape[1])
            w = min(k3a_10m.shape[2], s2_10m.shape[2])
            k3a_10m = k3a_10m[:, :h, :w]
            s2_10m = s2_10m[:, :h, :w]

        # n_valid 진단을 위해 valid_mask 를 미리 계산해 record 에 남김
        valid_mask = ((k3a_10m != 0).all(axis=0)) & ((s2_10m != 0).all(axis=0))
        record["n_valid_at_10m"] = int(valid_mask.sum())
        record["grid_size_10m"] = list(s2_10m.shape[1:])

        result = radiometric_simulation(
            img_s2=s2_10m,
            img_k3a=k3a_10m,
            band_names=bands,
            mad_max_iter=args.max_iter,
            mad_epsilon=args.epsilon,
            pif_threshold=args.pif_threshold,
            nodata_value=0.0,
        )

        # 학습된 (a, b) 를 K3A 2.5m 원본에 적용 — 출력은 2.5m simulated K3A
        simulated_2p5m = apply_linear_normalization(k3a_2p5m, result.band_coefficients)

        # K3A nodata(0) 위치는 출력도 0 으로 유지
        nodata_mask = (k3a_2p5m == 0)
        simulated_2p5m[nodata_mask] = 0
        simulated_2p5m = np.clip(simulated_2p5m, 0, 65535).astype(np.uint16)

        write_tif(out_tif, simulated_2p5m, k3a_profile)

        record.update({
            "status": "success",
            "iterations": int(result.ir_mad_result.iterations),
            "n_pifs": int(result.ir_mad_result.pif_mask.sum()),
            "canonical_correlations": [
                float(x) for x in result.ir_mad_result.canonical_correlations
            ],
            "band_coefficients": [
                {
                    "band_name": c.band_name,
                    "slope": c.slope,
                    "intercept": c.intercept,
                    "r_squared": c.r_squared,
                    "rmse": c.rmse,
                    "n_pifs": c.n_pifs,
                }
                for c in result.band_coefficients
            ],
            "output_tif": str(out_tif),
        })
        _save_record(record, out_json)

        logger.info("  저장: %s", out_tif.name)
        logger.info("        %s", out_json.name)
        return "success"

    except Exception as e:
        logger.exception("  실패 (%s): %s", type(e).__name__, e)
        record.update({
            "status": "fail",
            "reason_class": type(e).__name__,
            "reason_detail": str(e),
        })
        _save_record(record, out_json)
        return "fail"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IR-MAD 기반 방사모사 배치 (K3A 2.5m → S2 방사 특성)",
    )
    parser.add_argument(
        "--grid-dir", default=None,
        help=f"페어 단위 grid TIF 디렉토리 (기본: config/paths.json grid_dir = {_PATHS_CONFIG['grid_dir']})",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help=f"결과 저장 디렉토리 (기본: config/paths.json rad_sim_dir = {_PATHS_CONFIG['rad_sim_dir']})",
    )
    parser.add_argument(
        "--bands", nargs="+", default=K3A_BAND_ORDER,
        help="처리할 밴드 (기본: Blue Green Red NIR)",
    )
    parser.add_argument("--max-iter", type=int, default=50, help="IR-MAD 최대 반복")
    parser.add_argument("--epsilon", type=float, default=1e-3, help="IR-MAD 수렴 임계값")
    parser.add_argument("--pif-threshold", type=float, default=0.95, help="PIF 확률 임계값")
    parser.add_argument("--overwrite", action="store_true", help="기존 출력 덮어쓰기")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("rad_sim_batch")

    grid_dir = (
        Path(args.grid_dir).resolve()
        if args.grid_dir
        else (PROJECT_ROOT / _PATHS_CONFIG["grid_dir"]).resolve()
    )
    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else (PROJECT_ROOT / _PATHS_CONFIG["rad_sim_dir"]).resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if not grid_dir.exists():
        logger.error("grid 디렉토리를 찾을 수 없습니다: %s", grid_dir)
        sys.exit(1)

    bands = list(args.bands)
    pairs = discover_pairs(grid_dir)

    if not pairs:
        logger.error("페어를 찾지 못했습니다: %s", grid_dir)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("방사모사 배치 시작: 페어 %d개", len(pairs))
    logger.info("  grid_dir: %s", grid_dir)
    logger.info("  out_dir : %s", out_dir)
    logger.info("  bands   : %s", bands)
    logger.info("=" * 60)

    success = 0
    skip = 0
    fail = 0

    # rasterio 호출 전체를 PROJ_DATA 로 격리 — system PROJ (PostgreSQL/PostGIS 등)
    # 가 잡혀 proj.db schema 버전 충돌이 나는 걸 막는다.
    env_kwargs = {"PROJ_LIB": PROJ_DATA} if PROJ_DATA else {}
    with rasterio.env.Env(**env_kwargs):
        for label in sorted(pairs.keys(), key=lambda s: int(s) if s.isdigit() else 0):
            outcome = _process_pair(label, pairs[label], out_dir, bands, args, logger)
            if outcome == "success":
                success += 1
            elif outcome == "skip":
                skip += 1
            else:
                fail += 1

    summary = {
        "success": success,
        "skip": skip,
        "fail": fail,
        "total": success + skip + fail,
        "out_dir": str(out_dir),
        "bands": bands,
    }
    summary_path = out_dir / "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("배치 종료 — 성공 %d, 스킵 %d, 실패 %d", success, skip, fail)
    logger.info("summary: %s", summary_path)
    if fail > 0:
        logger.info("실패 페어 진단·재처리 후보 탐색: "
                    "python module3_simulation/scripts/triage_failures.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
