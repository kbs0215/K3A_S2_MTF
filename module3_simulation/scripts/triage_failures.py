"""triage_failures.py - 방사모사 실패 페어 진단 + 재처리 후보 탐색

흐름:
  1) data/output/rad_sim/*_radsim.json 모두 읽기
  2) status="fail" 페어 추출 (skip 도 별도 카운트)
  3) 각 실패 페어:
     - K3A scene 을 data/interim/k3a_extracted/{scene}/ 에서 로드 → bbox + 촬영일
     - 사용된 S2 SAFE 식별 (S2 grid 파일명에서 tile+date 추출)
     - 로컬 다른 S2 SAFE 후보 탐색 (K3A bbox 포함 + 사용된 SAFE 와 다름 + ±N일)
  4) 권장 조치 분류:
     - try-local-alt:   로컬 대체 S2 후보 있음
     - needs-download:  로컬에 적합 S2 없음 → Copernicus 검색·다운로드 필요
     - manual:          K3A scene 없거나 식별 불가 (드문 케이스)
  5) 사람용 콘솔 출력 + 기계용 _triage_plan.json 저장

기본은 read-only. 실제 재처리는 사용자가 보고 수동으로 module2/3 재실행.
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import rasterio
from rasterio.warp import transform_bounds

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from module1_data_download.src.k3a_loader import load_k3a_scene
from shared.utils.paths import load_paths
from shared.utils.proj_env import PROJ_DATA  # noqa: F401  GDAL/PROJ 환경 초기화


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


# S2 grid 파일명 → tile + datetime 추출 (lenient)
_S2_TILE_DT_RE = re.compile(r"(T[A-Z0-9]+)_(\d{8}T\d{6})")


def parse_s2_grid_filename(grid_path_str: str) -> Optional[Tuple[str, str]]:
    """grid 파일명 stem 안의 (Txxxxx, YYYYMMDDThhmmss) 첫 매치를 반환."""
    stem = Path(grid_path_str).stem
    m = _S2_TILE_DT_RE.search(stem)
    return (m.group(1), m.group(2)) if m else None


def build_s2_safe_index(s2_dir: Path) -> Dict[Tuple[str, str], Path]:
    """data/raw/s2/*.SAFE 를 (tile, datetime) → Path 딕셔너리로 인덱싱.

    SAFE 명 예: S2A_MSIL1C_20151021T022702_N0500_R003_T52SDG_20231016T205424.SAFE
                |    |        |              |     |     |     |
                0    1        2 (sensing)    3     4     5     6 (process)
    """
    index: Dict[Tuple[str, str], Path] = {}
    for safe in s2_dir.glob("*.SAFE"):
        parts = safe.name.split("_")
        if len(parts) < 6:
            continue
        sensing_dt = parts[2]
        tile = parts[5]
        index[(tile, sensing_dt)] = safe
    return index


def get_s2_bbox_wgs84(safe_dir: Path, cache: dict) -> Optional[tuple]:
    """SAFE 의 WGS84 axis-aligned bbox. B02 jp2 의 native CRS bounds 를 변환."""
    if safe_dir in cache:
        return cache[safe_dir]
    try:
        b02 = next(safe_dir.rglob("*_B02.jp2"), None)
        if b02 is None:
            cache[safe_dir] = None
            return None
        with rasterio.open(b02) as src:
            wgs = transform_bounds(
                src.crs, "EPSG:4326",
                src.bounds.left, src.bounds.bottom,
                src.bounds.right, src.bounds.top,
                densify_pts=21,
            )
        cache[safe_dir] = wgs
        return wgs
    except Exception:
        cache[safe_dir] = None
        return None


def bbox_contains(outer: tuple, inner: tuple) -> bool:
    """outer 가 inner 를 (경계 포함) 완전히 덮으면 True."""
    return (outer[0] <= inner[0] and outer[1] <= inner[1]
            and outer[2] >= inner[2] and outer[3] >= inner[3])


def get_s2_date_from_safe_name(safe_name: str) -> datetime:
    try:
        date_str = safe_name.split("_")[2][:8]
        return datetime.strptime(date_str, "%Y%m%d")
    except Exception:
        return datetime(2020, 1, 1)


def find_local_alternatives(
    k3a_bbox: tuple,
    k3a_acq_date: datetime,
    s2_dir: Path,
    exclude_safe_name: Optional[str],
    max_date_diff_days: int,
    bbox_cache: dict,
) -> List[Tuple[Path, int]]:
    """로컬 SAFE 중 K3A bbox 포함 + 사용된 SAFE 와 다름 + 날짜 윈도우 안.

    Returns: [(safe_path, abs_date_diff_days), ...] 날짜 가까운 순.
    """
    candidates: List[Tuple[Path, int]] = []
    for safe in s2_dir.glob("*.SAFE"):
        if exclude_safe_name and safe.name == exclude_safe_name:
            continue
        bbox = get_s2_bbox_wgs84(safe, bbox_cache)
        if bbox is None or not bbox_contains(bbox, k3a_bbox):
            continue
        diff = abs((get_s2_date_from_safe_name(safe.name) - k3a_acq_date).days)
        if diff > max_date_diff_days:
            continue
        candidates.append((safe, diff))
    candidates.sort(key=lambda x: x[1])
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="방사모사 실패 페어 진단 + 대체 S2 후보 제안 (read-only)",
    )
    parser.add_argument("--rad-sim-dir", default=None,
                        help="방사모사 결과 디렉토리 (기본: paths.json rad_sim_dir)")
    parser.add_argument("--s2-dir", default=None,
                        help="로컬 S2 SAFE 디렉토리 (기본: paths.json s2_raw_dir)")
    parser.add_argument("--k3a-dir", default=None,
                        help="K3A scene 디렉토리 (기본: paths.json k3a_extracted_dir)")
    parser.add_argument("--max-date-diff", type=int, default=30,
                        help="대체 S2 날짜 차이 한도 (기본 30일)")
    parser.add_argument("--out", default=None,
                        help="triage plan JSON 저장 경로 (기본: rad_sim_dir/_triage_plan.json)")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("triage")

    paths = load_paths()
    rad_sim_dir = (Path(args.rad_sim_dir).resolve() if args.rad_sim_dir
                   else (PROJECT_ROOT / paths["rad_sim_dir"]).resolve())
    s2_dir = (Path(args.s2_dir).resolve() if args.s2_dir
              else (PROJECT_ROOT / paths["s2_raw_dir"]).resolve())
    k3a_dir = (Path(args.k3a_dir).resolve() if args.k3a_dir
               else (PROJECT_ROOT / paths["k3a_extracted_dir"]).resolve())
    plan_path = (Path(args.out).resolve() if args.out
                 else rad_sim_dir / "_triage_plan.json")

    if not rad_sim_dir.exists():
        logger.error("rad_sim_dir 없음: %s", rad_sim_dir)
        sys.exit(1)

    # 페어 JSON 읽기 (_summary.json / _triage_plan.json 등 underscore prefix 는 제외)
    records: List[dict] = []
    for jp in sorted(rad_sim_dir.glob("*_radsim.json")):
        if jp.name.startswith("_"):
            continue
        try:
            with open(jp, "r", encoding="utf-8") as f:
                records.append(json.load(f))
        except Exception as e:
            logger.warning("JSON 읽기 실패 %s: %s", jp.name, e)

    if not records:
        logger.error("rad_sim_dir 에 페어 JSON 없음: %s", rad_sim_dir)
        sys.exit(1)

    failures = [r for r in records if r.get("status") == "fail"]
    successes = [r for r in records if r.get("status") == "success"]
    skips = [r for r in records if r.get("status") == "skip"]

    logger.info("=" * 60)
    logger.info("방사모사 결과 — 성공 %d, 실패 %d, 스킵 %d (총 %d)",
                len(successes), len(failures), len(skips), len(records))
    logger.info("=" * 60)

    plan: Dict = {
        "summary": {
            "success": len(successes),
            "fail": len(failures),
            "skip": len(skips),
            "total": len(records),
        },
        "actions": [],
    }

    if not failures:
        logger.info("실패 페어 없음. 종료.")
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)
        return

    # SAFE 인덱스 + bbox 캐시 (재사용)
    safe_index = build_s2_safe_index(s2_dir)
    bbox_cache: dict = {}

    n_local = 0
    n_download = 0
    n_manual = 0

    for rec in failures:
        label = str(rec.get("pair_label", "?"))
        scene_stem = rec.get("scene_stem", "?")
        reason_class = rec.get("reason_class", "?")
        reason_detail = rec.get("reason_detail", "")
        n_valid = rec.get("n_valid_at_10m")

        logger.info("\n[페어 %s] 실패: %s", label, reason_class)
        logger.info("  K3A scene  : %s", scene_stem)
        if n_valid is not None:
            grid_size = rec.get("grid_size_10m", "?")
            logger.info("  grid 10m   : %s, n_valid=%d", grid_size, n_valid)
        logger.info("  사유       : %s", reason_detail[:200])

        # 사용된 S2 SAFE 식별 — S2 grid 파일명에서 tile + datetime 추출
        s2_grid_files = rec.get("s2_grid_files", {})
        used_safe_path: Optional[Path] = None
        if s2_grid_files:
            first_path = next(iter(s2_grid_files.values()))
            parsed = parse_s2_grid_filename(first_path)
            if parsed:
                tile, dt = parsed
                used_safe_path = safe_index.get((tile, dt))

        used_safe_name = used_safe_path.name if used_safe_path else None
        logger.info("  사용 S2    : %s", used_safe_name or "(식별 불가)")

        # K3A scene 로드
        scene_dir = k3a_dir / scene_stem
        if not scene_dir.exists():
            logger.warning("  K3A scene 디렉토리 없음 → manual")
            plan["actions"].append({
                "pair_label": label,
                "scene_stem": scene_stem,
                "action": "manual",
                "note": f"K3A scene 디렉토리 없음: {scene_dir}",
                "reason_class": reason_class,
                "reason_detail": reason_detail,
            })
            n_manual += 1
            continue

        try:
            scene = load_k3a_scene(scene_dir)
        except Exception as e:
            logger.warning("  K3A scene 로드 실패: %s", e)
            plan["actions"].append({
                "pair_label": label,
                "scene_stem": scene_stem,
                "action": "manual",
                "note": f"K3A scene 로드 실패: {e}",
                "reason_class": reason_class,
                "reason_detail": reason_detail,
            })
            n_manual += 1
            continue

        if scene.bounds is None:
            logger.warning("  K3A bbox 없음 → manual")
            plan["actions"].append({
                "pair_label": label,
                "scene_stem": scene_stem,
                "action": "manual",
                "note": "K3A 1RCoordinate.txt 없음 / bbox 추출 실패",
                "reason_class": reason_class,
                "reason_detail": reason_detail,
            })
            n_manual += 1
            continue

        k3a_bbox = scene.bounds.to_bbox()
        acq_date_str = scene.get_acquisition_date() or "20200101"
        acq_date = datetime.strptime(acq_date_str, "%Y%m%d")

        # 로컬 대체 S2 탐색
        alternatives = find_local_alternatives(
            k3a_bbox, acq_date, s2_dir, used_safe_name,
            args.max_date_diff, bbox_cache,
        )

        if alternatives:
            logger.info("  로컬 대체 후보 %d개:", len(alternatives))
            for safe, diff in alternatives[:3]:
                logger.info("    - %s (Δ=%d일)", safe.name, diff)
            plan["actions"].append({
                "pair_label": label,
                "scene_stem": scene_stem,
                "action": "try-local-alt",
                "k3a_bbox": list(k3a_bbox),
                "k3a_acq_date": acq_date_str,
                "used_s2_safe": used_safe_name,
                "alternatives": [
                    {"safe": safe.name, "date_diff_days": diff}
                    for safe, diff in alternatives[:5]
                ],
                "reason_class": reason_class,
                "reason_detail": reason_detail,
                "note": (
                    "사용된 S2 SAFE 를 임시로 옮기거나 삭제한 뒤 module2 재실행 → "
                    "module3 재실행. 또는 module2.run_coregistration_pipeline 을 "
                    "직접 호출해 (K3A bands × 대체 S2 SAFE) 페어를 처리."
                ),
            })
            n_local += 1
        else:
            window_start = acq_date - timedelta(days=args.max_date_diff)
            window_end = acq_date + timedelta(days=args.max_date_diff)
            logger.info("  로컬 대체 없음 → 다운로드 필요 (%s ~ %s)",
                        window_start.date(), window_end.date())
            plan["actions"].append({
                "pair_label": label,
                "scene_stem": scene_stem,
                "action": "needs-download",
                "k3a_bbox": list(k3a_bbox),
                "k3a_acq_date": acq_date_str,
                "search_window": {
                    "start": window_start.strftime("%Y-%m-%d"),
                    "end": window_end.strftime("%Y-%m-%d"),
                },
                "used_s2_safe": used_safe_name,
                "reason_class": reason_class,
                "reason_detail": reason_detail,
                "note": (
                    f"K3A bbox {k3a_bbox} 를 포함하고 ±{args.max_date_diff}일 "
                    "윈도우 안에 있는 S2 L1C 을 module1/scripts (s2_download) 로 "
                    "추가 다운로드 후 module2/3 재실행."
                ),
            })
            n_download += 1

    plan["summary"]["fail_breakdown"] = {
        "try-local-alt": n_local,
        "needs-download": n_download,
        "manual": n_manual,
    }

    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    logger.info("\n%s", "=" * 60)
    logger.info("triage 완료 — 로컬대체 %d, 다운로드필요 %d, 수동검토 %d",
                n_local, n_download, n_manual)
    logger.info("계획 저장: %s", plan_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
