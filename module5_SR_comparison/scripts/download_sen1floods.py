"""download_sen1floods.py - Sen1Floods11 HandLabeled subset downloader

흐름:
  1) Sen1Floods11 split CSV (train/val/test) 합쳐서 HandLabeled chip 인덱스 구축
  2) --events 로 지정된 region 만 필터 (기본 Bolivia/India/USA)
  3) event 당 --max-per-event 개까지 S2Hand + LabelHand 다운로드
  4) S2Hand 13밴드 → BGRN 4밴드 (B02/B03/B04/B08) 추출, LZW 압축 후 저장
  5) 원본 13밴드 TIFF 는 기본 삭제 (--keep-13band 면 유지)

데이터:
  GCS bucket gs://sen1floods11 는 public HTTP 접근 가능
  base URL: https://storage.googleapis.com/sen1floods11/v1.1/
  L1C TOA reflectance × 10000 (uint16)
"""

import argparse
import csv
import io
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

import rasterio
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from shared.utils.proj_env import PROJ_DATA  # noqa: F401  GDAL/PROJ 환경 초기화


GCS_BASE = "https://storage.googleapis.com/sen1floods11/v1.1"
SPLIT_FILES = [
    "flood_train_data.csv",
    "flood_valid_data.csv",
    "flood_test_data.csv",
]

# Sen1Floods11 S2Hand 밴드 순서 (1-indexed):
#   1=B01  2=B02(Blue)  3=B03(Green)  4=B04(Red)  5=B05  6=B06  7=B07
#   8=B08(NIR)  9=B8A  10=B09  11=B10  12=B11  13=B12
BGRN_BANDS: List[int] = [2, 3, 4, 8]
BGRN_DESC: Tuple[str, ...] = ("B02_Blue", "B03_Green", "B04_Red", "B08_NIR")

DEFAULT_EVENTS: List[str] = ["Bolivia", "India", "USA"]


def fetch_chip_index() -> List[Tuple[str, str]]:
    """모든 split CSV 를 읽어 (region, chip_id) 페어를 dedup 정렬해 반환."""
    pairs: Set[Tuple[str, str]] = set()
    for split_name in SPLIT_FILES:
        url = f"{GCS_BASE}/splits/flood_handlabeled/{split_name}"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        reader = csv.reader(io.StringIO(resp.text))
        for row in reader:
            if not row or not row[0].strip():
                continue
            # row[0] 형식: "Region_ChipID_S1Hand.tif"
            first = row[0].strip()
            stem = first.replace("_S1Hand.tif", "").replace("_S2Hand.tif", "")
            parts = stem.split("_", 1)
            if len(parts) != 2:
                continue
            region, chip_id = parts
            pairs.add((region, chip_id))
    return sorted(pairs)


def download_file(url: str, dst: Path, overwrite: bool = False) -> bool:
    """파일 다운로드. 이미 존재하고 overwrite=False 면 False 반환."""
    if dst.exists() and not overwrite:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with requests.get(url, stream=True, timeout=180) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
    tmp.replace(dst)
    return True


def extract_bgrn(src_tif: Path, dst_tif: Path) -> None:
    """13밴드 S2Hand 에서 BGRN 4밴드만 추출해 LZW 압축 GeoTIFF 로 저장."""
    with rasterio.open(src_tif) as src:
        data = src.read(BGRN_BANDS)
        profile = src.profile.copy()
    profile.update(
        count=len(BGRN_BANDS),
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    with rasterio.open(dst_tif, "w", **profile) as dst:
        dst.write(data)
        dst.descriptions = BGRN_DESC


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    here = Path(__file__).resolve()
    module_root = here.parents[1]
    default_out = module_root / "data" / "sen1floods11"

    parser = argparse.ArgumentParser(
        description="Download Sen1Floods11 HandLabeled subset (BGRN extract).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--events",
        nargs="+",
        default=DEFAULT_EVENTS,
        help="다운로드할 region 이름 목록",
    )
    parser.add_argument(
        "--max-per-event",
        type=int,
        default=50,
        help="event 당 최대 chip 수",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_out,
        help="출력 루트",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 파일 덮어쓰기",
    )
    parser.add_argument(
        "--keep-13band",
        action="store_true",
        help="원본 13밴드 S2Hand TIFF 도 유지",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out_root: Path = args.output_dir
    s2_dir = out_root / "S2Hand"
    label_dir = out_root / "LabelHand"
    full13_dir = s2_dir / "_full13"

    print(f"[1/3] Fetching chip index from GCS …")
    all_pairs = fetch_chip_index()
    print(f"  found {len(all_pairs)} HandLabeled chips across all splits")

    available_regions = sorted({r for r, _ in all_pairs})
    unknown = [ev for ev in args.events if ev not in available_regions]
    if unknown:
        print(f"  WARNING: unknown regions (no chips found): {unknown}")
        print(f"  available regions: {available_regions}")

    print(f"[2/3] Filtering events: {args.events} (max {args.max_per_event}/event)")
    selected: List[Tuple[str, str]] = []
    per_event = {ev: 0 for ev in args.events}
    for region, chip_id in all_pairs:
        if region not in per_event:
            continue
        if per_event[region] >= args.max_per_event:
            continue
        selected.append((region, chip_id))
        per_event[region] += 1
    for ev in args.events:
        print(f"  {ev}: {per_event[ev]} chips")
    print(f"  total: {len(selected)} chips")

    if not selected:
        print("  no chips selected — exiting.")
        return 1

    print(f"[3/3] Downloading to {out_root}")
    n_s2_new = n_s2_skip = n_label_new = n_label_skip = n_fail = 0

    for i, (region, chip_id) in enumerate(selected, 1):
        stem = f"{region}_{chip_id}"
        s2_name = f"{stem}_S2Hand.tif"
        label_name = f"{stem}_LabelHand.tif"
        s2_url = f"{GCS_BASE}/data/flood_events/HandLabeled/S2Hand/{s2_name}"
        label_url = f"{GCS_BASE}/data/flood_events/HandLabeled/LabelHand/{label_name}"

        s2_13_path = full13_dir / s2_name
        s2_bgrn_path = s2_dir / s2_name
        label_path = label_dir / label_name

        try:
            need_extract = args.overwrite or not s2_bgrn_path.exists()
            if need_extract:
                if download_file(s2_url, s2_13_path, overwrite=args.overwrite):
                    n_s2_new += 1
                else:
                    n_s2_skip += 1
                extract_bgrn(s2_13_path, s2_bgrn_path)
                if not args.keep_13band:
                    s2_13_path.unlink(missing_ok=True)
            else:
                n_s2_skip += 1

            if download_file(label_url, label_path, overwrite=args.overwrite):
                n_label_new += 1
            else:
                n_label_skip += 1
        except Exception as e:
            print(f"  [{i}/{len(selected)}] {stem} FAIL: {e}")
            n_fail += 1
            continue

        if i % 10 == 0 or i == len(selected):
            print(f"  [{i}/{len(selected)}] {stem} done")

    if not args.keep_13band and full13_dir.exists():
        try:
            full13_dir.rmdir()
        except OSError:
            pass

    print()
    print(f"S2  : {n_s2_new} new, {n_s2_skip} skipped, {n_fail} failed")
    print(f"Label: {n_label_new} new, {n_label_skip} skipped")
    print(f"Output: {out_root.resolve()}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
