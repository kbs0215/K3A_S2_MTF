"""rpc.py - K3A RPC 사이드카 파일 탐색 및 파싱

K3A 영상의 RPC (Rational Polynomial Coefficients) 정보를 _rpc.txt 외부
사이드카에서 읽어 rasterio.rpc.RPC 객체로 변환합니다.

GDAL 은 {tif_stem}_rpc.txt 만 자동 인식하므로, 멀티스펙트럼 통합
RPC(_M_rpc.txt) 만 있는 씬은 별도 처리(ortho.py 참고) 가 필요합니다.
"""

from pathlib import Path
from typing import List, Optional

from rasterio.rpc import RPC


def _find_rpc_file(tif_path: Path) -> Optional[Path]:
    """TIF에 대응하는 _rpc.txt 파일을 찾습니다.

    1순위: {stem}_rpc.txt (밴드별 개별 RPC, 예: _B_rpc.txt)
    2순위: {scene_id}_M_rpc.txt (멀티스펙트럼 통합 RPC)
    """
    per_band = tif_path.with_name(tif_path.stem + "_rpc.txt")
    if per_band.exists():
        return per_band
    scene_id = tif_path.stem.rsplit("_", 1)[0]  # '_B' 제거 → 씬 ID
    multi = tif_path.parent / f"{scene_id}_M_rpc.txt"
    if multi.exists():
        return multi
    return None


def _parse_rpc_txt(rpc_file: Path) -> RPC:
    """K3A _rpc.txt 파일을 파싱하여 rasterio RPC 객체를 반환합니다."""
    kv: dict[str, str] = {}
    with open(rpc_file, "r") as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                k, v = line.split(":", 1)
                kv[k.strip()] = v.strip()

    def val(key: str) -> float:
        return float(kv[key].split()[0])

    def coeffs(prefix: str) -> List[float]:
        return [float(kv[f"{prefix}_{i}"].split()[0]) for i in range(1, 21)]

    return RPC(
        height_off=val("HEIGHT_OFF"),
        height_scale=val("HEIGHT_SCALE"),
        lat_off=val("LAT_OFF"),
        lat_scale=val("LAT_SCALE"),
        line_den_coeff=coeffs("LINE_DEN_COEFF"),
        line_num_coeff=coeffs("LINE_NUM_COEFF"),
        line_off=val("LINE_OFF"),
        line_scale=val("LINE_SCALE"),
        long_off=val("LONG_OFF"),
        long_scale=val("LONG_SCALE"),
        samp_den_coeff=coeffs("SAMP_DEN_COEFF"),
        samp_num_coeff=coeffs("SAMP_NUM_COEFF"),
        samp_off=val("SAMP_OFF"),
        samp_scale=val("SAMP_SCALE"),
    )
