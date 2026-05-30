"""stretch.py - figure 디스플레이용 퍼센타일 스트레치 (NoData=0 인지).

기본은 채널별 독립 스트레치 (per-channel) — RS 표준 디스플레이 방식.
입력 영상의 DN 스케일이 달라도 (K3A raw 12-bit vs S2 reflectance) 패널 간
시각 톤이 균일해져 컨텐츠 비교가 쉬워진다.
"""

from typing import Optional, Tuple

import numpy as np


def percentile_stretch(
    rgb: np.ndarray,
    pmin: float = 2.0,
    pmax: float = 98.0,
    nodata_value: float = 0.0,
    vmin_vmax: Optional[Tuple[float, float]] = None,
    per_channel: bool = True,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """(3,H,W) RGB 스택을 [0,1] float32 로 스트레치.

    Args:
        rgb: shape (3,H,W). dtype 임의 (uint16/float).
        pmin/pmax: 퍼센타일 범위.
        nodata_value: 통계에서 제외할 값.
        vmin_vmax: 외부에서 주어지면 그것을 그대로 사용 (shared stretch). per_channel=True 일 때는
            (3,2) array-like 도 허용 ([[vmin_R,vmax_R],[vmin_G,vmax_G],[vmin_B,vmax_B]]).
            scalar (vmin,vmax) 면 모든 채널에 동일 적용.
        per_channel: True 면 R/G/B 각각 독립 퍼센타일 (기본). False 면 3채널 통합 단일 vmin/vmax.

    Returns:
        (stretched (H,W,3) float32 in [0,1], 대표 (vmin,vmax) — per_channel 시 (G채널) 값)
    """
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError(f"expected (3,H,W) RGB, got {rgb.shape}")

    arr = rgb.astype(np.float32)
    out = np.zeros_like(arr)

    if per_channel:
        rep_vmin, rep_vmax = 0.0, 1.0
        for c in range(3):
            ch = arr[c]
            if vmin_vmax is None:
                valid = ch[ch != nodata_value]
                if valid.size == 0:
                    vmin, vmax = 0.0, 1.0
                else:
                    vmin = float(np.percentile(valid, pmin))
                    vmax = float(np.percentile(valid, pmax))
                    if vmax <= vmin:
                        vmax = vmin + 1.0
            else:
                vv = np.asarray(vmin_vmax, dtype=np.float32)
                if vv.shape == (2,):
                    vmin, vmax = float(vv[0]), float(vv[1])
                elif vv.shape == (3, 2):
                    vmin, vmax = float(vv[c, 0]), float(vv[c, 1])
                else:
                    raise ValueError(f"vmin_vmax shape {vv.shape} not supported")
            out[c] = np.clip((ch - vmin) / (vmax - vmin), 0.0, 1.0)
            if c == 1:  # G 채널을 대표값으로
                rep_vmin, rep_vmax = vmin, vmax
        return np.transpose(out, (1, 2, 0)).astype(np.float32), (rep_vmin, rep_vmax)

    # 3채널 통합 (legacy)
    if vmin_vmax is None:
        valid = arr[arr != nodata_value]
        if valid.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin = float(np.percentile(valid, pmin))
            vmax = float(np.percentile(valid, pmax))
            if vmax <= vmin:
                vmax = vmin + 1.0
    else:
        vmin, vmax = float(vmin_vmax[0]), float(vmin_vmax[1])
    out = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    return np.transpose(out, (1, 2, 0)).astype(np.float32), (vmin, vmax)
