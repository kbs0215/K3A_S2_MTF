"""ndwi.py - NDWI = (G − NIR) / (G + NIR) + Otsu threshold

학습·가중치 불필요한 spectral 기반 물탐지. SR 의 분광 충실도 평가용 (보조 metric).
입력은 raw DN BGRN, 출력은 binary uint8 water mask.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from skimage.filters import threshold_otsu


def compute_ndwi(bgrn: np.ndarray) -> np.ndarray:
    """BGRN 4채널 raw DN → NDWI [-1, 1] float32."""
    assert bgrn.shape[0] == 4
    g = bgrn[1].astype(np.float32)
    nir = bgrn[3].astype(np.float32)
    denom = g + nir
    denom = np.where(denom > 0, denom, 1.0)
    return (g - nir) / denom


def ndwi_water_mask(
    bgrn: np.ndarray,
    thresh: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """NDWI → binary water mask. thresh None 이면 Otsu 자동.

    반환: (mask uint8 [H,W], 적용된 threshold)
    """
    ndwi = compute_ndwi(bgrn)
    if thresh is None:
        finite = ndwi[np.isfinite(ndwi)]
        if finite.size == 0 or finite.min() == finite.max():
            thresh = 0.0
        else:
            try:
                thresh = float(threshold_otsu(finite))
            except Exception:
                thresh = 0.0
    mask = (ndwi > thresh).astype(np.uint8)
    return mask, float(thresh)
