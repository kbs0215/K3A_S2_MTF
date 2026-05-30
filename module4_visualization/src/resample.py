"""resample.py - K3A 2.5m -> 10m 다운샘플 헬퍼.

- aggregate_4x4_mean: module3 에서 재노출 (블록 평균, NoData permissive)
- bilinear_4x_downsample: "단순 보간" baseline (PIL bilinear)
"""

from typing import Optional

import numpy as np
from PIL import Image

from module3_simulation.src.mtf import aggregate_4x4_mean as _aggregate_4x4_mean


def aggregate_4x4_mean(arr: np.ndarray) -> np.ndarray:
    """module3_simulation.src.mtf.aggregate_4x4_mean 재노출."""
    return _aggregate_4x4_mean(arr)


def bilinear_4x_downsample(arr: np.ndarray, nodata_value: Optional[float] = 0.0) -> np.ndarray:
    """K3A 2.5m -> 10m 단순 보간 다운샘플 (bilinear).

    Args:
        arr: (H,W) 또는 (N,H,W). 입력이 4 의 배수 차원이 아닐 시 잘라냄 (aggregate_4x4_mean 동작 정렬).
        nodata_value: 보간 후 nodata 영역 복원에 사용. None 이면 보존 안 함.

    Returns:
        4x 다운샘플된 동일 dtype 배열. NoData 픽셀(>=50% NoData 인 4x4 블록)은 nodata_value 로 강제 설정.
    """
    if arr.ndim == 2:
        return _bilinear_2d(arr, nodata_value)
    if arr.ndim == 3:
        return np.stack([_bilinear_2d(arr[i], nodata_value) for i in range(arr.shape[0])], axis=0)
    raise ValueError(f"expected 2D or 3D array, got {arr.ndim}D")


def _bilinear_2d(arr2d: np.ndarray, nodata_value: Optional[float]) -> np.ndarray:
    h, w = arr2d.shape
    h2 = (h // 4) * 4
    w2 = (w // 4) * 4
    arr2d = arr2d[:h2, :w2]
    h_out, w_out = h2 // 4, w2 // 4

    src_dtype = arr2d.dtype
    pil_mode = "F"
    img = Image.fromarray(arr2d.astype(np.float32), mode=pil_mode)
    img_ds = img.resize((w_out, h_out), resample=Image.BILINEAR)
    out = np.array(img_ds, dtype=np.float32)

    if nodata_value is not None:
        nodata_mask = (arr2d == nodata_value).astype(np.float32)
        nm_img = Image.fromarray(nodata_mask, mode="F")
        nm_ds = np.array(nm_img.resize((w_out, h_out), resample=Image.BILINEAR), dtype=np.float32)
        out[nm_ds >= 0.5] = float(nodata_value)

    return out.astype(src_dtype)
