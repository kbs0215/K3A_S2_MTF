"""sr_inference.py - 비교군의 SR 함수 통합 인터페이스.

각 함수 시그니처: (lr_bgrn_dn, ...) -> hr_bgrn_dn (uint16 raw DN, 4× upsample)

- bicubic_sr  : scipy bicubic. 가중치 불필요. (method B)
- prithvi_sr  : best_sr_model.pth. 224×224 학습 → 큰 입력은 tile_stitch 로 처리. (method D)
"""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy.ndimage import zoom

from module5_SR_comparison.src.prithvi_sr import FullSRModel, prithvi_sr_infer_tile
from module5_SR_comparison.src.tile_stitch import tile_stitch_inference


# =====================================================================
# B : Bicubic
# =====================================================================

def bicubic_sr(lr_bgrn_dn: np.ndarray, scale: int = 4) -> np.ndarray:
    """Per-channel bicubic ×scale upsample.

    scipy.ndimage.zoom(order=3) = cubic spline interpolation.
    uint16 DN 보존 (clip 후 캐스팅).
    """
    assert lr_bgrn_dn.ndim == 3 and lr_bgrn_dn.shape[0] == 4
    lr_f = lr_bgrn_dn.astype(np.float32)
    out_bands = [
        zoom(lr_f[b], zoom=scale, order=3, mode="reflect", grid_mode=False)
        for b in range(lr_f.shape[0])
    ]
    out = np.stack(out_bands, axis=0)
    return np.clip(out, 0.0, 65535.0).astype(np.uint16)


# =====================================================================
# D : Prithvi-SR (tile-and-stitch)
# =====================================================================

def prithvi_sr(
    lr_bgrn_dn: np.ndarray,
    model: FullSRModel,
    device: Union[str, "torch.device"] = "cpu",  # noqa: F821
    tile_lr: int = 224,
    stride_lr: int = 144,
) -> np.ndarray:
    """Prithvi-SR 4×. 입력이 224×224 면 단일 타일, 그 외엔 sliding window 평균."""
    assert lr_bgrn_dn.ndim == 3 and lr_bgrn_dn.shape[0] == 4
    _, H, W = lr_bgrn_dn.shape

    if (H, W) == (tile_lr, tile_lr):
        return prithvi_sr_infer_tile(model, lr_bgrn_dn, device=device)

    def _infer(tile: np.ndarray) -> np.ndarray:
        return prithvi_sr_infer_tile(model, tile, device=device)

    return tile_stitch_inference(
        lr_chw=lr_bgrn_dn,
        infer_fn=_infer,
        tile_lr=tile_lr,
        stride_lr=stride_lr,
        scale=4,
    )
