"""tile_stitch.py - 큰 입력을 fixed-size 모델로 처리하는 sliding-window 추론.

Prithvi-SR 은 224×224 LR 입력에 학습됨. Sen1Floods11 chip 은 512×512.
타일링 → 각 타일 SR → 출력 겹침 영역 평균.

기본 stride=144 (overlap 80, ≈36%) — 3×3=9 타일 / 512 입력.
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np


def _tile_positions(length: int, tile: int, stride: int) -> List[int]:
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


def tile_stitch_inference(
    lr_chw: np.ndarray,
    infer_fn: Callable[[np.ndarray], np.ndarray],
    tile_lr: int = 224,
    stride_lr: int = 144,
    scale: int = 4,
) -> np.ndarray:
    """Sliding-window SR with average blending in overlap.

    Args:
        lr_chw: [C, H, W] uint16 raw DN
        infer_fn: 단일 타일 추론 함수, [C, tile_lr, tile_lr] -> [C, tile_lr*scale, tile_lr*scale]
        tile_lr: LR 타일 한 변
        stride_lr: LR 슬라이딩 stride
        scale: 업스케일 배율 (출력은 H*scale × W*scale)

    Returns:
        [C, H*scale, W*scale] uint16 raw DN
    """
    assert lr_chw.ndim == 3, f"expected CHW, got {lr_chw.shape}"
    C, H, W = lr_chw.shape
    H_hr, W_hr = H * scale, W * scale
    tile_hr = tile_lr * scale

    ys = _tile_positions(H, tile_lr, stride_lr)
    xs = _tile_positions(W, tile_lr, stride_lr)

    acc = np.zeros((C, H_hr, W_hr), dtype=np.float64)
    cnt = np.zeros((H_hr, W_hr), dtype=np.float64)

    for y in ys:
        for x in xs:
            tile_lr_data = lr_chw[:, y:y + tile_lr, x:x + tile_lr]
            tile_hr_data = infer_fn(tile_lr_data)
            assert tile_hr_data.shape == (C, tile_hr, tile_hr), (
                f"infer_fn returned {tile_hr_data.shape}, expected ({C},{tile_hr},{tile_hr})"
            )
            y_hr, x_hr = y * scale, x * scale
            acc[:, y_hr:y_hr + tile_hr, x_hr:x_hr + tile_hr] += tile_hr_data.astype(np.float64)
            cnt[y_hr:y_hr + tile_hr, x_hr:x_hr + tile_hr] += 1.0

    cnt = np.where(cnt > 0, cnt, 1.0)
    out = acc / cnt[None, ...]
    return np.clip(out, 0.0, 65535.0).astype(np.uint16)
