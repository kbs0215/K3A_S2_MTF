"""metrics.py - 이진 segmentation metric (water 클래스 기준).

모든 함수:
  - pred, label  : 2D 이진 배열 (0/1, uint8 또는 bool)
  - valid_mask   : optional. 평가 제외 픽셀 (라벨 -1 등) 마스킹용
  - 반환값       : float. 분모 0 인 경우 nan (예: 라벨에 water/boundary/small CC 없음)

지원 metric:
  - iou_score        : 표준 IoU (Jaccard)
  - f1_score         : Dice / F1
  - boundary_iou     : 라벨 경계 ±dilation 픽셀 zone 으로 한정한 IoU
  - small_cc_iou     : 라벨 CC 중 면적 < max_area 인 객체만 한정한 IoU
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, label as cc_label


def _binarize(arr: np.ndarray) -> np.ndarray:
    return (arr > 0).astype(bool)


def iou_score(
    pred: np.ndarray,
    label: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> float:
    p = _binarize(pred)
    g = _binarize(label)
    if valid_mask is not None:
        p = p & valid_mask
        g = g & valid_mask
        inter = (p & g).sum()
        union = (p | g).sum()
    else:
        inter = (p & g).sum()
        union = (p | g).sum()
    return float(inter) / float(union) if union > 0 else float("nan")


def f1_score(
    pred: np.ndarray,
    label: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> float:
    p = _binarize(pred)
    g = _binarize(label)
    if valid_mask is not None:
        p = p & valid_mask
        g = g & valid_mask
    tp = (p & g).sum()
    fp = (p & ~g).sum()
    fn = (~p & g).sum()
    denom = 2 * tp + fp + fn
    return float(2 * tp) / float(denom) if denom > 0 else float("nan")


def boundary_iou(
    pred: np.ndarray,
    label: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    dilation: int = 1,
) -> float:
    """라벨 경계 ±dilation 픽셀 zone 만 평가."""
    g = _binarize(label)
    if not g.any():
        return float("nan")
    g_dil = binary_dilation(g, iterations=dilation)
    g_ero = binary_erosion(g, iterations=dilation)
    zone = g_dil & ~g_ero
    if valid_mask is not None:
        zone = zone & valid_mask
    if not zone.any():
        return float("nan")
    p = _binarize(pred) & zone
    g_z = g & zone
    inter = (p & g_z).sum()
    union = (p | g_z).sum()
    return float(inter) / float(union) if union > 0 else float("nan")


def small_cc_iou(
    pred: np.ndarray,
    label: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    max_area: int = 100,
) -> float:
    """라벨에서 면적 < max_area 인 CC 만 평가 영역으로 한정."""
    g = _binarize(label)
    if not g.any():
        return float("nan")
    cc, n = cc_label(g)
    if n == 0:
        return float("nan")
    sizes = np.bincount(cc.ravel())
    sizes[0] = 0  # background
    small_ids = np.where((sizes > 0) & (sizes < max_area))[0]
    if small_ids.size == 0:
        return float("nan")
    small_mask = np.isin(cc, small_ids)
    if valid_mask is not None:
        small_mask = small_mask & valid_mask
    if not small_mask.any():
        return float("nan")
    p = _binarize(pred) & small_mask
    g_s = g & small_mask
    inter = (p & g_s).sum()
    union = (p | g_s).sum()
    return float(inter) / float(union) if union > 0 else float("nan")


def all_metrics(
    pred: np.ndarray,
    label: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    boundary_dilation: int = 1,
    small_cc_max_area: int = 100,
) -> dict:
    return {
        "iou": iou_score(pred, label, valid_mask),
        "f1": f1_score(pred, label, valid_mask),
        "boundary_iou": boundary_iou(pred, label, valid_mask, dilation=boundary_dilation),
        "small_cc_iou": small_cc_iou(pred, label, valid_mask, max_area=small_cc_max_area),
    }
