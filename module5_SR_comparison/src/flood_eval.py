"""flood_eval.py - segmentation_models_pytorch U-Net 사전학습 로더 + 추론

모델: smp.Unet(resnet34, ImageNet pretrained, in_channels=4, classes=1)
  - 학습 없이 ImageNet pretrained weight 만 사용
  - 4번째 채널 (NIR) 의 첫 conv weight 는 Red 채널 weight 복사 (long-wavelength similarity)

입력 정규화:
  - 우리 데이터는 BGRN order, raw DN (reflectance × 10000, uint16)
  - DN / 10000 → [0,1] 클립
  - ImageNet RGB order 매칭 위해 BGRN → RGBN reorder
  - ImageNet mean/std 적용, NIR 채널은 Red stats 재사용

출력:
  - logit [1, 1, H, W] → sigmoid → probability map
  - 4×4 mean pool 옵션 (2.5m → 10m), threshold 0.5 → binary mask
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F


IMAGENET_MEAN_RGBN = np.array([0.485, 0.456, 0.406, 0.485], dtype=np.float32)
IMAGENET_STD_RGBN  = np.array([0.229, 0.224, 0.225, 0.229], dtype=np.float32)


def build_unet(
    in_channels: int = 4,
    encoder_name: str = "resnet34",
    nir_init_from_red: bool = True,
) -> smp.Unet:
    """ImageNet pretrained U-Net + NIR=Red weight 안전장치."""
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=1,
    )
    if in_channels == 4 and nir_init_from_red:
        with torch.no_grad():
            # smp 가 in_channels=4 로 conv1 을 확장하며 자동 weight 평균/확장 수행함.
            # NIR 채널만 명시적으로 Red 채널 weight 복사 → 분광적으로 유사한 초기화.
            # 채널 순서: RGBN (우리가 입력 reorder 후) → conv1.weight[:, 0]=R, [:, 3]=NIR
            model.encoder.conv1.weight[:, 3] = model.encoder.conv1.weight[:, 0]
    return model


def load_unet(
    device: Union[str, torch.device] = "cpu",
    in_channels: int = 4,
    encoder_name: str = "resnet34",
    nir_init_from_red: bool = True,
) -> smp.Unet:
    model = build_unet(in_channels=in_channels, encoder_name=encoder_name,
                       nir_init_from_red=nir_init_from_red)
    model.to(device).eval()
    return model


def normalize_for_unet(bgrn_dn: np.ndarray) -> np.ndarray:
    """BGRN raw DN uint16 → RGBN normalized float32 (ImageNet stats)."""
    assert bgrn_dn.shape[0] == 4, f"expected 4 channels (BGRN), got {bgrn_dn.shape}"
    bgrn = np.clip(bgrn_dn.astype(np.float32) / 10000.0, 0.0, 1.0)
    rgbn = bgrn[[2, 1, 0, 3]]  # BGRN → RGBN
    return (rgbn - IMAGENET_MEAN_RGBN[:, None, None]) / IMAGENET_STD_RGBN[:, None, None]


def _ensure_divisible(t: torch.Tensor, divisor: int = 32) -> tuple:
    """ResNet 5 단계 다운샘플링 → 입력 H, W 가 32 의 배수여야 함. 필요시 reflect pad."""
    h, w = t.shape[-2:]
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    if pad_h == 0 and pad_w == 0:
        return t, (0, 0, 0, 0)
    t_pad = F.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
    return t_pad, (0, pad_w, 0, pad_h)


@torch.no_grad()
def unet_predict_logit(
    model: smp.Unet,
    bgrn_dn: np.ndarray,
    device: Union[str, torch.device] = "cpu",
) -> np.ndarray:
    """4채널 BGRN raw DN → sigmoid 적용 전 logit map [H, W] float32."""
    x = normalize_for_unet(bgrn_dn)
    t = torch.from_numpy(x).unsqueeze(0).to(device)
    t_pad, (l, r, top, bot) = _ensure_divisible(t, divisor=32)
    logit = model(t_pad)
    if r or bot:
        logit = logit[..., :logit.shape[-2] - bot, :logit.shape[-1] - r]
    return logit.squeeze(0).squeeze(0).cpu().numpy()


def mean_pool_4x4(arr: np.ndarray) -> np.ndarray:
    """4×4 mean pool — 마지막 2개 축 기준. 사이즈가 4의 배수여야 함."""
    if arr.ndim == 2:
        h, w = arr.shape
        assert h % 4 == 0 and w % 4 == 0, f"size not div by 4: {arr.shape}"
        return arr.reshape(h // 4, 4, w // 4, 4).mean(axis=(1, 3))
    if arr.ndim == 3:
        c, h, w = arr.shape
        assert h % 4 == 0 and w % 4 == 0, f"size not div by 4: {arr.shape}"
        return arr.reshape(c, h // 4, 4, w // 4, 4).mean(axis=(2, 4))
    raise ValueError(f"expected 2D or 3D, got {arr.shape}")


def unet_water_mask_at_10m(
    model: smp.Unet,
    bgrn_dn: np.ndarray,
    input_resolution_m: float,
    device: Union[str, torch.device] = "cpu",
    threshold: float = 0.5,
) -> np.ndarray:
    """U-Net 추론 → (필요시) 10m 로 mean pool → threshold → binary mask uint8.

    input_resolution_m = 10.0 면 그대로 threshold.
    input_resolution_m = 2.5  면 4×4 mean pool 후 threshold.
    """
    logit = unet_predict_logit(model, bgrn_dn, device=device)
    prob = 1.0 / (1.0 + np.exp(-logit))  # sigmoid
    if abs(input_resolution_m - 10.0) < 1e-3:
        pass
    elif abs(input_resolution_m - 2.5) < 1e-3:
        prob = mean_pool_4x4(prob)
    else:
        raise ValueError(f"unsupported input_resolution_m={input_resolution_m}")
    return (prob > threshold).astype(np.uint8)
