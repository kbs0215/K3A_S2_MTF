"""prithvi_sr.py - Prithvi-EO-v2 기반 SR 모델 정의 + 로더 + 추론

구조 (사용자 학습 노트북 그대로):
  Encoder : Prithvi-EO-v2-300M (terratorch BACKBONE_REGISTRY) + LoRA(r=16, α=16, qkv/proj)
  Decoder : UnpatchingLayer → LocalExtractor fusion → HybridAttentionBlock×4 → PixelShuffle(4)
            + bicubic skip + zero-init residual head
  Forward : (x_6ch, lr_img) → HR [0,1] 스케일 4밴드

Normalization (사용자 spec):
  x_6ch  : (raw_DN - mean) / std, S2 L2A stats. 4채널 BGRN 만 z-score, 마지막 2채널 = 0 pad.
  lr_img : clip(raw_DN / 10000, 0, 1)
  출력   : [0,1] 도메인. 다운스트림에서 uint16 DN 으로 환원 시 *10000 후 clip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

import terratorch.models  # noqa: F401  registry side-effect
from terratorch.registry import BACKBONE_REGISTRY


# =====================================================================
# Normalization constants (S2 L2A reflectance × 10000 DN)
# =====================================================================

PRITHVI_MEAN = np.array([1087, 1342, 1433, 2734, 1958, 1363], dtype=np.float32)
PRITHVI_STD  = np.array([2248, 2179, 2178, 1850, 1242, 1049], dtype=np.float32)


# =====================================================================
# Sub-modules
# =====================================================================

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        res = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(res))


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.conv(self.avg_pool(x))


class HybridAttentionBlock(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.main(x)
        out = self.ca(out)
        out = self.sa(out)
        return x + out


class LocalExtractor(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class UnpatchingLayer(nn.Module):
    def __init__(self, in_dim: int = 1024, out_dim: int = 64):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(in_dim, out_dim * 2, kernel_size=4, stride=4)
        self.up2 = nn.ConvTranspose2d(out_dim * 2, out_dim, kernel_size=4, stride=4)
        self.smooth = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.up1(x))
        x = self.act(self.up2(x))
        return self.act(self.smooth(x))


# =====================================================================
# Decoder + Full model
# =====================================================================

class PrithviSRDecoder(nn.Module):
    def __init__(
        self,
        prithvi_embed_dim: int = 1024,
        feature_dim: int = 64,
        out_channels: int = 4,
    ):
        super().__init__()
        self.unpatch = UnpatchingLayer(in_dim=prithvi_embed_dim, out_dim=feature_dim)
        self.local_ext = LocalExtractor(in_channels=out_channels, out_channels=feature_dim)
        self.fusion = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.refinement = nn.Sequential(
            *[HybridAttentionBlock(channels=feature_dim) for _ in range(4)]
        )
        self.up_conv = nn.Conv2d(feature_dim, out_channels * 16, kernel_size=3, padding=1)
        nn.init.constant_(self.up_conv.weight, 0)
        nn.init.constant_(self.up_conv.bias, 0)
        self.pixel_shuffle = nn.PixelShuffle(4)

    def forward(self, prithvi_tokens: torch.Tensor, lr_img: torch.Tensor) -> torch.Tensor:
        B = prithvi_tokens.size(0)
        x = prithvi_tokens[:, 1:, :].transpose(1, 2).reshape(B, -1, 14, 14)
        global_feat = self.unpatch(x)
        local_feat = self.local_ext(lr_img)
        fused = self.fusion(torch.cat([global_feat, local_feat], dim=1))
        refined = self.refinement(fused)
        residual = self.pixel_shuffle(self.up_conv(refined))
        bicubic_up = F.interpolate(lr_img, scale_factor=4, mode="bicubic", align_corners=False)
        return residual + bicubic_up


class FullSRModel(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x_6ch: torch.Tensor, lr_img: torch.Tensor) -> torch.Tensor:
        encoder_outputs = self.encoder(x_6ch)
        # Layer -1 + Layer -4 평균 → global feature
        combined_feat = (encoder_outputs[-1] + encoder_outputs[-4]) * 0.5
        return self.decoder(combined_feat, lr_img)


# =====================================================================
# Encoder wrapping (terratorch + LoRA)
# =====================================================================

def get_prithvi_encoder_with_lora(
    checkpoint_path: Optional[Union[str, Path]] = None,
    model_name: str = "prithvi_eo_v2_300",
) -> nn.Module:
    """Prithvi 백본 + LoRA 어댑터.

    checkpoint_path 가 None 이면 백본 가중치는 로드하지 않음 (이후 FullSRModel
    state_dict 로딩 시 어차피 덮어쓰기 됨 — 추론 전용 경로).
    """
    encoder = BACKBONE_REGISTRY.build(model_name, pretrained=False, num_frames=1)
    if checkpoint_path is not None:
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
        encoder.load_state_dict(state_dict, strict=False)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["qkv", "proj"],
        lora_dropout=0.1,
        bias="none",
    )
    return get_peft_model(encoder, lora_config)


def build_full_sr_model(prithvi_checkpoint: Optional[Union[str, Path]] = None) -> FullSRModel:
    encoder = get_prithvi_encoder_with_lora(checkpoint_path=prithvi_checkpoint)
    decoder = PrithviSRDecoder(prithvi_embed_dim=1024, feature_dim=64, out_channels=4)
    return FullSRModel(encoder, decoder)


def load_prithvi_sr_model(
    sr_checkpoint: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    prithvi_checkpoint: Optional[Union[str, Path]] = None,
) -> FullSRModel:
    """best_sr_model.pth 로드 → eval 모드 FullSRModel 반환."""
    model = build_full_sr_model(prithvi_checkpoint=prithvi_checkpoint)
    ckpt = torch.load(str(sr_checkpoint), map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt.get("model", ckpt)))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load_prithvi_sr_model] missing keys: {len(missing)} (first 5: {missing[:5]})")
    if unexpected:
        print(f"[load_prithvi_sr_model] unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")
    model.to(device).eval()
    return model


# =====================================================================
# Inference helpers
# =====================================================================

def _normalize_for_encoder(lr_bgrn_dn: np.ndarray) -> np.ndarray:
    """BGRN 4채널 raw DN → 6채널 z-score normalized (마지막 2채널 = 0).
    
    인코더에 입력할 때 채널 순서를 RGBN으로 재배치합니다.
    z-score 정규화는 각 채널 고유의 스탯을 사용해 수행합니다.
      - Blue (idx 0): mean=1087, std=2248
      - Green (idx 1): mean=1342, std=2179
      - Red (idx 2): mean=1433, std=2178
      - NIR (idx 3): mean=2734, std=1850
    """
    assert lr_bgrn_dn.shape[0] == 4
    raw_f = lr_bgrn_dn.astype(np.float32)
    
    # 1. 각 채널 고유의 스탯으로 z-score 정규화
    blue_norm = (raw_f[0] - 1087.0) / 2248.0
    green_norm = (raw_f[1] - 1342.0) / 2179.0
    red_norm = (raw_f[2] - 1433.0) / 2178.0
    nir_norm = (raw_f[3] - 2734.0) / 1850.0
    
    # 2. 인코더가 기대하는 RGBN 순서로 6채널 어레이 생성 (마지막 2채널 = 0)
    x_6ch = np.zeros((6, raw_f.shape[1], raw_f.shape[2]), dtype=np.float32)
    x_6ch[0] = red_norm
    x_6ch[1] = green_norm
    x_6ch[2] = blue_norm
    x_6ch[3] = nir_norm
    return x_6ch


def _normalize_for_skip(lr_bgrn_dn: np.ndarray) -> np.ndarray:
    """BGRN 4채널 raw DN → [0,1] 클립."""
    return np.clip(lr_bgrn_dn.astype(np.float32) / 10000.0, 0.0, 1.0)


@torch.no_grad()
def prithvi_sr_infer_tile(
    model: FullSRModel,
    lr_bgrn_dn: np.ndarray,
    device: Union[str, torch.device] = "cpu",
) -> np.ndarray:
    """단일 224×224 타일 SR 추론.

    Input  : lr_bgrn_dn  uint16 raw DN, shape [4, 224, 224]
    Output : hr_bgrn_dn  uint16 raw DN, shape [4, 896, 896]
    """
    assert lr_bgrn_dn.shape == (4, 224, 224), f"expected (4,224,224), got {lr_bgrn_dn.shape}"
    x_6ch = _normalize_for_encoder(lr_bgrn_dn)
    lr_img = _normalize_for_skip(lr_bgrn_dn)

    x_6ch_t = torch.from_numpy(x_6ch).unsqueeze(0).to(device)
    lr_img_t = torch.from_numpy(lr_img).unsqueeze(0).to(device)

    # terratorch Prithvi encoder 는 [B, C, T, H, W] 입력을 기대 (num_frames=1 빌드 시)
    # 만약 학습 노트북이 [B, C, H, W] 4D 로 넘겼다면 아래 한 줄 주석 처리.
    x_6ch_t = x_6ch_t.unsqueeze(2)

    hr_t = model(x_6ch_t, lr_img_t)  # [1, 4, 896, 896] in [0,1]
    hr = hr_t.squeeze(0).cpu().numpy()
    hr_dn = np.clip(hr * 10000.0, 0.0, 65535.0).astype(np.uint16)
    return hr_dn
