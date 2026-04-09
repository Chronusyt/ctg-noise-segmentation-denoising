"""
1D U-Net style physiological multitask model.

Supported inputs:
    no-mask v1:
        noisy FHR [B, 1, L]
    gt-mask auxiliary v2:
        noisy FHR + 5-class GT masks [B, 6, L]

Outputs:
    reconstruction [B, 1, L]
    acceleration logits [B, 1, L]
    deceleration logits [B, 1, L]
    scalar heads [B] for baseline / STV / LTV / baseline variability
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

try:
    from .unet1d_denoiser import DoubleConv1D
except ImportError:  # Allow `python src/.../unet1d_physiological_multitask.py` smoke tests.
    from unet1d_denoiser import DoubleConv1D


class ScalarRegressionHead(nn.Module):
    """Global average pooling + small MLP scalar regression head."""

    def __init__(self, in_channels: int, hidden_channels: int = 128, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.pool(x)).squeeze(-1)


class UNet1DPhysiologicalMultitask(nn.Module):
    """
    Shared-encoder physiological multitask U-Net.

    The decoder feature map is shared by the reconstruction and temporal event
    heads; scalar physiological heads read from the bottleneck representation.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
        scalar_hidden_channels: int = 128,
        dropout: float = 0.0,
        residual_reconstruction: bool = True,
    ):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.residual_reconstruction = residual_reconstruction

        self.enc = nn.ModuleList()
        ch = in_channels
        for i in range(depth):
            out_ch = base_channels * (2**i)
            self.enc.append(DoubleConv1D(ch, out_ch))
            ch = out_ch

        bottleneck_ch = base_channels * (2**depth)
        self.bottleneck = DoubleConv1D(ch, bottleneck_ch)

        self.dec = nn.ModuleList()
        self.up = nn.ModuleList()
        ch = bottleneck_ch
        for i in range(depth - 1, -1, -1):
            out_ch = base_channels * (2**i)
            self.up.append(nn.ConvTranspose1d(ch, out_ch, kernel_size=2, stride=2))
            self.dec.append(DoubleConv1D(out_ch * 2, out_ch))
            ch = out_ch

        self.reconstruction_head = nn.Conv1d(ch, 1, kernel_size=1)
        self.acceleration_head = nn.Conv1d(ch, 1, kernel_size=1)
        self.deceleration_head = nn.Conv1d(ch, 1, kernel_size=1)

        self.baseline_head = ScalarRegressionHead(bottleneck_ch, scalar_hidden_channels, dropout)
        self.stv_head = ScalarRegressionHead(bottleneck_ch, scalar_hidden_channels, dropout)
        self.ltv_head = ScalarRegressionHead(bottleneck_ch, scalar_hidden_channels, dropout)
        self.baseline_variability_head = ScalarRegressionHead(bottleneck_ch, scalar_hidden_channels, dropout)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        skips = []
        h = x
        for i, block in enumerate(self.enc):
            h = block(h)
            skips.append(h)
            if i < self.depth - 1:
                h = nn.functional.max_pool1d(h, 2)

        bottleneck = self.bottleneck(h)
        h = bottleneck

        for i, (up, block) in enumerate(zip(self.up, self.dec)):
            h = up(h)
            s = skips[-(i + 1)]
            if h.shape[-1] != s.shape[-1]:
                h = h[..., : s.shape[-1]]
            h = torch.cat([h, s], dim=1)
            h = block(h)

        reconstruction = self.reconstruction_head(h)
        if self.residual_reconstruction:
            reconstruction = reconstruction + x[:, :1, :]

        return {
            "reconstruction": reconstruction,
            "acc_logits": self.acceleration_head(h),
            "dec_logits": self.deceleration_head(h),
            "baseline": self.baseline_head(bottleneck),
            "stv": self.stv_head(bottleneck),
            "ltv": self.ltv_head(bottleneck),
            "baseline_variability": self.baseline_variability_head(bottleneck),
        }


def _test() -> None:
    for in_channels in (1, 6):
        model = UNet1DPhysiologicalMultitask(in_channels=in_channels, base_channels=32, depth=3)
        x = torch.randn(4, in_channels, 240)
        y = model(x)
        assert y["reconstruction"].shape == (4, 1, 240)
        assert y["acc_logits"].shape == (4, 1, 240)
        assert y["dec_logits"].shape == (4, 1, 240)
        for key in ("baseline", "stv", "ltv", "baseline_variability"):
            assert y[key].shape == (4,), f"{key}: {y[key].shape}"
    print("UNet1DPhysiologicalMultitask test OK")


if __name__ == "__main__":
    _test()
