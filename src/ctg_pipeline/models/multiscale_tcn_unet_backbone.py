"""Multi-scale TCN U-Net style backbone for typed constrained residual editing."""
from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn

try:
    from .unet1d_denoiser import DoubleConv1D
except ImportError:  # Allow standalone smoke tests.
    from unet1d_denoiser import DoubleConv1D


class MultiScaleConvStem1D(nn.Module):
    """Parallel multi-kernel convolution stem followed by channel fusion."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Sequence[int] = (3, 7, 15),
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel, padding=kernel // 2),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                )
                for kernel in kernel_sizes
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(out_channels * len(kernel_sizes), out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [branch(x) for branch in self.branches]
        return self.fuse(torch.cat(features, dim=1))


class DilatedResidualTCNBlock1D(nn.Module):
    """Residual 1D convolution block with dilation-aware context."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class MHSABottleneck1D(nn.Module):
    """A small pre-norm self-attention bottleneck for low-resolution features."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        expansion: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_channels = channels * expansion
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        attn_input = self.norm1(h)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        h = h + attn_output
        h = h + self.ffn(self.norm2(h))
        return h.transpose(1, 2)


class MultiscaleTCNUNetBackbone1D(nn.Module):
    """
    Multi-scale stem + dilated TCN encoder + optional MHSA bottleneck + U-Net decoder.

    Returns:
    - `time_features`: decoder output for reconstruction/event heads
    - `global_features`: bottleneck output for scalar heads
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
        bottleneck_type: str = "none",
    ) -> None:
        super().__init__()
        if bottleneck_type not in {"none", "mhsa"}:
            raise ValueError(f"Unsupported bottleneck_type: {bottleneck_type}")

        self.depth = depth
        self.bottleneck_type = bottleneck_type
        stage_channels = [base_channels * (2**i) for i in range(depth)]
        self.time_channels = stage_channels[0]
        self.global_channels = stage_channels[-1]

        self.stem = MultiScaleConvStem1D(in_channels, stage_channels[0])
        self.encoder_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for stage_idx, channels in enumerate(stage_channels):
            base_dilation = 2**stage_idx
            self.encoder_stages.append(
                nn.Sequential(
                    DilatedResidualTCNBlock1D(
                        channels,
                        kernel_size=3,
                        dilation=base_dilation,
                        dropout=dropout,
                    ),
                    DilatedResidualTCNBlock1D(
                        channels,
                        kernel_size=3,
                        dilation=base_dilation * 2,
                        dropout=dropout,
                    ),
                )
            )
            if stage_idx < depth - 1:
                self.downsamples.append(
                    nn.Sequential(
                        nn.Conv1d(channels, stage_channels[stage_idx + 1], kernel_size=4, stride=2, padding=1),
                        nn.BatchNorm1d(stage_channels[stage_idx + 1]),
                        nn.GELU(),
                    )
                )

        self.bottleneck = (
            nn.Identity()
            if bottleneck_type == "none"
            else MHSABottleneck1D(stage_channels[-1], num_heads=4, expansion=2, dropout=dropout)
        )

        self.up = nn.ModuleList()
        self.decoder = nn.ModuleList()
        current_channels = stage_channels[-1]
        for stage_idx in range(depth - 2, -1, -1):
            out_channels = stage_channels[stage_idx]
            self.up.append(nn.ConvTranspose1d(current_channels, out_channels, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv1D(out_channels * 2, out_channels))
            current_channels = out_channels

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        skips = []
        h = self.stem(x)
        for stage_idx, encoder in enumerate(self.encoder_stages):
            h = encoder(h)
            if stage_idx < self.depth - 1:
                skips.append(h)
                h = self.downsamples[stage_idx](h)

        bottleneck = self.bottleneck(h)
        h = bottleneck
        for up, block, skip in zip(self.up, self.decoder, reversed(skips)):
            h = up(h)
            if h.shape[-1] != skip.shape[-1]:
                h = h[..., : skip.shape[-1]]
            h = torch.cat([h, skip], dim=1)
            h = block(h)

        return {
            "time_features": h,
            "global_features": bottleneck,
        }


def _test() -> None:
    for bottleneck_type in ("none", "mhsa"):
        model = MultiscaleTCNUNetBackbone1D(
            in_channels=6,
            base_channels=16,
            depth=4,
            dropout=0.0,
            bottleneck_type=bottleneck_type,
        )
        x = torch.randn(2, 6, 240)
        y = model(x)
        assert y["time_features"].shape == (2, 16, 240)
        assert y["global_features"].shape == (2, 128, 30)
    print("MultiscaleTCNUNetBackbone1D test OK")


if __name__ == "__main__":
    _test()
