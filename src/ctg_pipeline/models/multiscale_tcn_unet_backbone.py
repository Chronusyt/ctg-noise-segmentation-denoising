"""Encoder/decoder backbone family for physiological multitask editing."""
from __future__ import annotations

from typing import Callable, Dict, Sequence

import torch
import torch.nn as nn

try:
    from .modern_tcn_backbone import ModernTCNBlock1D
    from .unet1d_denoiser import DoubleConv1D
except ImportError:  # Allow standalone smoke tests.
    from modern_tcn_backbone import ModernTCNBlock1D
    from unet1d_denoiser import DoubleConv1D


class SingleScaleConvStem1D(nn.Module):
    """Single-kernel convolution stem used by non-multiscale variants."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))

    def reference_parameter(self) -> torch.nn.Parameter:
        return self.conv.weight


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

    def reference_parameter(self) -> torch.nn.Parameter:
        return self.branches[0][0].weight


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


class ConvNeXtBlock1D(nn.Module):
    """A lightweight ConvNeXt-style residual block for 1D signals."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        expansion: int = 4,
        dropout: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        super().__init__()
        hidden_channels = channels * expansion
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.norm = nn.LayerNorm(channels)
        self.pointwise_in = nn.Linear(channels, hidden_channels)
        self.activation = nn.GELU()
        self.pointwise_out = nn.Linear(hidden_channels, channels)
        self.dropout = nn.Dropout(dropout)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(channels))
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.depthwise(x).transpose(1, 2)
        h = self.norm(h)
        h = self.pointwise_in(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.pointwise_out(h)
        if self.gamma is not None:
            h = h * self.gamma
        h = self.dropout(h).transpose(1, 2)
        return residual + h


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


def _make_tcn_stage(channels: int, stage_idx: int, dropout: float) -> nn.Sequential:
    base_dilation = 2**stage_idx
    return nn.Sequential(
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


def _make_modern_tcn_stage(
    channels: int,
    stage_idx: int,
    blocks_per_stage: int,
    kernel_size: int,
    expansion: int,
    dropout: float,
) -> nn.Sequential:
    base_dilation = 2**stage_idx
    blocks = []
    for block_idx in range(blocks_per_stage):
        blocks.append(
            ModernTCNBlock1D(
                channels,
                kernel_size=kernel_size,
                dilation=base_dilation * (2**block_idx),
                expansion=expansion,
                dropout=dropout,
            )
        )
    return nn.Sequential(*blocks)


def _make_convnext_stage(
    channels: int,
    _stage_idx: int,
    blocks_per_stage: int,
    kernel_size: int,
    expansion: int,
    dropout: float,
) -> nn.Sequential:
    blocks = []
    for _ in range(blocks_per_stage):
        blocks.append(
            ConvNeXtBlock1D(
                channels,
                kernel_size=kernel_size,
                expansion=expansion,
                dropout=dropout,
            )
        )
    return nn.Sequential(*blocks)


class _EncoderDecoderBackbone1D(nn.Module):
    """Generic stem + hierarchical encoder + optional bottleneck + U-Net decoder."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        depth: int,
        dropout: float,
        bottleneck_type: str,
        stem_factory: Callable[[int, int], nn.Module],
        stage_factory: Callable[[int, int], nn.Module],
    ) -> None:
        super().__init__()
        if bottleneck_type not in {"none", "mhsa"}:
            raise ValueError(f"Unsupported bottleneck_type: {bottleneck_type}")

        self.depth = depth
        self.bottleneck_type = bottleneck_type
        stage_channels = [base_channels * (2**i) for i in range(depth)]
        self.time_channels = stage_channels[0]
        self.global_channels = stage_channels[-1]

        self.stem = stem_factory(in_channels, stage_channels[0])
        self.encoder_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for stage_idx, channels in enumerate(stage_channels):
            self.encoder_stages.append(stage_factory(channels, stage_idx))
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

    def reference_parameter(self) -> torch.nn.Parameter:
        if hasattr(self.stem, "reference_parameter"):
            return self.stem.reference_parameter()
        return self.stem[0].weight

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


class UNetBackbone1D(nn.Module):
    """Single-scale stem + ordinary U-Net encoder/decoder."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
        use_multiscale_stem: bool = False,
    ) -> None:
        super().__init__()
        del dropout  # Kept for a uniform constructor signature across backbones.
        self.depth = depth
        self.time_channels = base_channels
        self.global_channels = base_channels * (2**depth)
        self.stem = (
            MultiScaleConvStem1D(in_channels, base_channels)
            if use_multiscale_stem
            else SingleScaleConvStem1D(in_channels, base_channels)
        )

        self.enc = nn.ModuleList()
        ch = base_channels
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

    def reference_parameter(self) -> torch.nn.Parameter:
        if hasattr(self.stem, "reference_parameter"):
            return self.stem.reference_parameter()
        return self.stem[0].weight

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        skips = []
        h = self.stem(x)
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

        return {
            "time_features": h,
            "global_features": bottleneck,
        }


class MultiscaleUNetBackbone1D(UNetBackbone1D):
    """Multi-scale stem + ordinary U-Net encoder/decoder."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            depth=depth,
            dropout=dropout,
            use_multiscale_stem=True,
        )


class TCNUNetBackbone1D(_EncoderDecoderBackbone1D):
    """Single-scale stem + dilated residual TCN encoder + U-Net decoder."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
        bottleneck_type: str = "none",
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            depth=depth,
            dropout=dropout,
            bottleneck_type=bottleneck_type,
            stem_factory=SingleScaleConvStem1D,
            stage_factory=lambda channels, stage_idx: _make_tcn_stage(channels, stage_idx, dropout),
        )


class MultiscaleTCNUNetBackbone1D(_EncoderDecoderBackbone1D):
    """Multi-scale stem + dilated residual TCN encoder + U-Net decoder."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
        bottleneck_type: str = "none",
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            depth=depth,
            dropout=dropout,
            bottleneck_type=bottleneck_type,
            stem_factory=MultiScaleConvStem1D,
            stage_factory=lambda channels, stage_idx: _make_tcn_stage(channels, stage_idx, dropout),
        )


class ModernTCNUNetBackbone1D(_EncoderDecoderBackbone1D):
    """Single-scale stem + ModernTCN encoder + U-Net decoder."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
        bottleneck_type: str = "none",
        blocks_per_stage: int = 2,
        kernel_size: int = 5,
        expansion: int = 2,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            depth=depth,
            dropout=dropout,
            bottleneck_type=bottleneck_type,
            stem_factory=SingleScaleConvStem1D,
            stage_factory=lambda channels, stage_idx: _make_modern_tcn_stage(
                channels,
                stage_idx,
                blocks_per_stage=blocks_per_stage,
                kernel_size=kernel_size,
                expansion=expansion,
                dropout=dropout,
            ),
        )


class ConvNeXtUNetBackbone1D(_EncoderDecoderBackbone1D):
    """Single-scale stem + ConvNeXt1D encoder + U-Net decoder."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
        bottleneck_type: str = "none",
        blocks_per_stage: int = 2,
        kernel_size: int = 7,
        expansion: int = 4,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            depth=depth,
            dropout=dropout,
            bottleneck_type=bottleneck_type,
            stem_factory=SingleScaleConvStem1D,
            stage_factory=lambda channels, stage_idx: _make_convnext_stage(
                channels,
                stage_idx,
                blocks_per_stage=blocks_per_stage,
                kernel_size=kernel_size,
                expansion=expansion,
                dropout=dropout,
            ),
        )


class MultiscaleModernTCNUNetBackbone1D(_EncoderDecoderBackbone1D):
    """Multi-scale stem + ModernTCN encoder + U-Net decoder."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
        bottleneck_type: str = "none",
        blocks_per_stage: int = 2,
        kernel_size: int = 5,
        expansion: int = 2,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            depth=depth,
            dropout=dropout,
            bottleneck_type=bottleneck_type,
            stem_factory=MultiScaleConvStem1D,
            stage_factory=lambda channels, stage_idx: _make_modern_tcn_stage(
                channels,
                stage_idx,
                blocks_per_stage=blocks_per_stage,
                kernel_size=kernel_size,
                expansion=expansion,
                dropout=dropout,
            ),
        )


def _test() -> None:
    configs = [
        (UNetBackbone1D, {"depth": 4}),
        (MultiscaleUNetBackbone1D, {"depth": 4}),
        (TCNUNetBackbone1D, {"depth": 4}),
        (ModernTCNUNetBackbone1D, {"depth": 4}),
        (ConvNeXtUNetBackbone1D, {"depth": 4}),
        (MultiscaleModernTCNUNetBackbone1D, {"depth": 4}),
    ]
    for cls, kwargs in configs:
        model = cls(
            in_channels=6,
            base_channels=16,
            dropout=0.0,
            **kwargs,
        )
        y = model(torch.randn(2, 6, 240))
        assert y["time_features"].shape == (2, 16, 240)
        assert y["global_features"].shape == (2, model.global_channels, 30)

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
