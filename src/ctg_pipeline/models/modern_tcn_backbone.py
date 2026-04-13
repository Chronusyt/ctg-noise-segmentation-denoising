"""ModernTCN-style backbone blocks for physiological multitask modeling."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class ModernTCNBlock1D(nn.Module):
    """A lightweight ModernTCN-style residual block for 1D signals."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        dilation: int = 1,
        expansion: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        hidden_channels = channels * expansion
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
        )
        self.depthwise_norm = nn.BatchNorm1d(channels)
        self.pointwise_in = nn.Conv1d(channels, hidden_channels, kernel_size=1)
        self.pointwise_act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.pointwise_out = nn.Conv1d(hidden_channels, channels, kernel_size=1)
        self.output_norm = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.depthwise(x)
        h = self.depthwise_norm(h)
        h = self.pointwise_in(h)
        h = self.pointwise_act(h)
        h = self.dropout(h)
        h = self.pointwise_out(h)
        h = self.output_norm(h)
        h = self.dropout(h)
        return residual + h


class ModernTCNBackbone1D(nn.Module):
    """
    A compact ModernTCN-style shared backbone.

    Returns a dict with:
    - `time_features`: shared temporal features for reconstruction/event heads
    - `global_features`: shared features for scalar pooling heads
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        depth: int = 3,
        blocks_per_stage: int = 2,
        kernel_size: int = 5,
        expansion: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        stage_channels = [base_channels * (2**i) for i in range(depth)]
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stage_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm1d(stage_channels[0]),
            nn.GELU(),
        )

        self.transitions = nn.ModuleList()
        self.stages = nn.ModuleList()
        in_ch = stage_channels[0]
        for stage_idx, out_ch in enumerate(stage_channels):
            if stage_idx == 0:
                self.transitions.append(nn.Identity())
            else:
                self.transitions.append(
                    nn.Sequential(
                        nn.Conv1d(in_ch, out_ch, kernel_size=1),
                        nn.BatchNorm1d(out_ch),
                        nn.GELU(),
                    )
                )
            blocks = []
            for block_idx in range(blocks_per_stage):
                blocks.append(
                    ModernTCNBlock1D(
                        out_ch,
                        kernel_size=kernel_size,
                        dilation=2**block_idx,
                        expansion=expansion,
                        dropout=dropout,
                    )
                )
            self.stages.append(nn.Sequential(*blocks))
            in_ch = out_ch

        self.out_channels = stage_channels[-1]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.stem(x)
        for transition, stage in zip(self.transitions, self.stages):
            h = transition(h)
            h = stage(h)
        return {
            "time_features": h,
            "global_features": h,
        }
