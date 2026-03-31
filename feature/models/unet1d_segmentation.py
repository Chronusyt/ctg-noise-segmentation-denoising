"""
1D U-Net for noise segmentation (baseline)

输入: [B, 1, 240]
输出: [B, 1, 240] logits
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv1D(nn.Module):
    """两次 1D 卷积 + BN + ReLU。"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet1DSegmentation(nn.Module):
    """
    轻量 1D U-Net，用于 240 点时序分割。
    输入 [B, 1, 240]，输出 [B, 1, 240] logits。
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
    ):
        super().__init__()
        self.depth = depth
        # Encoder
        self.enc = nn.ModuleList()
        ch = in_channels
        for i in range(depth):
            out_ch = base_channels * (2**i)
            self.enc.append(DoubleConv1D(ch, out_ch))
            ch = out_ch

        # Bottleneck
        self.bottleneck = DoubleConv1D(ch, base_channels * (2**depth))

        # Decoder
        self.dec = nn.ModuleList()
        self.up = nn.ModuleList()
        ch = base_channels * (2**depth)
        for i in range(depth - 1, -1, -1):
            out_ch = base_channels * (2**i)
            self.up.append(nn.ConvTranspose1d(ch, out_ch, kernel_size=2, stride=2))
            self.dec.append(DoubleConv1D(out_ch * 2, out_ch))
            ch = out_ch

        self.out = nn.Conv1d(ch, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, 240]
        skips = []
        h = x
        for i, block in enumerate(self.enc):
            h = block(h)
            skips.append(h)
            if i < self.depth - 1:
                h = nn.functional.max_pool1d(h, 2)

        h = self.bottleneck(h)

        for i, (up, block) in enumerate(zip(self.up, self.dec)):
            h = up(h)
            # 对齐长度（可能因奇数长度产生 1 点差异）
            s = skips[-(i + 1)]
            if h.shape[-1] != s.shape[-1]:
                h = h[..., : s.shape[-1]]
            h = torch.cat([h, s], dim=1)
            h = block(h)

        return self.out(h)


def _test():
    model = UNet1DSegmentation(in_channels=1, out_channels=1, base_channels=32, depth=3)
    x = torch.randn(4, 1, 240)
    y = model(x)
    assert y.shape == (4, 1, 240), f"Expected (4,1,240), got {y.shape}"
    print("UNet1DSegmentation test OK")


if __name__ == "__main__":
    _test()
