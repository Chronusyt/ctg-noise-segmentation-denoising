"""
1D U-Net for multi-label noise segmentation（五类逐点分割）

与 binary segmentation 独立，不修改原 unet1d_segmentation.py。

输入: [B, 1, 240]
输出: [B, 5, 240] logits（5 通道独立，用 sigmoid 非 softmax）
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


class UNet1DMultilabelSegmentation(nn.Module):
    """
    1D U-Net，五类噪声分割。
    输入 [B, 1, 240]，输出 [B, 5, 240] logits。
    五通道：halving, doubling, mhr, missing, spike。
    使用 sigmoid，非 softmax（multi-label 可共存）。
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 5,
        base_channels: int = 32,
        depth: int = 3,
    ):
        super().__init__()
        self.depth = depth
        self.out_channels = out_channels

        self.enc = nn.ModuleList()
        ch = in_channels
        for i in range(depth):
            out_ch = base_channels * (2**i)
            self.enc.append(DoubleConv1D(ch, out_ch))
            ch = out_ch

        self.bottleneck = DoubleConv1D(ch, base_channels * (2**depth))

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
            s = skips[-(i + 1)]
            if h.shape[-1] != s.shape[-1]:
                h = h[..., : s.shape[-1]]
            h = torch.cat([h, s], dim=1)
            h = block(h)

        return self.out(h)


def _test():
    model = UNet1DMultilabelSegmentation(in_channels=1, out_channels=5, base_channels=32, depth=3)
    x = torch.randn(4, 1, 240)
    y = model(x)
    assert y.shape == (4, 5, 240), f"Expected (4,5,240), got {y.shape}"
    print("UNet1DMultilabelSegmentation test OK")


if __name__ == "__main__":
    _test()
