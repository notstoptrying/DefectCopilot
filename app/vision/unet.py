"""轻量 UNet：200×200 输入的二值缺陷分割网络。

参数量约 1.9M，6G 显存下 batch 8 占用不到 2G，方便在笔记本上复现。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _match(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """把跳跃连接对齐到上采样结果的空间尺寸。

    200 这类不能被 16 整除的输入，四次池化后尺寸向下取整（200→12），
    上采样回来是 24 而非 25；奇数尺寸下还会反向出现上采样偏大。
    统一用插值对齐最稳，且不改变最终输出分辨率。
    """
    if t.shape[-2:] != ref.shape[-2:]:
        t = F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    return t


class DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 32):
        super().__init__()
        self.d1 = DoubleConv(in_ch, base)
        self.d2 = DoubleConv(base, base * 2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.d4 = DoubleConv(base * 4, base * 8)
        self.bottleneck = DoubleConv(base * 8, base * 16)
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.c4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.c3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.c2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.c1 = DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))
        d4 = self.d4(self.pool(d3))
        b = self.bottleneck(self.pool(d4))
        u4 = self.c4(torch.cat([_match(self.up4(b), d4), d4], dim=1))
        u3 = self.c3(torch.cat([_match(self.up3(u4), d3), d3], dim=1))
        u2 = self.c2(torch.cat([_match(self.up2(u3), d2), d2], dim=1))
        u1 = self.c1(torch.cat([_match(self.up1(u2), d1), d1], dim=1))
        out = self.head(u1)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
