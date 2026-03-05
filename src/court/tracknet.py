from __future__ import annotations

import torch
from torch import nn


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, layers: int = 2) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        c_in = in_channels
        for _ in range(layers):
            modules.append(nn.Conv2d(c_in, out_channels, kernel_size=3, padding=1, bias=False))
            modules.append(nn.BatchNorm2d(out_channels))
            modules.append(nn.ReLU(inplace=True))
            c_in = out_channels
        self.block = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BallTrackerNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 15) -> None:
        super().__init__()

        self.enc1 = _ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = _ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = _ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = _ConvBlock(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = _ConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = _ConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = _ConvBlock(128, 64)

        self.head = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.head(d1)
