from __future__ import annotations

import torch
from torch import nn


class BallTrackerNet(nn.Module):
    """TrackNet-like architecture used by TennisCourtDetector pretrained weights.

    The layer naming (conv1..conv18 and deconv blocks) is intentionally explicit to
    match checkpoint keys from the original project.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 15) -> None:
        super().__init__()

        # encoder
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)

        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(512)

        # decoder
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)

        self.conv11 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(256)
        self.conv12 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(256)
        self.conv13 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn13 = nn.BatchNorm2d(256)

        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)

        self.conv14 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn14 = nn.BatchNorm2d(128)
        self.conv15 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn15 = nn.BatchNorm2d(128)

        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.conv16 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn16 = nn.BatchNorm2d(64)
        self.conv17 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn17 = nn.BatchNorm2d(64)

        self.conv18 = nn.Conv2d(64, out_channels, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.pool(x)

        x = self.relu(self.bn8(self.conv8(x)))
        x = self.relu(self.bn9(self.conv9(x)))
        x = self.relu(self.bn10(self.conv10(x)))

        x = self.deconv1(x)

        x = self.relu(self.bn11(self.conv11(x)))
        x = self.relu(self.bn12(self.conv12(x)))
        x = self.relu(self.bn13(self.conv13(x)))

        x = self.deconv2(x)

        x = self.relu(self.bn14(self.conv14(x)))
        x = self.relu(self.bn15(self.conv15(x)))

        x = self.deconv3(x)

        x = self.relu(self.bn16(self.conv16(x)))
        x = self.relu(self.bn17(self.conv17(x)))
        x = self.conv18(x)
        return x
