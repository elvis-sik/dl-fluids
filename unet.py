"""UNet (plus a couple of other network architectures).

UNet implementation taken from here:
https://github.com/milesial/Pytorch-UNet
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearCD(torch.nn.Module):
    """Linear model to predict a single scalar, C_D."""
    def __init__(self, input_shape=(3, 64, 64), input_channels=None):
        super().__init__()
        inp_size = np.prod(input_shape)
        self.linear = nn.Linear(inp_size, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x.view(x.size(0), 1, 1, 1)


class SimpleNet(torch.nn.Module):
    """Two layered fully-convolutional network.

    Transforms tensors from shape (input_channels, r, r) into ones with shape
    (output_channels, r, r).
    """
    def __init__(
            self,
            input_channels=3,
            out_channels=3,
    ):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(
            in_channels=input_channels,
            out_channels=1,
            kernel_size=(3, 3),
        )

        self.conv2 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding=2,
        )

    def forward(self, x):
        out_1 = self.conv1(x)
        out_2 = F.relu(out_1)
        return self.conv2(out_2)


class UNetCD(nn.Module):
    """UNet with a linear layer on top of it to predict C_D."""
    def __init__(self,
                 input_channels=None,
                 out_channels=None,
                 unet=None,
                 resolution=(64, 64),
                 bilinear=True,
                 dropout=0):
        super().__init__()

        if not unet:
            unet = UNet(input_channels, out_channels, resolution, bilinear,
                        dropout),
        else:
            out_channels = unet.out_channels
            resolution = unet.resolution

        self.unet = unet
        self.linear = nn.Linear(out_channels * np.prod(resolution), 1)

    def forward(self, x):
        x = self.unet(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x.view(x.size(0), 1, 1, 1)


class UNet(nn.Module):
    """UNet neural network architecture.

    From the UNet implementation in the link below
    https://github.com/milesial/Pytorch-UNet
    """
    def __init__(
            self,
            input_channels,
            out_channels,
            resolution=(64, 64),
            bilinear=True,
            dropout=0,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.resolution = resolution
        self.out_channels = out_channels
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(input_channels, 64, dropout=dropout)
        self.down1 = Down(64, 128, dropout=dropout)
        self.down2 = Down(128, 256, dropout=dropout)
        self.down3 = Down(256, 512, dropout=dropout)
        self.down4 = Down(512, 1024 // factor, dropout=dropout)

        self.up1 = Up(1024, 512 // factor, bilinear, dropout=dropout)
        self.up2 = Up(512, 256 // factor, bilinear, dropout=dropout)
        self.up3 = Up(256, 128 // factor, bilinear, dropout=dropout)
        self.up4 = Up(128, 64, bilinear, dropout=dropout)

        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.outc(x)

        return x
        # return torch.tanh(x)


class DoubleConv(nn.Module):
    """DoubleConv operation for UNet.

    From the UNet implementation in the link below
    https://github.com/milesial/Pytorch-UNet
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=None,
                 dropout=0.7):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Down operation for UNet.

    From the UNet implementation in the link below
    https://github.com/milesial/Pytorch-UNet
    """
    def __init__(self, in_channels, out_channels, dropout=0.7):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout=dropout))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Up operation for UNet.

    From the UNet implementation in the link below
    https://github.com/milesial/Pytorch-UNet
    """
    def __init__(self, in_channels, out_channels, bilinear=True, dropout=0.7):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
            self.conv = DoubleConv(in_channels,
                                   out_channels,
                                   in_channels // 2,
                                   dropout=dropout)
        else:
            self.up = nn.ConvTranspose2d(in_channels,
                                         in_channels // 2,
                                         kernel_size=2,
                                         stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """OutConv operation for UNet.

    From the UNet implementation in the link below
    https://github.com/milesial/Pytorch-UNet
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
