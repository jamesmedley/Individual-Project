""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class WaveletDown(nn.Module):
    """Downscaling using Discrete Wavelet Transform (DWT) instead of max-pooling"""
    def __init__(self, in_channels, out_channels, wave='haar'):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.dwt = DWTForward(J=1, mode='zero', wave=wave)  # Single-level DWT

    def forward(self, x):
        Yl, Yh = self.dwt(x)
        return self.conv(Yl), Yh


class WaveletUp(nn.Module):
    """Upscaling using inverse Discrete Wavelet Transform (IDWT) with skip connection"""
    def __init__(self, in_channels, out_channels, wave='haar'):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.idwt = DWTInverse(mode='zero', wave=wave)
        self.halve_channels = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)

    def forward(self, x, skip_connection, Yh):
        x = self.halve_channels(x)  # Halve channels before IDWT
        x = self.idwt((x, Yh))
        x = torch.cat([skip_connection, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
