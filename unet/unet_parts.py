import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DTCWTForward, DTCWTInverse


class DoubleConv(nn.Module):
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
    def __init__(self, in_channels, out_channels, J=1, biort='near_sym_b', qshift='qshift_b'):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.dtcwt = DTCWTForward(J=J, biort=biort, qshift=qshift)

    def forward(self, x):
        Yl, Yh = self.dtcwt(x)
        return self.conv(Yl), Yh


class WaveletUp(nn.Module):
    def __init__(self, in_channels, out_channels, J=1, biort='near_sym_b', qshift='qshift_b'):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.dtcwt_inverse = DTCWTInverse(biort=biort, qshift=qshift)
        self.halve_channels = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)

    def forward(self, x, skip_connection, Yh):
        x = self.halve_channels(x)  # Halve channels before IDTCWT
        x = self.dtcwt_inverse((x, Yh))
        x = torch.cat([skip_connection, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
