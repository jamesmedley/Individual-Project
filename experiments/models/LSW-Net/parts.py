
""" Parts of the LSWNet model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DTCWTForward, DTCWTInverse, DWTInverse
from einops import rearrange


class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.single_conv(x)


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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = DoubleConv(7*in_channels, out_channels)  # 13 times for yl + 12 coefficients
        self.dtcwt = DTCWTForward(J=1)#.cuda()  # Single-level DTCWT
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        Yl, Yh = self.dtcwt(x)
        Yh = Yh[0]  # extract first level of decomposition
        # Yh shape: (N, C, O(rientations), H, W, I(real or imaginary))

        # Method 1 - 13 channels (Yl + 12 coefficients)
        #Yl_pooled = self.max_pool(Yl)
        #Yh_rearranged = rearrange(Yh, 'b c o h w i -> b (c i o) h w')
        #dtcwt_output_1 = torch.cat([Yl_pooled, Yh_rearranged], dim=1)

        # Method 2 - 7 channels (Magnitude-based)
        Yl_pooled = self.max_pool(Yl)
        magnitude = torch.sqrt(Yh[..., 0] ** 2 + Yh[..., 1] ** 2)  # Compute magnitude per orientation
        magnitude_concat = rearrange(magnitude, 'b c o h w -> b (c o) h w')  # Concatenate over orientations
        dtcwt_output_2 = torch.cat([Yl_pooled, magnitude_concat], dim=1)

        return self.double_conv(dtcwt_output_2)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False, n_final_skip=0):
        super().__init__()
        self.bilinear = bilinear
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        expected_in_channels = (in_channels // 2 + n_final_skip) if n_final_skip > 0 else in_channels
        self.conv = DoubleConv(expected_in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)