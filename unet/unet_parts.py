
""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DTCWTForward, DTCWTInverse
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
    def __init__(self, in_channels, out_channels, wave='haar'):
        super().__init__()
        self.double_conv = DoubleConv(13*in_channels, out_channels)  # 13 times for yl + 12 coefficients
        self.dtcwt = DTCWTForward(J=1).cuda()  # Single-level DTCWT
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        Yl, Yh = self.dtcwt(x)
        Yl = self.max_pool(Yl[0].unsqueeze(0))
        Yh = rearrange(Yh[0], 'b c o h w i -> b (c i o) h w')
        dtcwt_output = torch.cat([Yl, Yh], dim=1)
        return self.double_conv(dtcwt_output)


class WaveletUp(nn.Module):
    """Upscaling using inverse Discrete Wavelet Transform (IDWT) with skip connection"""
    def __init__(self, in_channels, out_channels, n_final_skip=0, wave='haar'):
        super().__init__()
        # firstly a single conv to double number of features
        self.single_conv = SingleConv(in_channels, 2*in_channels)

        # followed by the idwt, which will quarter the number of features.
        self.idwt = DWTInverse(mode='zero', wave=wave).cuda()

        # finally, we have ordinary double convolution to expected number of out channels
        expected_in_channels = (in_channels // 2 + n_final_skip) if n_final_skip > 0 else in_channels  # in case of extra input skip
        self.double_conv = DoubleConv(expected_in_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.single_conv(x)

        C_quarter = x.shape[1] // 4
        x_LL = x[:, :C_quarter]  # First quarter is LL
        x_LH = x[:, C_quarter:2 * C_quarter]  # Second quarter is LH
        x_HL = x[:, 2 * C_quarter:3 * C_quarter]  # Third quarter is HL
        x_HH = x[:, 3 * C_quarter:]  # Fourth quarter is HH

        x = self.idwt((x_LL, [torch.stack([x_LH, x_HL, x_HH], dim=2)]))  # Output: N x C x (2H) x (2W)

        x = torch.cat([skip_connection, x], dim=1)
        return self.double_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
