
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = DoubleConv(13*in_channels, out_channels)  # 13 times for yl + 12 coefficients
        self.dtcwt = DTCWTForward(J=1).cuda()  # Single-level DTCWT
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        Yl, Yh = self.dtcwt(x)
        Yh = Yh[0]  # extract first level of decomposition
        # Yh shape: (N, C, O(rientations), H, W, I(real or imaginary))

        # Method 1 - 13 channels (Yl + 12 coefficients)
        Yl_pooled = self.max_pool(Yl)
        Yh_rearranged = rearrange(Yh, 'b c o h w i -> b (c i o) h w')
        dtcwt_output_1 = torch.cat([Yl_pooled, Yh_rearranged], dim=1)

        # Method 2 - 7 channels (Magnitude-based)
        #Yl_pooled = self.max_pool(Yl)
        #magnitude = torch.sqrt(Yh[..., 0] ** 2 + Yh[..., 1] ** 2)  # Compute magnitude per orientation
        #magnitude_concat = rearrange(magnitude, 'b c o h w -> b (c o) h w')  # Concatenate over orientations
        #dtcwt_output_2 = torch.cat([Yl_pooled, magnitude_concat], dim=1)

        return self.double_conv(dtcwt_output_1)


class WaveletUp(nn.Module):
    """Upscaling using inverse Discrete Wavelet Transform (IDWT) with skip connection"""
    def __init__(self, in_channels, out_channels, n_final_skip=0):
        super().__init__()
        # firstly a single conv to 6.5x number of features (this guarantees divisible by 13)
        self.single_conv = SingleConv(in_channels, 6.5*in_channels)

        # then upsample 1/13th for yl
        self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        # followed by the idwt, which will 1/13 the number of features.
        self.idwt = DTCWTInverse().cuda()

        # finally, we have ordinary double convolution to expected number of out channels
        expected_in_channels = (in_channels // 2 + n_final_skip) if n_final_skip > 0 else in_channels  # in case of extra input skip
        self.double_conv = DoubleConv(expected_in_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.single_conv(x)

        batch, channels, height, width = x.shape
        num_base_channels = channels // 13
        Yl, Yh = torch.split(x, [num_base_channels, 12 * num_base_channels], dim=1)

        Yl = self.up(Yl)  # upsample yl features

        # Yh shape: (N, C, O(rientations), H, W, I(real or imaginary))
        Yh = rearrange(Yh, 'b (c o i) h w -> b c o h w i', i=2, o=6)

        # Perform inverse DTCWT
        x = self.idwt((Yl, [Yh]))

        return self.double_conv(torch.cat([skip_connection, x], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
