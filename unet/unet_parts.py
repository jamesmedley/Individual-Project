""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt


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


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class WaveletDown(nn.Module):
    """Downscaling using Discrete Wavelet Transform (DWT) instead of max-pooling"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        # Apply 2D Discrete Wavelet Transform (DWT)
        x_detached = x.detach().cpu().numpy()
        coeffs = pywt.dwt2(x_detached, 'haar')
        LL, (LH, HL, HH) = coeffs  # Extract the four subbands

        # Convert back to PyTorch tensors
        LL = torch.tensor(LL, dtype=torch.float32).to(x.device)
        LH = torch.tensor(LH, dtype=torch.float32).to(x.device)
        HL = torch.tensor(HL, dtype=torch.float32).to(x.device)
        HH = torch.tensor(HH, dtype=torch.float32).to(x.device)

        # Pass LL subband through convolutional layers
        x_out = self.conv(LL)

        return x_out, (LH, HL, HH)  # Store high-frequency components for later


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class WaveletUp(nn.Module):
    """Upscaling using inverse Discrete Wavelet Transform (IDWT) with skip connection"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.halve_channels = nn.Conv2d(in_channels, in_channels//2, kernel_size=1)

    def forward(self, x, skip_connection, coeffs_down):
        LH, HL, HH = coeffs_down
        LL = x

        LL = self.halve_channels(LL)  # halve number of channels in LL from previous block

        coeffs_up_np = (LL.detach().cpu().numpy(),
                        (LH.detach().cpu().numpy(), HL.detach().cpu().numpy(), HH.detach().cpu().numpy()))

        image = pywt.idwt2(coeffs_up_np, 'haar')  # Use Haar for IDWT
        image = torch.tensor(image, dtype=torch.float32).to(x.device)

        x = torch.cat([skip_connection, image], dim=1)

        x_out = self.conv(x)

        return x_out


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
