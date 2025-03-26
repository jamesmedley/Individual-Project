""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from kymatio.torch import Scattering2D


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.input_shape = (128, 128)
        J = 1
        L = 16

        self.scattering = ScatteringModule(J=J, shape=self.input_shape, L=L)
        n_scattering_channels = 1 + L * J + (L ** 2 * J * (J - 1)) // 2
        n_input_channels = n_channels * n_scattering_channels

        self.inc = (DoubleConv(n_input_channels, 128))  # replace with scattering
        # removed down1
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear, n_final_skip=67))  # skip: 64 + input channels
        self.outc = (OutConv(64, n_classes))

        self.skip_upconv = (SkipUpConv(n_input_channels, 64))

    def forward(self, x):
        x = x.contiguous()
        scattering_coeffs = self.scattering(x)
        B, C, scattering_channels, H, W = scattering_coeffs.shape
        scattering_coeffs = scattering_coeffs.view(B, -1, H, W)  # Shape: (B, C * scattering_channels, H', W')

        skip_scattering_coeffs = self.skip_upconv(scattering_coeffs)
        skip = torch.cat([x, skip_scattering_coeffs], dim=1)

        x1 = self.inc(scattering_coeffs)
        x3 = self.down2(x1)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x1)
        x = self.up4(x, skip)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
        self.skip_upconv = torch.utils.checkpoint(self.skip_upconv)
