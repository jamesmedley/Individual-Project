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
        self.J = 2
        self.L = 8

        self.S = Scattering2D(J=self.J, shape=self.input_shape, L=self.L)
        n_order1 = n_channels * (1 + self.J * self.L)
        n_order2 = n_channels * ((self.L ** 2 * self.J * (self.J - 1)) // 2)
        n_input_channels = n_order1 + n_order2

        self.inc = (DoubleConv(n_input_channels, 256))  # replace with scattering
        # removed down1
        # removed down2
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear, n_final_skip=67))  # skip: 64 + input channels
        self.outc = (OutConv(64, n_classes))

        self.skip_upconv1 = (SkipDoubleUpConv(n_input_channels, 64))  # all coeffs to skip to each layer
        self.skip_upconv2 = (SkipUpConv(n_input_channels, 128))

    def forward(self, x):
        scattering_coeffs = self.S.scattering(x.contiguous())  # Shape: (B, C, scattering_channels, H', W')
        B, C, scat_channels, H, W = scattering_coeffs.shape
        all_coeffs = scattering_coeffs.view(B, -1, H, W)  # Shape: (B, C * scattering_channels, H', W')

        skip_order_1 = torch.cat([x, all_coeffs], dim=1)
        skip_order_2 = self.skip_upconv2(all_coeffs)

        x1 = self.inc(all_coeffs)
        x4 = self.down3(x1)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x1)
        x = self.up3(x, skip_order_2)
        x = self.up4(x, skip_order_1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
        self.skip_upconv1 = torch.utils.checkpoint(self.skip_upconv)
        self.skip_upconv2 = torch.utils.checkpoint(self.skip_upconv)


