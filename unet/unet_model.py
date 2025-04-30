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
        self.J = 3
        self.L = 8

        self.S = Scattering2D(J=self.J, shape=self.input_shape, L=self.L)
        n_order1 = n_channels * (1 + self.J * self.L)
        n_order2 = n_channels * ((self.L ** 2 * self.J * (self.J - 1)) // 2)
        n_input_channels = n_order1 + n_order2

        self.inc = (DoubleConv(n_input_channels, 512))
        # removed down1
        # removed down2
        # removed down3
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear, n_final_skip=0))
        self.up2 = (Up(512, 256 // factor, bilinear, n_final_skip=0))
        self.up3 = (Up(256, 128 // factor, bilinear, n_final_skip=0))
        self.up4 = (Up(128, 64, bilinear, n_final_skip=67))  # skip: 64 + input channels
        self.outc = (OutConv(64, n_classes))

        self.skip_upconv1 = (SkipTripleUpConv(n_input_channels, 64))

    def forward(self, x):
        scattering_coeffs = self.S.scattering(x.contiguous())  # Shape: (B, C, scattering_channels, H', W')
        B, C, scat_channels, H, W = scattering_coeffs.shape
        all_coeffs = scattering_coeffs.view(B, -1, H, W)  # Shape: (B, C * scattering_channels, H', W')

        skip = self.skip_upconv1(all_coeffs)
        skip = torch.cat([x, skip], dim=1)

        x1 = self.inc(all_coeffs)
        x5 = self.down4(x1)
        x = self.up1(x5, None)
        x = self.up2(x, None)
        x = self.up3(x, None)
        x = self.up4(x, skip)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
        self.skip_upconv1 = torch.utils.checkpoint(self.skip_upconv1)
        self.skip_upconv2 = torch.utils.checkpoint(self.skip_upconv2)
        self.skip_upconv3 = torch.utils.checkpoint(self.skip_upconv3)

