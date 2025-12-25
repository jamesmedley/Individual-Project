""" Full assembly of the parts to form the complete network """

from .parts import *
from kymatio.torch import Scattering2D


class WSN_UNet_J4(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(WSN_UNet_J4, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.input_shape = (576, 560)
        self.J = 4
        self.L = 8

        self.S = Scattering2D(J=self.J, shape=self.input_shape, L=self.L)
        n_order1 = n_channels * (1 + self.J * self.L)
        n_order2 = n_channels * ((self.L ** 2 * self.J * (self.J - 1)) // 2)
        n_input_channels = n_order1 + n_order2

        self.inc = (DoubleConv(n_input_channels, 1024))
        # removed down1
        # removed down2
        # removed down3
        factor = 2 if bilinear else 1
        # removed down4
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear, n_final_skip=67))  # skip: 64 + input channels
        self.outc = (OutConv(64, n_classes))

        self.skip_upconv1 = (SkipFourUpConv(n_order1, 64))
        self.skip_upconv2 = (SkipTripleUpConv(n_order1, 128))
        self.skip_upconv3 = (SkipDoubleUpConv(n_order2, 256))
        self.skip_upconv4 = (SkipUpConv(n_order2, 512))

    def forward(self, x):
        scattering_coeffs = self.S.scattering(x.contiguous())  # Shape: (B, C, scattering_channels, H', W')
        B, C, scat_channels, H, W = scattering_coeffs.shape
        all_coeffs = scattering_coeffs.view(B, -1, H, W)  # Shape: (B, C * scattering_channels, H', W')

        # Compute index boundaries for orders
        n_order0 = 1
        n_order1 = self.J * self.L
        n_order2 = (self.L ** 2 * self.J * (self.J - 1)) // 2

        # Combine Order 0 and Order 1 into a single tensor
        coeffs_order1 = scattering_coeffs[:, :, :n_order0 + n_order1, :, :]  # (B, C, n_order0 + n_order1, H', W')
        coeffs_order2 = scattering_coeffs[:, :, n_order0 + n_order1:, :, :]  # (B, C, n_order2, H', W')

        # Reshape for CNN layers: Merge scattering channels into the channel dimension but keep RGB separate
        coeffs_order1 = coeffs_order1.reshape(B, C * (n_order0 + n_order1), H, W)  # (B, C * (n_order0 + n_order1), H', W')
        coeffs_order2 = coeffs_order2.reshape(B, C * n_order2, H, W)  # (B, C * n_order2, H', W')
        skip_scat_coeffs_o1 = self.skip_upconv1(coeffs_order1)  # param: first n_order1 coefficients

        skip_order_1 = torch.cat([x, skip_scat_coeffs_o1], dim=1)
        skip_order_2 = self.skip_upconv2(coeffs_order1)
        skip_order_3 = self.skip_upconv3(coeffs_order2)
        skip_order_4 = self.skip_upconv4(coeffs_order2)

        x1 = self.inc(all_coeffs)
        x = self.up1(x1, skip_order_4)
        x = self.up2(x, skip_order_3)
        x = self.up3(x, skip_order_2)
        x = self.up4(x, skip_order_1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
        self.skip_upconv1 = torch.utils.checkpoint(self.skip_upconv1)
        self.skip_upconv2 = torch.utils.checkpoint(self.skip_upconv2)
        self.skip_upconv3 = torch.utils.checkpoint(self.skip_upconv3)
