""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from kymatio.torch import Scattering2D


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, J=1, L=8, input_shape=(128, 128)):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.input_shape = input_shape

        self.S1 = Scattering2D(J=J, shape=input_shape, L=L)
        self.S2 = Scattering2D(J=J+1, shape=input_shape, L=L)
        self.S3 = Scattering2D(J=J+2, shape=input_shape, L=L)
        self.S4 = Scattering2D(J=J+3, shape=input_shape, L=L)

        n_scattering_channels_1 = 1 + L * J + (L ** 2 * J * (J - 1)) // 2
        n_scattering_channels_2 = 1 + L * (J+1) + (L ** 2 * (J+1) * J) // 2
        n_scattering_channels_3 = 1 + L * (J+2) + (L ** 2 * (J+2) * (J+1)) // 2
        n_scattering_channels_4 = 1 + L * (J+3) + (L ** 2 * (J+3) * (J+2)) // 2

        self.inc = (DoubleConv(n_channels*n_scattering_channels_1, 128))
        self.down1 = (Down(128 + n_channels*n_scattering_channels_2, 256))
        self.down2 = (Down(256 + n_channels*n_scattering_channels_3, 512))
        factor = 2 if bilinear else 1
        self.down3 = (Down(512 + n_channels*n_scattering_channels_4, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear, n_final_skip=3))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        input_tensor = x.detach().clone()

        scat_1 = self.S1.scattering(x.contiguous())
        B, C, scattering_channels, H, W = scat_1.shape
        scat_1_coeffs = scat_1.view(B, -1, H, W)

        scat_2 = self.S2.scattering(x.contiguous())
        B, C, scattering_channels, H, W = scat_2.shape
        scat_2_coeffs = scat_2.view(B, -1, H, W)

        scat_3 = self.S3.scattering(x.contiguous())
        B, C, scattering_channels, H, W = scat_3.shape
        scat_3_coeffs = scat_3.view(B, -1, H, W)

        scat_4 = self.S4.scattering(x.contiguous())
        B, C, scattering_channels, H, W = scat_4.shape
        scat_4_coeffs = scat_4.view(B, -1, H, W)

        x1 = self.inc(scat_1_coeffs)
        x2 = self.down1(x1, scat_2_coeffs)
        x3 = self.down2(x2, scat_3_coeffs)
        x4 = self.down3(x3, scat_4_coeffs)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, input_tensor)  # skip 3 input channels
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)