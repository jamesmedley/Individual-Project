""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from kymatio.torch import Scattering2D


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, J=1, L=16, input_shape=(128, 128)):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.input_shape = input_shape

        self.S = Scattering2D(J=J, shape=input_shape, L=L)
        n_scattering_channels = 1 + L * J + (L ** 2 * J * (J - 1)) // 2

        self.inc = (DoubleConv(n_channels*n_scattering_channels, 128))
        self.down1 = (Down(128, 256))
        self.down2 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down3 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear, n_final_skip=3))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        input_tensor = x.detach().clone()

        scat_1 = self.S.scattering(x.contiguous())
        B, C, scattering_channels, H, W = scat_1.shape
        scat_1_coeffs = scat_1.view(B, -1, H, W)

        x1 = self.inc(scat_1_coeffs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
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