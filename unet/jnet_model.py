""" Full assembly of the parts to form the complete network """
import torch.utils.checkpoint

from .jnet_parts import *
from kymatio.torch import Scattering2D


class JNet(nn.Module):
    def __init__(self, n_channels, n_classes, L=16, J=1, input_shape=(128, 128), bilinear=False):
        super(JNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.J = J
        self.L = L
        self.scattering_shapes = [input_shape]
        for _ in range(3):
            self.scattering_shapes.append(tuple(x // 2 for x in self.scattering_shapes[-1]))

        self.S = Scattering2D(J=J, shape=input_shape, L=L)
        self.S1 = Scattering2D(J=J, shape=self.scattering_shapes[1], L=L)
        self.S2 = Scattering2D(J=J, shape=self.scattering_shapes[2], L=L)
        self.S3 = Scattering2D(J=J, shape=self.scattering_shapes[3], L=L)

        scattering_channels = 1 + L * J + (L ** 2 * J * (J - 1)) // 2
        n_input_channels = n_channels * scattering_channels

        self.inc = (DoubleConv(n_input_channels, 128))
        self.double1 = (DoubleConv(128*scattering_channels, 256))
        self.double2 = (DoubleConv(256*scattering_channels, 512))
        self.double3 = (DoubleConv(512*scattering_channels, 1024))
        self.up1 = (Up(1024, 512))  # skip from fourth block
        self.up2 = (Up(512, 256))  # skip from third block
        self.up3 = (Up(256, 128))  # skip from second block
        self.up4 = (Up(128, 64, final_skip=3))  # skip from input
        self.outc = (OutConv(64, n_classes))  # skip from input

    def forward(self, x):
        input_tensor = x.detach().clone()

        scat_input = self.S.scattering(x.contiguous())
        B, C, scattering_channels, H, W = scat_input.shape
        scat_input = scat_input.view(B, -1, H, W)

        x1 = self.inc(scat_input)
        down1 = self.S1.scattering(x1.contiguous())
        B, C, scattering_channels, H, W = down1.shape
        down1 = down1.view(B, -1, H, W)

        x2 = self.double1(down1)
        down2 = self.S2.scattering(x2.contiguous())
        B, C, scattering_channels, H, W = down2.shape
        down2 = down2.view(B, -1, H, W)

        x3 = self.double2(down2)
        down3 = self.S3.scattering(x3.contiguous())
        B, C, scattering_channels, H, W = down3.shape
        down3 = down3.view(B, -1, H, W)

        x4 = self.double3(down3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, input_tensor)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.double1 = torch.utils.checkpoint(self.down1)
        self.double2 = torch.utils.checkpoint(self.down2)
        self.double3 = torch.utils.checkpoint(self.down3)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
