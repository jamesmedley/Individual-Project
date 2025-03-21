""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder (Contracting Path)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = WaveletDown(64, 128)  # Using WaveletDown instead of Down
        self.down2 = WaveletDown(128, 256)
        self.down3 = WaveletDown(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = WaveletDown(512, 1024 // factor)

        # Decoder (Expanding Path)
        self.up1 = WaveletUp(1024, 512 // factor)
        self.up2 = WaveletUp(512, 256 // factor)
        self.up3 = WaveletUp(256, 128 // factor)
        self.up4 = WaveletUp(128, 64)

        # Output layer
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2, coeffs2 = self.down1(x1)
        x3, coeffs3 = self.down2(x2)
        x4, coeffs4 = self.down3(x3)
        x5, coeffs5 = self.down4(x4)

        x = self.up1(x5, x4, coeffs5)
        x = self.up2(x, x3, coeffs4)
        x = self.up3(x, x2, coeffs3)
        x = self.up4(x, x1, coeffs2)

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
