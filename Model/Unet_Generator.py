import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import torchsummary


from Model.Layer_Modules import *
from Model.unet_parts import *


class UNet_vit(nn.Module):
    def __init__(self, n_channels, bilinear=False):
        super(UNet_vit, self).__init__()
        self.n_channels = n_channels
        # self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.pixel_VIT = PixelwiseViT(1024, 12, 384, 384, (1024, 14, 14), rezero=True)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.outc = OutConv(64, 3)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # print(x5.shape)

        vit_x = self.pixel_VIT(x5)

        x = self.up1(vit_x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits

# model = UNet_vit(3).to('cuda')
# torchsummary.summary(model, (3, 224, 224),device='cuda')