import torch
import torch.nn as nn
import torch.optim as optim
import torchsummary


class disc(nn.Module):
    def __init__(self):
        super(disc, self).__init__()

        self.model = nn.Sequential(
            self.__conv_layer(3, 64, norm=False),
            self.__conv_layer(64, 128),
            self.__conv_layer(128, 256),
            self.__conv_layer(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, 1, 1),
        )

    def __conv_layer(self, in_features, out_features, stride=2, norm=True):
        layer = [nn.Conv2d(in_features, out_features, 4, stride, 1)]

        if norm:
            layer.append(nn.InstanceNorm2d(out_features))

        layer.append(nn.LeakyReLU(0.2))
        layer = nn.Sequential(*layer)

        return layer

    def forward(self, x):
        return self.model(x)