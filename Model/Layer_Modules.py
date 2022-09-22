import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, f):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(f, f, 3, 1, 1),
            nn.BatchNorm2d(f), nn.ReLU(),
            nn.Conv2d(f, f, 3, 1, 1)
        )
        self.norm = nn.BatchNorm2d(f)

    def forward(self, x):
        y = self.conv(x)
        y = self.norm(y)
        y = F.relu(y)
        return F.relu(x+y)


class PositionWiseFFN(nn.Module):
    def __init__(self, in_features, ffn_features) -> None:
        super().__init__()

        self.FFN = nn.Sequential(
            nn.Linear(in_features, ffn_features),
            nn.GELU(),
            nn.Linear(ffn_features, in_features),
        )

    def forward(self, x):
        return self.FFN(x)


class PPEG(nn.Module):
    def __init__(self, H, W, in_features=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(in_features, in_features, 7, 1, 7 // 2, groups=in_features)
        self.proj1 = nn.Conv2d(in_features, in_features, 5, 1, 5 // 2, groups=in_features)
        self.proj2 = nn.Conv2d(in_features, in_features, 3, 1, 3 // 2, groups=in_features)
        self.H = H
        self.W = W

    def forward(self, x):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, self.H, self.W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransBlock(nn.Module):
    def __init__(self, in_features, out_features, rezero=True):
        super(TransBlock, self).__init__()

        self.norm1 = nn.LayerNorm(in_features)
        self.attention = nn.MultiheadAttention(in_features, 8)

        # self.ppeg = PPEG(1, 1, in_features=in_features)

        self.norm2 = nn.LayerNorm(in_features)
        self.ffn = PositionWiseFFN(in_features, out_features)

        self.rezero = rezero

        if rezero:
            self.re_alpha = nn.Parameter(torch.zeros((1,)))
        else:
            self.re_alpha = 1

    def forward(self, x):
        y1 = self.norm1(x)
        y1, _atten_weights = self.attention(y1, y1, y1)

        y = x + self.re_alpha * y1

        # y = self.ppeg(y)

        y2 = self.norm1(y)
        y2 = self.ffn(y2)

        y = y + self.re_alpha * y2

        return y


class FourierEmbedding(nn.Module):
    # arXiv: 2011.13775

    def __init__(self, features, height, width):
        super().__init__()
        self.projector = nn.Linear(2, features)
        self._height = height
        self._width = width

    def forward(self, y, x):
        # x : (N, L)
        # y : (N, L)
        x_norm = 2 * x / (self._width - 1) - 1
        y_norm = 2 * y / (self._height - 1) - 1

        # z : (N, L, 2)
        z = torch.cat((x_norm.unsqueeze(2), y_norm.unsqueeze(2)), dim=2)

        return torch.sin(self.projector(z))


class TransformerEncoder(nn.Module):

    def __init__(
            self, features, ffn_features, n_blocks, rezero=True):
        super().__init__()

        self.encoder = nn.Sequential(*[
            TransBlock(
                features, ffn_features, rezero,
            ) for _ in range(n_blocks)
        ])

    def forward(self, x):
        # x : (N, L, features)

        # y : (L, N, features)
        y = x.permute((1, 0, 2))
        y = self.encoder(y)

        # result : (N, L, features)
        result = y.permute((1, 0, 2))

        return result


class ViTInput(nn.Module):
    def __init__(
            self, input_features, embed_features, features, height, width):
        super().__init__()
        self._height = height
        self._width = width

        x = torch.arange(width).to(torch.float32)
        y = torch.arange(height).to(torch.float32)

        x, y = torch.meshgrid(x, y)
        self.x = x.reshape((1, -1))
        self.y = y.reshape((1, -1))

        self.register_buffer('x_const', self.x)
        self.register_buffer('y_const', self.y)

        self.embed = FourierEmbedding(embed_features, height, width)
        self.output = nn.Linear(embed_features + input_features, features)

    def forward(self, x):
        # x     : (N, L, input_features)
        # embed : (1, height * width, embed_features)
        #       = (1, L, embed_features)
        embed = self.embed(self.y_const, self.x_const)

        # embed : (1, L, embed_features)
        #      -> (N, L, embed_features)
        embed = embed.expand((x.shape[0], *embed.shape[1:]))

        # result : (N, L, embed_features + input_features)
        result = torch.cat([embed, x], dim=2)

        # (N, L, features)
        return self.output(result)


class PixelwiseViT(nn.Module):
    def __init__(
            self, features, n_blocks, ffn_features, embed_features, image_shape, rezero=True):
        super().__init__()

        self.image_shape = image_shape

        self.trans_input = ViTInput(
            image_shape[0], embed_features, features,
            image_shape[1], image_shape[2],
        )

        self.encoder = TransformerEncoder(
            features, ffn_features, n_blocks, rezero
        )

        self.trans_output = nn.Linear(features, image_shape[0])

    def forward(self, x):
        # x : (N, C, H, W)

        # itokens : (N, C, H * W)
        itokens = x.view(*x.shape[:2], -1)

        # itokens : (N, C,     H * W)
        #        -> (N, H * W, C    )
        #         = (N, L,     C)
        itokens = itokens.permute((0, 2, 1))

        # y : (N, L, features)
        y = self.trans_input(itokens)
        y = self.encoder(y)

        # otokens : (N, L, C)
        otokens = self.trans_output(y)

        # otokens : (N, L, C)
        #        -> (N, C, L)
        #         = (N, C, H * W)
        otokens = otokens.permute((0, 2, 1))

        # result : (N, C, H, W)
        result = otokens.view(*otokens.shape[:2], *self.image_shape[1:])

        return result


class channel_compression(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
          in_channels (int):  Number of input channels.
          out_channels (int): Number of output channels.
          stride (int):       Controls the stride.
        """
        super(channel_compression, self).__init__()

        self.skip = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
          self.skip = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels))
        else:
          self.skip = None

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.block(x)
        out += (x if self.skip is None else self.skip(x))
        out = F.relu(out)
        return out


class Res_PixelwiseViT(nn.Module):
    def __init__(
            self, features, vit_input_feature, n_blocks, ffn_features, embed_features, image_shape, rezero=True):
        super().__init__()

        self.image_shape = image_shape

        encoder = [
            self.__conv_block(features, vit_input_feature)
            ]

        self.trans_input = ViTInput(
            image_shape[0], embed_features, vit_input_feature,
            image_shape[1], image_shape[2],
        )

        self.vit_encoder = TransformerEncoder(
            vit_input_feature, ffn_features, n_blocks, rezero
        )

        self.trans_output = nn.Linear(vit_input_feature, image_shape[0])

        decoder = [
            self.__conv_block(vit_input_feature, features, upsample=True)
            ]

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)
        self.channel_compresion = channel_compression(features * 2, features)
        self.residual = ResBlock(features)

    def __conv_block(self, in_features, out_features, upsample=False):
        if upsample:
            conv = nn.ConvTranspose2d(in_features, out_features, 3, 2, 1, output_padding=1)

        else:
            conv = nn.Conv2d(in_features, out_features, 3, 2, 1)

        return nn.Sequential(
            conv,
            nn.InstanceNorm2d(out_features),
            nn.ReLU()
        )

    def forward(self, x):
        # x : (N, C, H, W)

        # itokens : (N, C, H * W)
        x_down = self.encoder(x)
        itokens = x_down.view(*x_down.shape[:2], -1)

        # itokens : (N, C,     H * W)
        #        -> (N, H * W, C    )
        #         = (N, L,     C)
        itokens = itokens.permute((0, 2, 1))

        # y : (N, L, features)
        y = self.trans_input(itokens)
        y = self.vit_encoder(y)

        # otokens : (N, L, C)
        otokens = self.trans_output(y)

        # otokens : (N, L, C)
        #        -> (N, C, L)
        #         = (N, C, H * W)
        otokens = otokens.permute((0, 2, 1))

        # result : (N, C, H, W)
        vit_result = otokens.view(*otokens.shape[:2], *self.image_shape[1:])

        x_up = self.decoder(vit_result)

        x_concat = torch.cat((x, x_up), dim=1)

        x_compression = self.channel_compresion(x_concat)

        x_res_result = self.residual(x_compression)

        return x_res_result