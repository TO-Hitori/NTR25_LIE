from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ESDNet(nn.Module):
    def __init__(
            self,
            en_channels=48,
            en_growth_rate=32,
            de_channels=64,
            de_growth_rate=32,
            sam_layers=2,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels=en_channels,
            growth_rate=en_growth_rate,
            sam_layers=sam_layers
        )
        self.decoder = Decoder(
            en_channels=en_channels,
            de_channels=de_channels,
            growth_rate=de_growth_rate,
            sam_layers=sam_layers
        )

    def forward(self, x):
        _, _, H, W = x.shape
        scale_factor = 2 ** 5
        pad_h = (scale_factor - H % scale_factor) % scale_factor
        pad_w = (scale_factor - W % scale_factor) % scale_factor

        if pad_h != 0 or pad_w != 0:
            F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        y1, y2, y3 = self.encoder(x)
        o1, o2, o3 = self.decoder(y1, y2, y3)

        o1 = o1[:, :, :H, :W]
        o2 = o2[:, :, :H//2, :W//2]
        o3 = o3[:, :, :H//4, :W//4]

        return o1, o2, o3

"""
basic block
"""

class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, dilation_list: List[int], use_residual: bool = True):
        super().__init__()
        num_layers = len(dilation_list)
        self.use_residual = use_residual

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in_channels = in_channels + i * growth_rate
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=layer_in_channels,
                              out_channels=growth_rate,
                              kernel_size=3,
                              stride=1,
                              padding=dilation_list[i],
                              dilation=dilation_list[i],
                              bias=True,
                              padding_mode='reflect'),
                    nn.SiLU(inplace=True)
                )
            )
        self.final_conv = nn.Conv2d(in_channels + num_layers * growth_rate, in_channels, kernel_size=1)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            layer_input = torch.cat(features, dim=1)
            layer_output = layer(layer_input)
            features.append(layer_output)
        all_feature = torch.cat(features, dim=1)
        output = self.final_conv(all_feature)

        if self.use_residual:
            output += x
        return output


"""
SAM: Semantic-Aligned Scale-Aware Module
    - 1. Pyramid context extraction
    - 2. Cross-scale dynamic fusion
"""


class Pyramid_context_extraction(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, dilation_list: List[int]):
        super().__init__()
        self.shared_block = ResidualDenseBlock(in_channels, growth_rate, dilation_list, use_residual=False)

    def forward(self, x):
        f0 = x
        f1 = F.interpolate(x, scale_factor=0.50, mode='bilinear')
        f2 = F.interpolate(x, scale_factor=0.25, mode='bilinear')

        y0, y1, y2 = map(self.shared_block, (f0, f1, f2))
        y1 = F.interpolate(y1, scale_factor=2, mode='bilinear')
        y2 = F.interpolate(y2, scale_factor=4, mode='bilinear')

        return y0, y1, y2


class Cross_scale_dynamic_fusion(nn.Module):
    def __init__(self, in_channels: int, mlp_ratio: int = 4):
        super().__init__()
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // mlp_ratio),
            nn.GELU(),
            nn.Linear(in_channels // mlp_ratio, in_channels // mlp_ratio),
            nn.GELU(),
            nn.Linear(in_channels // mlp_ratio, in_channels),
            nn.Tanh()
        )

    def forward(self, y0, y1, y2):
        v0, v1, v2 = map(self.GAP, (y0, y1, y2))
        V = torch.cat([v0, v1, v2], dim=1)
        W = self.mlp(V.squeeze(-1).squeeze(-1)).unsqueeze(-1).unsqueeze(-1)
        w0, w1, w2 = torch.chunk(W, 3, 1)
        out = w0 * y0 + w1 * y1 + w2 * y2
        return out


class SAM(nn.Module):
    """
    每个SAM下采样两次，输入尺度应为 4 的倍数
    """

    def __init__(self, in_channels: int, growth_rate: int, dilation_list: List[int]):
        super().__init__()
        self.PCE = Pyramid_context_extraction(in_channels, growth_rate, dilation_list)
        self.CSDF = Cross_scale_dynamic_fusion(in_channels * 3)

    def forward(self, x):
        y0, y1, y2 = self.PCE(x)
        out = self.CSDF(y0, y1, y2)
        return out + x


class Encoder_Layer(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, layer: int, sam_layers: int):
        super().__init__()
        self.RDB = ResidualDenseBlock(in_channels=in_channels, growth_rate=growth_rate, dilation_list=[1, 2, 1])
        sam_layers = [SAM(in_channels, growth_rate, dilation_list=[1, 2, 3, 2, 1]) for _ in range(sam_layers)]
        self.SAM_layer = nn.Sequential(*sam_layers)
        self.down_sample = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, 2, 1),
            nn.SiLU(inplace=True)
        ) if layer < 3 else nn.Identity()

    def forward(self, x):
        hidden = self.RDB(x)
        hidden = self.SAM_layer(hidden)
        hidden_downsampled = self.down_sample(hidden)
        return hidden, hidden_downsampled


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 48, growth_rate: int = 32, sam_layers: int = 2):
        super().__init__()
        self.pixel_unshuffle_conv_in = nn.Sequential(
            nn.PixelUnshuffle(downscale_factor=2),
            nn.Conv2d(in_channels=12, out_channels=in_channels, kernel_size=5, stride=1, padding=2,
                      padding_mode='reflect')
        )
        self.encoder_layer1 = Encoder_Layer(in_channels=in_channels, growth_rate=growth_rate, layer=1,
                                            sam_layers=sam_layers)
        self.encoder_layer2 = Encoder_Layer(in_channels=in_channels * 2, growth_rate=growth_rate, layer=2,
                                            sam_layers=sam_layers)
        self.encoder_layer3 = Encoder_Layer(in_channels=in_channels * 4, growth_rate=growth_rate, layer=3,
                                            sam_layers=sam_layers)

    def forward(self, image):
        # [b, 3, h, w] -> [b, 12, h//2, w//2] -> [b, c, h//2, w//2]
        hidden = self.pixel_unshuffle_conv_in(image)
        # [b, c, h//2, w//2], [b, 2c, h//4, w//4]
        hidden1, hidden_downsampled1 = self.encoder_layer1(hidden)
        # [b, 2c, h//4, w//4], [b, 4c, h//8, w//8]
        hidden2, hidden_downsampled2 = self.encoder_layer2(hidden_downsampled1)
        # [b, 4c, h//8, w//8]
        hidden3, _ = self.encoder_layer3(hidden_downsampled2)
        # [b, c, h//2, w//2], [b, 2c, h//4, w//4], [b, 4c, h//8, w//8]
        return hidden1, hidden2, hidden3


class Decoder_Layer(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, sam_layers: int):
        super().__init__()
        self.RDB = ResidualDenseBlock(in_channels, growth_rate, [1, 2, 1])
        sam_layers = [SAM(in_channels, growth_rate, dilation_list=[1, 2, 3, 2, 1]) for _ in range(sam_layers)]
        self.SAM_layer = nn.Sequential(*sam_layers)

        self.pixel_shuffle_conv_out = nn.Sequential(
            nn.Conv2d(in_channels, 12, 3, 1, 1),
            nn.PixelShuffle(upscale_factor=2)
        )

    def forward(self, x):
        hidden = self.RDB(x)
        hidden = self.SAM_layer(hidden)
        # [b, c, h, w] -> [b, 12, h, w] -> [b, 3, 2h, 2w]
        hidden_out = self.pixel_shuffle_conv_out(hidden)
        # [b, c, h, w] -> [b, c, 2h, 2w]
        hidden_upsampled = F.interpolate(hidden, scale_factor=2, mode='bilinear')
        # [b, 3, 2h, 2w], [b, c, 2h, 2w]
        return hidden_out, hidden_upsampled


class Decoder(nn.Module):
    def __init__(self, en_channels, de_channels, growth_rate, sam_layers):
        super().__init__()
        self.pre_conv3 = nn.Sequential(
            nn.Conv2d(4 * en_channels, de_channels, 3, 1, 1),
            nn.SiLU()
        )
        self.pre_conv2 = nn.Sequential(
            nn.Conv2d(2 * en_channels + de_channels, de_channels, 3, 1, 1),
            nn.SiLU()
        )
        self.pre_conv1 = nn.Sequential(
            nn.Conv2d(en_channels + de_channels, de_channels, 3, 1, 1),
            nn.SiLU()
        )

        self.decoder_layer1 = Decoder_Layer(de_channels, growth_rate, sam_layers)
        self.decoder_layer2 = Decoder_Layer(de_channels, growth_rate, sam_layers)
        self.decoder_layer3 = Decoder_Layer(de_channels, growth_rate, sam_layers)

    def forward(self, y1, y2, y3):
        # input: [b, e, h//2, w//2], [b, 2e, h//4, w//4], [b, 4e, h//8, w//8]
        # [b, 4e, h//8, w//8]
        x3 = y3
        # [b, 4e, h//8, w//8] -> [b, d, h//8, w//8]
        x3 = self.pre_conv3(x3)
        # [b, 3, h//4, w//4], [b, d, h//4, w//4]
        hidden_out3, hidden_upsampled3 = self.decoder_layer3(x3)

        # [b, 2e, h//4, w//4] cat [b, d, h//4, w//4] -> [b, 2e+d, h//4, w//4]
        x2 = torch.cat([y2, hidden_upsampled3], dim=1)
        # [b, 2e+d, h//4, w//4] -> [b, d, h//4, w//4]
        x2 = self.pre_conv2(x2)
        # [b, 3, h//2, w//2], [b, d, h//2, w//2]
        hidden_out2, hidden_upsampled2 = self.decoder_layer3(x2)

        # [b, e, h//2, w//2] cat [b, d, h//2, w//2] -> [b, e+d, h//2, w//2]
        x1 = torch.cat([y1, hidden_upsampled2], dim=1)
        # [b, e+d, h//2, w//2] -> [b, d, h//2, w//2]
        x1 = self.pre_conv1(x1)
        # [b, 3, h, w]
        hidden_out1, _ = self.decoder_layer1(x1)

        # [b, 3, h, w], [b, 3, h//2, w//2], [b, 3, h//4, w//4]
        return hidden_out1, hidden_out2, hidden_out3


def image_info(image):  # for debug
    print("=-" * 20)
    print("---type:", type(image))
    print("---shape:", image.shape)
    print("---dtype:", image.dtype)
    print("---range: ", image.min(), image.max())


from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    device = torch.device('cuda')

    tb = SummaryWriter('logs/models/ESDNet')

    model = ESDNet().to(device)
    print(model)
    x0 = torch.randn(1, 3, 512, 512).to(device)

    o1, o2, o3 = model(x0)
    image_info(o1)
    image_info(o2)
    image_info(o3)
    tb.add_graph(model, x0)
    tb.close()
