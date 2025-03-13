import torch
from torch import nn


class Illumination_Estimator(nn.Module):
    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        """
        # 估计输入图像的光照信息
        Args:
            n_fea_middle：中间特征层的通道数。
            n_fea_in：输入特征的通道数 4 = 3(Image) + 1(Illumination Prior)
            n_fea_out：输出特征的通道数
        """
        super(Illumination_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle,
            kernel_size=5, stride=1, padding=2,
            bias=True, groups=n_fea_in
        )
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img: [b, 3, h, w]
        # 沿通道维度计算均值 mean_c: [b, 1, h, w]
        mean_c = img.mean(dim=1).unsqueeze(1)
        # 拼接通道作为输入
        # [b, 3, h, w] cat [b, 1, h, w] -> [b, 4, h, w]
        input = torch.cat([img, mean_c], dim=1)

        x_1 = self.conv1(input)
        light_up_feature = self.depth_conv(x_1)  # [b, middle, h, w]
        light_up_map = self.conv2(light_up_feature)  # [b, 3, h, w]

        return light_up_feature, light_up_map




























