import torch
import torch.nn as nn
import torch.nn.functional as F

#差异化互补模块,需要输入一致[B,C,H,W]
class MDCM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, f_hsi, f_sar):
        # 差异计算
        gap_hsi = F.adaptive_avg_pool2d(f_sar - f_hsi, (1, 1))  # [B,C,1,1]
        gap_sar = F.adaptive_avg_pool2d(f_hsi - f_sar, (1, 1))

        # 动态权重生成
        hsi_weights = torch.sigmoid(gap_hsi)
        sar_weights = torch.sigmoid(gap_sar)

        # 跨模态特征增强
        f_hsi_enhanced = f_hsi + hsi_weights * f_sar
        f_sar_enhanced = f_sar + sar_weights * f_hsi

        return f_hsi_enhanced, f_sar_enhanced   #[B,C,H,W]

class _MCASPPConv(nn.Sequential):
    def __init__(self, in_channels, inter_channels, out_channels, atrous_rate,
                 drop_rate=0.1, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(_MCASPPConv, self).__init__()
        self.add_module('conv1', nn.Conv2d(in_channels, inter_channels, 1)),
        self.add_module('bn1', norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs))),
        self.add_module('relu1', nn.ReLU(True)),
        # 3x3空洞卷积，可以保持输出尺寸
        self.add_module('conv2', nn.Conv2d(inter_channels, out_channels, 3, dilation=atrous_rate, padding=atrous_rate)),
        self.add_module('bn2', norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs))),
        self.add_module('relu2', nn.ReLU(True)),
        self.drop_rate = drop_rate

    def forward(self, x):
        features = super(_MCASPPConv, self).forward(x)
        if self.drop_rate > 0:
            features = F.dropout(features, p=self.drop_rate, training=self.training)
        return features


class MCASPP(nn.Module):
    def __init__(self, channels, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(MCASPP, self).__init__()
        self.aspp_3 = _MCASPPConv(channels//2, channels//2, channels, 3, 0.1,
                                     norm_layer, norm_kwargs)
        self.aspp_6 = _MCASPPConv((channels//4)*3, channels//2, channels, 6, 0.1,
                                     norm_layer, norm_kwargs)
        self.aspp_12 = _MCASPPConv((channels//8)*7, channels//2, channels, 12, 0.1,
                                      norm_layer, norm_kwargs)
        self.aspp_18 = _MCASPPConv((channels//16)*15, channels//2, channels, 18, 0.1,
                                      norm_layer, norm_kwargs)
        self.MDCM = MDCM()     # 每个ASPP中的MDCM共享权重

        self.conv1 = nn.Conv2d((channels//2)*3, (channels//4)*3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d((channels//4)*7, (channels//8)*7, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d((channels//8)*15, (channels//16)*15, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d((channels//16)*31, (channels//32)*31, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d((channels//32)*31, channels, kernel_size=3, stride=1, padding=1)
        self.convhsi = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, hsi, sar):
        aspp3 = self.aspp_3(sar)      # C/2 -> C
        hsi, aspp3 = self.MDCM(hsi, aspp3)  # C,C->C,C
        sar = torch.cat([aspp3, sar], dim=1)    # 3/2C
        sar = self.conv1(sar)    # 3/2C->3/4C
        hsi = self.convhsi(hsi)    # C

        aspp6 = self.aspp_6(sar)    # 3/4C -> C
        hsi, aspp6 = self.MDCM(hsi, aspp6)  # C,C->C,C
        sar = torch.cat([aspp6, sar], dim=1)    # 7/4C
        sar = self.conv2(sar)    # 7/4C->7/8C
        hsi = self.convhsi(hsi)    # C

        aspp12 = self.aspp_12(sar)  # 7/8C -> C
        hsi, aspp12 = self.MDCM(hsi, aspp12)
        sar = torch.cat([aspp12, sar], dim=1)   # 15/8C
        sar = self.conv3(sar)    # 15/8C->15/16C
        hsi = self.convhsi(hsi)    # C

        aspp18 = self.aspp_18(sar)  # 15/16C->C
        hsi, aspp18 = self.MDCM(hsi, aspp18)
        sar = torch.cat([aspp18, sar], dim=1)   # 31/16C
        sar = self.conv4(sar)    # 31/16C->31/32C
        hsi = self.convhsi(hsi)    # C

        sar = self.conv5(sar)  # 31/32C->C

        return hsi, sar
