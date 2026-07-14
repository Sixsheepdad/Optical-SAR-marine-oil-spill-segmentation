import torch
import torch.nn as nn
from BLOCK.DLEM import DLEM
from BLOCK.SS3DConv import SS3DConv
from BLOCK.MCASPP import MCASPP

class OilDetNet(nn.Module):   # HSI_channel=20, sar_channel=1
    def __init__(self, HSI_channel, SAR_channel):
        super(OilDetNet, self).__init__()
        self.HSIEX = SS3DConv(in_channels=1)
        self.convhsi1 = nn.Conv2d(HSI_channel, 32, kernel_size=3, stride=1, padding=1)
        self.convhsi2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.convhsi3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.convhsi4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        self.convsar1 = nn.Conv2d(SAR_channel, 16, kernel_size=3, stride=1, padding=1)
        self.convsar2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.convsar3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.convfusion1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.convfusion2 = nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.convfusion3 = nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1)

        self.MCASPPUltra1 = MCASPP(channels=64)      # 需要sar为32  hsi为64
        self.MCASPPUltra2 = MCASPP(channels=256)      # 需要sar为128  hsi为256

        self.dlemsar1 = DLEM(in_channels=32, direction='horizontal', theta1_init=0.3, theta2_init=0.7, reduction=16)   # add
        self.dlemsar2 = DLEM(in_channels=32, direction='vertical', theta1_init=0.3, theta2_init=0.7, reduction=16)     # add
        self.dlemhsi1 = DLEM(in_channels=64, direction='horizontal', theta1_init=0.3, theta2_init=0.7, reduction=16)   # add
        self.dlemhsi2 = DLEM(in_channels=64, direction='vertical', theta1_init=0.3, theta2_init=0.7, reduction=16)     # add

        self.dlem1 = DLEM(in_channels=64, direction='diagonal1', theta1_init=0.3, theta2_init=0.7, reduction=16)       # Fhsi1使用
        self.dlem2 = DLEM(in_channels=64, direction='diagonal1', theta1_init=0.3, theta2_init=0.7, reduction=16)       # Fsar1使用
        self.dlem3 = DLEM(in_channels=128, direction='diagonal1', theta1_init=0.3, theta2_init=0.7, reduction=16)      # Ff1使用
        self.dlem4 = DLEM(in_channels=512, direction='diagonal2', theta1_init=0.3, theta2_init=0.7, reduction=16)      # Ff2使用

        self.conv1x1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        #---------------
        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        hsi_data = x[:, :20, :, :]  # 取前 20 个通道作为 HSI 数据    [B,20,256,256]
        sar_data = x[:, 20:, :, :]  # 取最后 1 个通道作为 SAR 数据    [B,1,256,256]

        # hsi_data = self.convhsi1(hsi_data)  # [B,32,256,256]
        # hsi_data = self.convhsi2(hsi_data)  # [B,64,256,256]
        hsi_data = self.HSIEX(hsi_data) # [B,64,256,256]    提取光谱空间信息
        # hsi_data = self.dlemhsi1(hsi_data)  # [B,64,256,256]    #提取水平边缘信息   # add
        # hsi_data = self.dlemhsi2(hsi_data)  # [B,64,256,256]    #提取垂直边缘信息   # add

        sar_data = self.convsar1(sar_data)  # [B,16,256,256]
        sar_data = self.convsar2(sar_data)  # [B,32,256,256]
        # sar_data = self.dlemsar1(sar_data)  # [B,32,256,256]    #提取水平边缘信息   # add
        # sar_data = self.dlemsar2(sar_data)  # [B,32,256,256]    #提取垂直边缘信息   # add

        hsi_data, sar_data = self.MCASPPUltra1(hsi_data, sar_data)  # [B,64,256,256], [B,64,256,256]  提取多尺度互补信息
        # ----------------ADD-----------------------------------------------------------------------------------------------------------
        Fusion_data1 = torch.cat([hsi_data, sar_data], dim=1)  # [B,128,256,256]        #转换为融合
        hsi_data = self.dlem1(hsi_data)      # [B,64,256,256]   #提取对角线1信息
        sar_data = self.dlem2(sar_data)     # [B,64,256,256]    #提取对角线1信息
        Fusion_data1 = self.convfusion2(self.convfusion1(self.dlem3(Fusion_data1)))  # [B,256,64,64]   提取对角线1信息
        hsi_data = self.convhsi4(self.convhsi3(hsi_data))   # [B,256,128,128]
        sar_data = self.convsar3(sar_data)  # [B,128,128,128]
        hsi_data, sar_data = self.MCASPPUltra2(hsi_data, sar_data)   # [B,256,128,128], [B,256,128,128]
        Fusion_data2 = torch.cat([hsi_data, sar_data], dim=1)  # [B,512,128,128]        #转换为融合
        Fusion_data2 = self.dlem4(Fusion_data2)     # [B,512,128,128]
        Fusion_data2 = self.convfusion3(Fusion_data2)   # [B,256,64,64]

        Fusion_data = Fusion_data1 + Fusion_data2   # [B,256,64,64]
        # Fusion_data = self.conv1x1(Fusion_data)         # [B,256,64,64]     # add

        # -----------------------------------------------------------------------------------------------------------------------------------
        Fusion_data = self.conv1(Fusion_data)  # [B,128,128,128]
        Fusion_data = self.conv2(Fusion_data)  # [B,64,128,128]
        Fusion_data = self.conv3(Fusion_data)  # [B,32,256,256]
        output = self.conv4(Fusion_data)  # [B,1,256,256]

        return output


if __name__ == "__main__":
    model = OilDetNet(HSI_channel=20, SAR_channel=1)
    dummy_input = torch.randn(8, 21, 256, 256)
    output = model(dummy_input)
    # print("输出hsi尺寸:", hsi.shape)  # [8, 64, 256, 256]
    # print("输出sar尺寸:", sar.shape)  # [8, 64, 256, 256]
    print("输出尺寸:", output.shape)  # [8, 1, 256, 256]



