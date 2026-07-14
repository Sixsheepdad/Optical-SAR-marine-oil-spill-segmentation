import torch
import torch.nn as nn
import torch.nn.functional as F

class SS3DConv(nn.Module):
    def __init__(self, in_channels=1):
        super(SS3DConv, self).__init__()
        # 3D卷积特征提取层
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=4, kernel_size=(7, 3, 3), padding=(3, 1, 1)),
            nn.BatchNorm3d(4),
            nn.ReLU(),
            nn.Conv3d(in_channels=4, out_channels=8, kernel_size=(5, 3, 3), padding=(2, 1, 1), stride=(1, 2, 2)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU()
        )

        # 2D卷积特征提取层
        # 这里需要根据你的3D卷积输出调整输入通道数
        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=128, kernel_size=(3, 3), padding=1),  # 2D卷积层
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=1),  # 2D卷积层
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2),  # 2D卷积层
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 20, 256, 256]->[B, 1, 20, 256, 256]
        # 3D 卷积层提取空间光谱特征
        x = self.conv3d_features(x)     # [B,16,20,128,128]
        x = x.view(x.size()[0], x.size()[1] * x.size()[2], x.size()[3], x.size()[4])    #[B,320,128,128]
        # 2D 卷积层进一步处理空间特征
        x = self.conv2d_features(x)  # [B,320,125,128]->[B,64,256,256]
        return x
