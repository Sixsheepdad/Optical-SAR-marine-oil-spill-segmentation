import torch
import torch.nn as nn
import math
import torch.nn.functional as F

#-----------------CoordAttention
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, in_channels, reduction=8):     # r控制中间通道数
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class DLEM(nn.Module):
    def __init__(self, in_channels, direction='horizontal', theta1_init=0.3, theta2_init=0.7, reduction=8):     #原0.3  0.7
        super().__init__()
        self.coord_att_vanilla = CoordAtt(in_channels=in_channels, reduction=reduction)
        self.coord_att_dlem = CoordAtt(in_channels=in_channels, reduction=reduction)
        self.direction = direction
        self.theta1 = nn.Parameter(torch.tensor(theta1_init))
        self.theta2 = nn.Parameter(torch.tensor(theta2_init))

        self.vanilla_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        # 初始化差分核：并行 group 卷积
        self._init_diff_kernel(in_channels, direction)

    def _init_diff_kernel(self, in_channels, direction):
        kernel = torch.zeros(3, 3)
        if direction == 'horizontal':
            kernel[1, :] = torch.tensor([1., 0., -1.])
        elif direction == 'vertical':
            kernel[:, 1] = torch.tensor([1., 0., -1.])
        elif direction == 'diagonal1':
            kernel[0, 0] = 1.0
            kernel[2, 2] = -1.0
            kernel[0, 2] = -1.0
            kernel[2, 0] = 1.0
        elif direction == 'diagonal2':
            kernel[0, 2] = 1.0
            kernel[2, 0] = -1.0
            kernel[0, 0] = -1.0
            kernel[2, 2] = 1.0
        else:
            raise ValueError("Unsupported direction")

        # 将 (3,3) 卷积核复制为 in_channels 个，作为 group 卷积核
        weight = kernel.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)
        self.diff_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.diff_conv.weight.data = weight
        self.diff_conv.weight.requires_grad_(False)

    def forward(self, x):
        vanilla_out = self.coord_att_vanilla(self.vanilla_conv(x))
        diff_out = self.coord_att_dlem(self.diff_conv(x))
        return self.theta1 * vanilla_out + self.theta2 * diff_out


#-----------------------------------------------------------------------
if __name__ == '__main__':
    # # 设置参数
    # B, C, H, W = 8, 128, 256, 256
    # inp = C
    #
    # # 创建随机输入张量
    # x = torch.randn(B, C, H, W)
    # # 创建 CoordAtt 实例
    # model = DLEM(in_channels=inp, direction='diagonal1', theta1_init=0.3, theta2_init=0.7, reduction=32)
    # #dlem = DLEM(in_channels=inp, direction='diagonal1', theta1_init=0.3, theta2_init=0.7)
    # # 前向传播
    # output = model(x)
    # # 打印输出形状
    # print("Input shape:", x.shape)
    # print("Output shape:", output.shape)
    #
    # # #--------------------
    # # x = torch.randn(8, 32, 256, 256)


    # 准备输入和模型（放到 GPU）
    x = torch.randn(8, 128, 256, 256).cuda()
    model = DLEM(in_channels=128, direction='diagonal1').cuda()
    model.eval()

    # 计时事件
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # 多次测量以求平均
    with torch.no_grad():
        timings = []
        for _ in range(10):
            starter.record()
            output = model(x)
            ender.record()
            torch.cuda.synchronize()  # 等待所有CUDA流完成
            timings.append(starter.elapsed_time(ender))  # 单位：毫秒

    avg_time = sum(timings) / len(timings)
    print(f"Average forward pass time (GPU): {avg_time:.3f} ms")



