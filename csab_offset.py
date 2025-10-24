# csab_offset
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvOffset2D(nn.Module):
    """
    轻量Deformable卷积模块（不依赖DCNv3）
    通过offset网络学习每个位置的采样偏移。
    """
    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.offset_conv = nn.Conv2d(channels, 2 * kernel_size * kernel_size, 3, padding=1)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.zero_pad = nn.ZeroPad2d(padding)
        self.weight = nn.Parameter(torch.randn(channels, channels, kernel_size, kernel_size) * 0.01)
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        offset = self.offset_conv(x)
        N, _, H, W = x.shape
        k = self.kernel_size
        # 构建标准卷积采样网格
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=x.device),
            torch.arange(W, device=x.device)
        )
        grid = torch.stack((grid_x, grid_y), dim=0).float()  # (2, H, W)
        grid = grid.unsqueeze(0).repeat(N, 1, 1, 1)  # (N, 2, H, W)
        # 应用偏移
        offset = offset.view(N, 2, k*k, H, W).mean(2)
        grid = grid + offset
        grid[:, 0] = 2.0 * grid[:, 0] / max(W - 1, 1) - 1.0
        grid[:, 1] = 2.0 * grid[:, 1] / max(H - 1, 1) - 1.0
        grid = grid.permute(0, 2, 3, 1)
        x_deform = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        out = F.conv2d(x_deform, self.weight, self.bias, stride=self.stride, padding=self.padding)
        return out


class CSAB_Offset(nn.Module):
    """
    Channel-Spatial Attention Block with Deformable Sampling
    """
    def __init__(self, channels, reduction=16, kernel_size=3):
        super().__init__()
        self.offset_conv = ConvOffset2D(channels, kernel_size)
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        # 空间注意力
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. 变形采样特征
        x_offset = self.offset_conv(x)
        # 2. 通道注意力
        avg_out = self.fc(self.avg_pool(x_offset))
        max_out = self.fc(self.max_pool(x_offset))
        ca = self.sigmoid(avg_out + max_out)
        x_ca = x_offset * ca
        # 3. 空间注意力
        avg_pool = torch.mean(x_ca, dim=1, keepdim=True)
        max_pool, _ = torch.max(x_ca, dim=1, keepdim=True)
        sa = self.sigmoid(self.conv_spatial(torch.cat([avg_pool, max_pool], dim=1)))
        x_sa = x_ca * sa
        return x_sa + x  # 残差连接
