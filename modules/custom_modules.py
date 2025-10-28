import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv, C3, SPPF # 导入YOLO的基础模块

# =======================================================
#               模块一: ECA (通道注意力)
# =======================================================
class ECA(nn.Module):
    """Efficient Channel Attention module"""
    def __init__(self, channels, k_size=None, gamma=2, b=1):
        super().__init__()
        if k_size is None:
            t = int(abs((math.log2(channels) + b) / gamma))
            k = t if t % 2 else t + 1
            k = max(3, k)
        else:
            k = k_size if k_size % 2 == 1 else k_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)
# =======================================================
#            模块二: CSAB_Offset (可变形注意力)
# =======================================================
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

# =======================================================
#      模块三: LCFANet Neck (MSASF + CLHFP)
# =======================================================
class LCFANeck(nn.Module):
    """
    LCFANet-inspired Neck to replace the entire YOLOv11 FPN/PAN.
    This module takes 3 feature maps from the backbone (P3, P4, P5) 
    and outputs 3 feature maps for the Detect head.
    """
    def __init__(self, c1, c2, c3): # Input channels from backbone P3, P4, P5
        super().__init__()
        # Determine internal and output channel counts based on YOLOv11n scale
        c_iv = [c1, c2, c3]
        c_out = [128, 256, 512] # Corresponds to YOLOv11n's Detect head inputs

        # --- MSASF Part (Multi-Scale Adaptive Spatial Fusion) ---
        self.msasf_projs = nn.ModuleList([nn.Conv2d(c, c_out[0], 1, bias=False) for c in c_iv])
        self.msasf_fusion = C3(c_out[0] * 3, c_out[0], n=2, shortcut=False)

        # --- CLHFP Part (Cross-Level Hierarchical Feature Pyramid) ---
        self.clhfp_up_p4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.clhfp_fusion_p3 = C3(c_out[0] + c_out[1], c_out[0], n=2, shortcut=False)
        
        # --- Down-sampling path to regenerate P4 and P5 for detection ---
        self.downsample_p4 = Conv(c_out[0], c_out[1], 3, 2)
        self.p4_fusion = C3(c_out[1] + c2, c_out[1], n=2, shortcut=False) # Fuse with original P4
        
        self.downsample_p5 = Conv(c_out[1], c_out[2], 3, 2)
        self.p5_fusion = C3(c_out[2] + c3, c_out[2], n=2, shortcut=False) # Fuse with original P5

    def forward(self, x):
        p3_in, p4_in, p5_in = x

        # 1. MSASF: Fuse all levels at the highest resolution (P3 level)
        p3_proj = self.msasf_projs[0](p3_in)
        p4_upsampled = F.interpolate(self.msasf_projs[1](p4_in), size=p3_proj.shape[-2:], mode='bilinear', align_corners=False)
        p5_upsampled = F.interpolate(self.msasf_projs[2](p5_in), size=p3_proj.shape[-2:], mode='bilinear', align_corners=False)
        msasf_out = self.msasf_fusion(torch.cat([p3_proj, p4_upsampled, p5_upsampled], 1))

        # 2. Top-down path with CLHFP-like fusion
        p3_out = self.clhfp_fusion_p3(torch.cat([msasf_out, self.clhfp_up_p4(p4_in)], 1))

        # 3. Bottom-up path to generate features for Detect head
        p4_out = self.p4_fusion(torch.cat([self.downsample_p4(p3_out), p4_in], 1))
        p5_out = self.p5_fusion(torch.cat([self.downsample_p5(p4_out), p5_in], 1))

        return [p3_out, p4_out, p5_out]

# (我们暂时不将DBL_TSD放在这里，因为它不是一个nn.Module，而是一个损失函数，将通过Trainer继承的方式注入)