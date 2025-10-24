# csab_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from csab_offset import CSAB_Offset

class AdaptiveSpatialFusion(nn.Module):
    """
    MSASF: Multi-Scale Adaptive Spatial Fusion
    对多层特征图自适应加权融合（轻量版）
    """
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1, bias=False) for c in in_channels_list
        ])
        self.spatial_att = nn.Sequential(
            nn.Conv2d(len(in_channels_list), 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.out_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)

    def forward(self, feats):
        # feats: list of multi-scale feature maps [P3, P4, P5]
        target_size = feats[0].shape[-2:]
        upsampled = [F.interpolate(self.proj[i](f), size=target_size, mode='bilinear', align_corners=False)
                     for i, f in enumerate(feats)]
        stacked = torch.stack(upsampled, dim=1)  # [N, n_scales, C, H, W]
        weight_map = self.spatial_att(stacked.mean(2))  # [N,1,H,W]
        fused = (stacked.mean(1) * weight_map + stacked.max(1).values * (1 - weight_map))
        return self.out_conv(fused)


class CrossLevelFeaturePyramid(nn.Module):
    """
    CLHFP: Cross-Level Hierarchical Feature Pyramid (简化版)
    跨层特征金字塔融合模块。
    """
    def __init__(self, channels=256):
        super().__init__()
        self.up = nn.ConvTranspose2d(channels, channels, 2, stride=2)
        self.down = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        self.mix = nn.Conv2d(channels, channels, 3, padding=1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, f_low, f_high):
        # f_low: 低层特征（高分辨率）
        # f_high: 高层特征（低分辨率）
        up_feat = F.interpolate(f_high, size=f_low.shape[-2:], mode='bilinear', align_corners=False)
        mix_feat = f_low + self.up(f_high) + self.mix(f_low)
        gate = self.gate(mix_feat)
        return gate * mix_feat + (1 - gate) * f_low


class CSAB_FusionBlock(nn.Module):
    """
    整合 CSAB + Offset + MSASF + CLHFP 的混合模块
    """
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.csab = CSAB_Offset(out_channels)
        self.msasf = AdaptiveSpatialFusion(in_channels_list, out_channels)
        self.clhfp = CrossLevelFeaturePyramid(out_channels)
        self.conv_out = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, feats):
        """
        feats: list [P3, P4, P5]
        """
        fused = self.msasf(feats)
        fused = self.clhfp(fused, feats[-1])  # 使用最高层语义增强
        fused = self.csab(fused)
        return self.conv_out(fused)
