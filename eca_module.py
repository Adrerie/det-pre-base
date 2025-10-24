# eca_module.py
import torch
import torch.nn as nn
import math

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
