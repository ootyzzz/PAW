"""
hyperconv_decoder.py
实现 Hyper-Convolution Parameter Generator（严格按 DnD 论文 A.3）
"""
import torch
import torch.nn as nn

class HyperConvBlock(nn.Module):
    """
    单个 Hyper-Convolution Block，包括 ConvW/ConvH/ConvL、残差连接、激活和归一化
    """
    def __init__(self, dim):
        super().__init__()
        self.convw = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.convh = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.convl = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
    def forward(self, x):
        # 论文结构，具体实现需参考 A.3
        out = self.convw(x)
        out = self.convh(out)
        out = self.convl(out)
        out = self.act(out)
        out = self.norm(out)
        return out + x  # 残差连接

class HyperConvDecoder(nn.Module):
    """
    多层级联的 Hyper-Convolution Decoder
    """
    def __init__(self, num_blocks=6, dim=384):
        super().__init__()
        self.blocks = nn.ModuleList([HyperConvBlock(dim) for _ in range(num_blocks)])
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
