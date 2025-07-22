"""
hyperconv_decoder.py
实现 Hyper-Convolution Parameter Generator
基于论文 Figure 3 的结构，修复维度问题
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperConvBlock(nn.Module):
    """
    单个 Hyper-Convolution Block
    修复版本：所有卷积操作在同一维度空间
    """
    def __init__(self, dim):
        super().__init__()
        # 修复：全部使用Conv1d，处理序列数据 [B, dim, seq_len]
        self.convw = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1, groups=1)
        self.convh = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1, groups=1)  
        self.convl = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1, groups=1)
        
        # 归一化和激活
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        """
        x: [B, dim, seq_len] 格式的输入
        """
        residual = x
        
        # 级联的三个卷积操作：ConvW -> ConvH -> ConvL
        out = self.convw(x)    # [B, dim, seq_len]
        out = self.convh(out)  # [B, dim, seq_len]  
        out = self.convl(out)  # [B, dim, seq_len]
        
        # 调整维度用于LayerNorm: [B, seq_len, dim]
        out = out.transpose(1, 2)  
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)
        out = out.transpose(1, 2)  # 回到 [B, dim, seq_len]
        
        # 残差连接
        return out + residual

class HyperConvDecoder(nn.Module):
    """
    3层级联的 Hyper-Convolution Decoder
    用于生成 LoRA 参数的 tokenized 表示
    """
    def __init__(self, num_blocks=3, dim=384, max_seq_len=512, output_dim=None):
        super().__init__()
        self.num_blocks = num_blocks
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.output_dim = output_dim or dim
        
        # 3个HyperConv块级联
        self.blocks = nn.ModuleList([
            HyperConvBlock(dim) for _ in range(num_blocks)
        ])
        
        # 输出投影层，将内部表示映射到目标维度
        self.output_proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, self.output_dim)
        )
        
        # 位置编码（可选）
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, dim) * 0.02
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, seq_len, dim] 或 [B, dim, seq_len]
        Returns:
            output: [B, seq_len, output_dim] - 生成的参数token表示
        """
        batch_size = x.shape[0]
        
        # 统一输入格式到 [B, dim, seq_len]
        if x.dim() == 3 and x.size(-1) == self.dim:
            x = x.transpose(1, 2)  # [B, seq_len, dim] -> [B, dim, seq_len]
        
        # 确保序列长度
        if x.shape[2] != self.max_seq_len:
            x = F.interpolate(x, size=self.max_seq_len, mode='linear', align_corners=False)
        
        # 添加位置编码
        pos_emb = self.pos_embedding[:, :self.max_seq_len, :].transpose(1, 2)  # [1, dim, seq_len]
        x = x + pos_emb
        
        # 通过HyperConv块
        for i, block in enumerate(self.blocks):
            x = block(x)
        
        # 转换为最终输出格式 [B, seq_len, output_dim]
        x = x.transpose(1, 2)  # [B, dim, seq_len] -> [B, seq_len, dim]
        x = self.output_proj(x)  # [B, seq_len, output_dim]
        
        return x
