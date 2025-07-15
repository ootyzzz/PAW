"""
metrics.py
定义损失函数（MSE）
"""
import torch

def mse_loss(pred, target):
    """
    计算均方误差损失
    """
    return torch.nn.functional.mse_loss(pred, target)
