"""
checkpoint_utils.py
LoRA checkpoint 的保存、加载与枚举
"""
import torch
import os

def save_checkpoint(model, path):
    """
    保存模型 checkpoint
    """
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path):
    """
    加载模型 checkpoint
    """
    model.load_state_dict(torch.load(path))
    return model

def list_checkpoints(folder):
    """
    枚举文件夹下所有 .pt checkpoint 路径
    """
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.pt')]
