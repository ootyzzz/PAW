"""
flatten.py
将 LoRA 权重展平成一维向量
"""
import torch

def flatten_lora_weights(lora_model):
    """
    展平 LoRA 权重为一维向量
    """
    return torch.cat([p.flatten() for p in lora_model.parameters()])
