#!/usr/bin/env python3
"""
检查迁移后的LoRA权重形状
"""
import torch
from safetensors import safe_open
import sys

def check_lora_weights(lora_path):
    """检查LoRA权重的形状"""
    print(f"检查LoRA权重: {lora_path}")
    
    weights_file = f"{lora_path}/adapter_model.safetensors"
    
    with safe_open(weights_file, framework="pt", device="cpu") as f:
        print(f"\n总共有 {len(f.keys())} 个权重:")
        
        for i, key in enumerate(f.keys()):
            if i >= 10:  # 只显示前10个
                print("...")
                break
            weight = f.get_tensor(key)
            print(f"  {key}: {weight.shape}")
    
    # 检查特定的问题权重
    print(f"\n检查特定层权重...")
    with safe_open(weights_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            if "layers.0.mlp.down_proj" in key:
                weight = f.get_tensor(key)
                print(f"  {key}: {weight.shape}")
            if "layers.0.mlp.up_proj" in key:
                weight = f.get_tensor(key)
                print(f"  {key}: {weight.shape}")

if __name__ == "__main__":
    lora_path = "/root/autodl-tmp/trained_t2l/qwen_to_llama_lora_x"
    check_lora_weights(lora_path)
