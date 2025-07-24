#!/usr/bin/env python3
"""
检查原始Qwen LoRA的权重形状
"""
import torch
from safetensors import safe_open

def check_original_lora():
    """检查原始Qwen LoRA权重"""
    original_lora = "/root/PAW/train_lora/runs/Qwen_Qwen2.5-1.5B/arc-challenge_lora_20250724_014727/final_model/adapter_model.safetensors"
    
    print("原始Qwen LoRA权重形状:")
    with safe_open(original_lora, framework="pt", device="cpu") as f:
        for i, key in enumerate(f.keys()):
            if i >= 10:
                print("...")
                break
            weight = f.get_tensor(key)
            print(f"  {key}: {weight.shape}")

if __name__ == "__main__":
    check_original_lora()
