#!/usr/bin/env python3
"""
检查新训练的Llama LoRA权重格式
"""
import torch
from safetensors import safe_open

def check_llama_lora_format():
    """检查Llama LoRA权重格式"""
    llama_lora_path = "/root/PAW/train_lora/runs/Llama-3.2-3B-Instruct/arc-challenge_lora_20250724_140508/final_model/adapter_model.safetensors"
    
    print("🔍 检查新训练的Llama LoRA权重格式:")
    print(f"文件路径: {llama_lora_path}")
    
    with safe_open(llama_lora_path, framework="pt", device="cpu") as f:
        print(f"\n总权重数: {len(f.keys())}")
        
        # 显示前15个权重
        for i, key in enumerate(f.keys()):
            if i >= 15:
                print("...")
                break
            weight = f.get_tensor(key)
            print(f"  {key}: {weight.shape}")
        
        # 检查特定层的A和B权重
        print(f"\n🎯 关键层权重检查:")
        for key in f.keys():
            if "layers.0.self_attn.k_proj" in key:
                weight = f.get_tensor(key)
                print(f"  {key}: {weight.shape}")

if __name__ == "__main__":
    check_llama_lora_format()
