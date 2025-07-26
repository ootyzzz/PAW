#!/usr/bin/env python3
"""
检查Llama-3.2-3B基础模型的权重形状
"""
import torch
from transformers import AutoModelForCausalLM

def check_llama_weights():
    """检查Llama基础模型权重"""
    model_path = "../../../autodl-tmp/models/Llama-3.2-3B-Instruct"
    
    print("加载Llama-3.2-3B-Instruct...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True
    )
    
    print("\nLlama模型关键层权重形状:")
    for name, param in model.named_parameters():
        if "layers.0" in name and any(x in name for x in ["k_proj", "v_proj", "down_proj", "up_proj"]):
            print(f"  {name}: {param.shape}")

if __name__ == "__main__":
    check_llama_weights()
