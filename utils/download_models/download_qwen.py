import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def download_qwen(base_path="./models"):
    # 设置模型名称
    model_name = "Qwen/Qwen2.5-0.5B"
    model_version = model_name.replace('/', '-')
    
    # 创建具体版本的保存目录
    save_path = os.path.join(base_path, model_version)
    os.makedirs(save_path, exist_ok=True)
    
    # 下载tokenizer
    print("正在下载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(save_path)
    
    # 下载模型
    print("正在下载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True
    )
    model.save_pretrained(save_path)
    
    print(f"模型和tokenizer已保存到: {save_path}")
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = download_qwen()