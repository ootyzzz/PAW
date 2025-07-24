#!/usr/bin/env python3
"""
使用最佳配置训练Llama-3.2-3B LoRA
复制Qwen最佳训练配置: steps=600, batch_size=6, lr=1.5e-5
"""

import os
import sys
import json
import yaml
from datetime import datetime
from pathlib import Path

# 添加src路径
sys.path.append('/root/PAW/train_lora/src')

from lightning_trainer import LightningLoRATrainer

def create_llama_lora_config():
    """创建Llama LoRA训练配置"""
    
    # 基于Qwen最佳配置
    config = {
        'data': {
            'drop_last': False,
            'max_length': 512,
            'shuffle': False,
            'test_file': '/root/PAW/data_to_lora/cs/arc-challenge/arc-challenge_test_formatted.jsonl',
            'train_file': '/root/PAW/data_to_lora/cs/arc-challenge/arc-challenge_train_formatted.jsonl',
            'using_validation_as_test': False
        },
        'experiment': {
            'base_model_path': '/root/autodl-tmp/models/Llama-3.2-3B-Instruct',
            'batch_size': 6,
            'created_at': datetime.now().isoformat(),
            'dataset': 'arc-challenge',
            'description': 'Lightning LoRA training on arc-challenge with Llama-3.2-3B-Instruct - 600 steps',
            'framework': 'lightning_swanlab',
            'model_name': 'Llama-3.2-3B-Instruct',
            'name': f'arc-challenge_lora_llama32_3b_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'tags': [
                'lightning',
                'swanlab', 
                'lora',
                'llama-3.2-3b-instruct',
                'arc-challenge',
                'batch6',
                'steps600'
            ]
        },
        'lightning': {
            'accelerator': 'auto',
            'devices': 'auto',
            'enable_progress_bar': True,
            'log_every_n_steps': 1,
            'mode': 'min',
            'monitor': 'train/loss',
            'precision': '16-mixed',
            'save_top_k': -1,
            'strategy': 'auto'
        },
        'lora': {
            'alpha': 32,
            'bias': 'none',
            'dropout': 0.1,
            'r': 16,
            'target_modules': [
                'q_proj',
                'v_proj', 
                'k_proj',
                'o_proj',
                'gate_proj',
                'up_proj',
                'down_proj'
            ],
            'task_type': 'CAUSAL_LM'
        },
        'model': {
            'cache_dir': './models',
            'name': '/root/autodl-tmp/models/Llama-3.2-3B-Instruct',
            'path': '/root/autodl-tmp/models/Llama-3.2-3B-Instruct',
            'torch_dtype': 'auto',
            'trust_remote_code': True
        },
        'swanlab': {
            'project': 'lora-training',
            'tags': [
                'lightning',
                'lora',
                'llama-3.2'
            ]
        },
        'training': {
            'batch_size': 6,
            'betas': [0.9, 0.999],
            'eps': 1e-08,
            'gradient_accumulation_steps': 1,
            'learning_rate_stage1': 1.5e-05,
            'learning_rate_stage2': 1.5e-05,
            'max_steps': 600,
            'optimizer': 'adamw',
            'save_steps': 1,
            'stage1_steps': 360,
            'stage2_steps': 240,
            'weight_decay': 0.01
        }
    }
    
    return config

def main():
    print("🚀 开始训练Llama-3.2-3B LoRA (复制Qwen最佳配置)")
    
    # 创建配置
    config = create_llama_lora_config()
    print(f"📝 实验名称: {config['experiment']['name']}")
    
    # 创建trainer并开始训练
    trainer = LightningLoRATrainer(config)
    
    print("🔥 开始训练...")
    trainer.train()
    
    print("✅ 训练完成!")
    print(f"📁 模型保存路径: {trainer.final_model_dir}")

if __name__ == "__main__":
    main()
