#!/usr/bin/env python3
"""
配置管理模块
包含配置创建、批次大小优化、实验配置等功能
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from .data import get_test_file_path


def get_optimal_batch_size(dataset_name: str, target_steps: int = 125) -> int:
    """根据数据集大小自动选择最优batch size"""
    dataset_sizes = {
        'arc-challenge': 1119,
        'arc-easy': 2251, 
        'boolq': 9427,
        'hellaswag': 39905,
        'openbookqa': 4957,
        'piqa': 16113,
        'winogrande': 40398
    }
    
    if dataset_name not in dataset_sizes:
        print(f"⚠️ 未知数据集 {dataset_name}，使用默认batch_size=32")
        return 32
    
    dataset_size = dataset_sizes[dataset_name]
    
    # 选择合适的batch size
    if dataset_size <= 1200:  # 小数据集
        return 8
    elif dataset_size <= 2500:  # 中小数据集  
        return 16
    else:  # 大数据集
        return 32


def analyze_batch_efficiency(dataset_name: str, batch_size: int, target_steps: int = 125):
    """分析batch size效率"""
    dataset_sizes = {
        'arc-challenge': 1119,
        'arc-easy': 2251, 
        'boolq': 9427,
        'hellaswag': 39905,
        'openbookqa': 4957,
        'piqa': 16113,
        'winogrande': 40398
    }
    
    if dataset_name not in dataset_sizes:
        return
        
    dataset_size = dataset_sizes[dataset_name]
    total_samples_needed = batch_size * target_steps
    epochs_needed = total_samples_needed / dataset_size
    
    print(f"📊 Batch效率分析:")
    print(f"  - 数据集: {dataset_name} ({dataset_size} 样本)")
    print(f"  - Batch大小: {batch_size}")
    print(f"  - {target_steps}步需要: {total_samples_needed} 样本")
    print(f"  - 需要循环: {epochs_needed:.2f} epochs")
    
    return epochs_needed


def create_lightning_config(
    dataset_name: str, 
    base_config: Dict[str, Any], 
    base_model_path: str = None, 
    batch_size: int = None, 
    max_steps: int = 125, 
    save_steps: int = 50, 
    learning_rate: float = 1e-4, 
    learning_rate_stage2: float = None,
    external_config_path: str = None
) -> Dict[str, Any]:
    """创建Lightning训练配置"""
    config = base_config.copy()
    
    # 如果提供了外部配置文件，读取并合并LoRA配置
    if external_config_path and os.path.exists(external_config_path):
        import yaml
        try:
            with open(external_config_path, 'r', encoding='utf-8') as f:
                external_config = yaml.safe_load(f)
            
            # 合并LoRA配置
            if 'lora' in external_config:
                config['lora'] = external_config['lora']
                print(f"📝 从外部配置文件读取LoRA设置: {external_config_path}")
                
                # 显示LoRA配置信息
                lora_config = external_config['lora']
                if 'target_modules' in lora_config:
                    print(f"   - 目标层: {lora_config['target_modules']}")
                if 'r' in lora_config:
                    print(f"   - 秩 (r): {lora_config['r']}")
                if 'lora_alpha' in lora_config:
                    print(f"   - Alpha: {lora_config['lora_alpha']}")
                if 'lora_dropout' in lora_config:
                    print(f"   - Dropout: {lora_config['lora_dropout']}")
                    
        except Exception as e:
            print(f"⚠️ 读取外部配置文件失败: {e}")
            print(f"   继续使用默认LoRA配置")
    
    # 自动选择最优batch size（如果未指定）
    if batch_size is None:
        batch_size = get_optimal_batch_size(dataset_name)
        print(f"🎯 自动选择batch_size={batch_size}用于{dataset_name}")
    
    # 分析batch效率
    analyze_batch_efficiency(dataset_name, batch_size, max_steps)
    
    # 智能选择测试文件路径 - 支持validation作为fallback
    test_file_path, using_validation = get_test_file_path(dataset_name)
    
    # 确保config有所需的sections
    if 'data' not in config:
        config['data'] = {}
    if 'training' not in config:
        config['training'] = {}
    if 'paths' not in config:
        config['paths'] = {}
    
    # 更新数据路径
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config['data']['train_file'] = os.path.join(project_root, f"data_to_lora/cs/{dataset_name}/{dataset_name}_train_formatted.jsonl")
    config['data']['test_file'] = os.path.join(project_root, test_file_path) if not os.path.isabs(test_file_path) else test_file_path
    config['data']['using_validation_as_test'] = using_validation
    
    # 从模型路径提取模型名称用于目录组织
    if base_model_path:
        model_name = Path(base_model_path).name
        # 处理 HuggingFace 格式的名称 (如 Qwen/Qwen2.5-1.5B -> Qwen2.5-1.5B)
        if '/' in model_name and not os.path.exists(base_model_path):
            model_name = model_name.split('/')[-1]
    else:
        model_name = "default-model"
    
    # 生成实验名称 - 使用更规范的命名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{dataset_name}_lora_{timestamp}"
    
    # Lightning + SwanLab 专用配置
    stage1_ratio = 0.6  # 60% 步数为Stage 1
    stage1_steps = int(max_steps * stage1_ratio)
    stage2_steps = max_steps - stage1_steps
    
    # 自动计算第二阶段学习率
    if learning_rate_stage2 is None:
        learning_rate_stage2 = learning_rate / 10
    
    config['training'].update({
        'batch_size': batch_size,
        'max_steps': max_steps,  # 使用参数化的步数
        'stage1_steps': stage1_steps,
        'stage2_steps': stage2_steps,
        'save_steps': save_steps,  # 保存最后多少步
        'learning_rate_stage1': learning_rate,
        'learning_rate_stage2': learning_rate_stage2,
    })
    
    # 现代化的输出目录结构 - 按数据集分组，再按模型分组
    base_dir = Path("./runs") / dataset_name / model_name / experiment_name.split('_')[-1]  # 只使用时间戳部分
    config['paths'].update({
        'experiment_dir': str(base_dir),
        'checkpoints_dir': str(base_dir / "checkpoints"),
        'tensorboard_dir': str(base_dir / "tensorboard_logs"),
        'swanlab_dir': str(base_dir / "swanlab_logs"),
        'final_model_dir': str(base_dir / "final_model"),
        'config_file': str(base_dir / "config.yaml")
    })
    
    # 实验元数据
    config['experiment'] = {
        'name': experiment_name,
        'dataset': dataset_name,
        'batch_size': batch_size,
        'model_name': model_name,  # 添加模型名称到元数据
        'base_model_path': base_model_path,  # 添加完整模型路径
        'framework': 'lightning_swanlab',
        'created_at': datetime.now().isoformat(),
        'description': f"Lightning LoRA training on {dataset_name} with {model_name} - {max_steps} steps",
        'tags': ["lightning", "swanlab", "lora", model_name.lower(), dataset_name, f"batch{batch_size}", f"steps{max_steps}"]
    }
    
    return config
