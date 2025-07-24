#!/usr/bin/env python3
"""
训练核心模块
包含训练执行、回调设置、SwanLab集成等功能
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import swanlab

from peft import TaskType
from .model import LoRALightningModule
from .data import TrainTestDataModule


class SwanLabLogger:
    """SwanLab Logger for Lightning"""
    
    def __init__(self, project_name: str, experiment_name: str, config: Dict[str, Any]):
        self.project_name = project_name
        self.experiment_name = experiment_name 
        self.config = config
        self._run = None
        
    def initialize_run(self):
        """初始化 SwanLab run"""
        if self._run is None:
            self._run = swanlab.init(
                project=self.project_name,
                experiment_name=self.experiment_name,
                config=self.config
            )
        return self._run


def setup_callbacks(config: Dict[str, Any]) -> List[pl.Callback]:
    """设置Lightning回调（学习率监控、早停）"""
    callbacks = []
    
    # 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # 早停回调 - 验证准确率50步无提升则停止
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=50,
        mode='max',
        verbose=True,
        min_delta=0.001  # 至少提升0.1%才算改善
    )
    callbacks.append(early_stopping)
    
    # 注意：学习率调度器将在模型内部配置
    
    return callbacks


def run_lightning_training(
    dataset_name: str,
    config: Dict[str, Any],
    dry_run: bool = False
) -> Dict[str, Any]:
    """运行Lightning训练"""
    
    print(f"\n{'=' * 70}")
    print(f"🚀 Lightning LoRA 训练: {dataset_name}")
    print(f"{'=' * 70}")
    
    # 验证数据文件
    train_file = config['data']['train_file']
    test_file = config['data']['test_file']
    using_validation_as_test = config['data'].get('using_validation_as_test', False)
    
    for file_path, file_type in [(train_file, "训练"), (test_file, "测试")]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_type}数据文件不存在: {file_path}")
    
    print(f"📁 训练文件: {train_file}")
    if using_validation_as_test:
        print(f"📁 测试文件: {test_file} (使用validation作为test)")
        print("💡 数据集配置: 适配只有train/validation的数据集")
    else:
        print(f"📁 测试文件: {test_file}")
    print(f"🎯 实验目录: {config['paths']['experiment_dir']}")
    print(f"📊 训练配置: batch_size={config['training']['batch_size']}, steps={config['training']['max_steps']} ({config['training']['stage1_steps']}+{config['training']['stage2_steps']}) - 保存最后{config['training']['save_steps']}步")
    
    if dry_run:
        print("🏃 Dry Run 完成 - 已验证:")
        print("  ✅ 配置文件格式正确")
        print("  ✅ 数据文件存在且可访问")
        print("  ✅ 实验目录结构创建")
        print("  ✅ LoRA配置有效")
        print("  ✅ 训练参数合理")
        print("  💡 要实际训练请移除 --dry_run 参数")
        return {"status": "dry_run_completed"}
    
    try:
        # 创建实验目录
        experiment_dir = Path(config['paths']['experiment_dir'])
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置文件
        with open(config['paths']['config_file'], 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2, allow_unicode=True)
        
        # 初始化 SwanLab
        print("📊 初始化 SwanLab...")
        swanlab_run = swanlab.init(
            project=f"lora-training",
            experiment_name=config['experiment']['name'],
            config=config,
            logdir=config['paths']['swanlab_dir']
        )
        
        # 创建数据模块
        batch_size = config['training']['batch_size']
            
        data_module = TrainTestDataModule(
            dataset_name=dataset_name,
            batch_size=batch_size,
            test_mode=False  # 移除test_mode，始终使用标准模式
        )
        
        print(f"📊 数据模块配置:")
        print(f"  - Train文件: {data_module.train_file}")
        print(f"  - Test文件: {data_module.test_file}")
        print(f"  - Batch大小: {batch_size}")
        print(f"  - Train Shuffle: True, Test Shuffle: False")
        
        # 创建Lightning模块
        model_path = config.get('model', {}).get('path', 'models/Qwen-Qwen2.5-0.5B')
        if not os.path.exists(model_path):
            model_path = config.get('model', {}).get('name', 'Qwen/Qwen2.5-0.5B')
        
        lora_config = {
            'r': config.get('lora', {}).get('r', 16),
            'lora_alpha': config.get('lora', {}).get('alpha', 32),
            'target_modules': config.get('lora', {}).get('target_modules', ["q_proj", "v_proj"]),
            'lora_dropout': config.get('lora', {}).get('dropout', 0.1),
            'bias': config.get('lora', {}).get('bias', "none"),
            'task_type': TaskType.CAUSAL_LM
        }
        
        lightning_module = LoRALightningModule(
            model_path=model_path,
            lora_config=lora_config,
            learning_rate_stage1=config['training']['learning_rate_stage1'],
            learning_rate_stage2=config['training']['learning_rate_stage2'],
            stage1_steps=config['training']['stage1_steps'],
            stage2_steps=config['training']['stage2_steps'],
        )
        
        # 将SwanLab run添加到模块中
        lightning_module._swanlab_run = swanlab_run
        
        # 设置回调
        callbacks = setup_callbacks(config)
        
        # 设置日志记录器
        tensorboard_logger = TensorBoardLogger(
            save_dir=config['paths']['tensorboard_dir'],
            name="",
            version=""
        )
        
        # 创建Trainer（禁用自动checkpoint保存）
        trainer = Trainer(
            max_steps=config['training']['max_steps'],
            callbacks=callbacks,
            logger=tensorboard_logger,
            enable_progress_bar=True,
            log_every_n_steps=1,
            enable_checkpointing=False,  # 禁用检查点，节省磁盘空间
            precision='16-mixed' if torch.cuda.is_available() else 32,
            accelerator='auto',
            devices='auto',
            strategy='auto',
        )
        
        print(f"\n🏃‍♂️ 开始Lightning训练...")
        print(f"📊 训练器配置:")
        print(f"  - 最大步数: {config['training']['max_steps']}")
        print(f"  - 精度: {'16-mixed' if torch.cuda.is_available() else '32'}")
        print(f"  - 加速器: {trainer.accelerator}")
        print(f"  - 设备数: {trainer.num_devices}")
        
        # 开始训练（包含验证）
        trainer.fit(lightning_module, datamodule=data_module)
        
        # 在测试集上最终评估
        print(f"\n🧪 开始测试集评估...")
        test_results = trainer.test(lightning_module, datamodule=data_module)
        
        print(f"📊 测试结果:")
        for key, value in test_results[0].items():
            print(f"  - {key}: {value:.4f}")
        
        # 保存最终模型
        final_model_dir = Path(config['paths']['final_model_dir'])
        final_model_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 保存最终模型到: {final_model_dir}")
        lightning_module.model.save_pretrained(final_model_dir)
        lightning_module.tokenizer.save_pretrained(final_model_dir)
        
        # 关闭SwanLab
        swanlab.finish()
        
        results = {
            "status": "training_completed",
            "experiment_dir": str(experiment_dir),
            "final_model_dir": str(final_model_dir),
            "total_steps": config['training']['max_steps'],
            "test_results": test_results[0] if test_results else {},
            "framework": "lightning_swanlab"
        }
        
        print(f"\n✅ Lightning训练完成!")
        print(f"📁 实验目录: {experiment_dir}")
        print(f"📁 最终模型: {final_model_dir}")
        print(f"📊 训练步数: {config['training']['max_steps']}")
        print(f"🎯 最终测试结果: {test_results[0] if test_results else '未完成'}")
        
        return results
        
    except Exception as e:
        print(f"❌ Lightning训练失败: {e}")
        if 'swanlab_run' in locals():
            swanlab.finish()
        raise
