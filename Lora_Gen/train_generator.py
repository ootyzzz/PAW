#!/usr/bin/env python3
"""
train_generator.py
LoRA参数生成器训练脚本
Lightning + SwanLab 版本

使用方法：
python train_generator.py --config config/generator_config.yaml
python train_generator.py --config config/generator_config.yaml --checkpoint_dir runs/arc-challenge_lora_20250721_005053/checkpoints
python train_generator.py --dry_run  # 验证配置和数据
"""
import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import warnings

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import swanlab

# 导入我们的模块
from core.lightning_module import (
    LoRAGeneratorLightningModule, 
    setup_callbacks, 
    SwanLabLogger
)
from core.data_module import LoRAGeneratorDataModule, create_prompt_splits
from core.generator import LoRATokenizer

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning)

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )

def validate_config(config: Dict[str, Any]) -> bool:
    """验证配置文件"""
    required_keys = [
        'model', 'data', 'training', 'lightning', 'logging'
    ]
    
    for key in required_keys:
        if key not in config:
            print(f"❌ 配置文件缺少必需键: {key}")
            return False
    
    # 验证路径
    checkpoint_dir = Path(config['data']['checkpoint_dir'])
    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint目录不存在: {checkpoint_dir}")
        return False
    
    return True

def setup_experiment_paths(config: Dict[str, Any], experiment_name: str) -> Dict[str, Any]:
    """设置实验路径"""
    base_dir = Path("Lora_Gen/experiments") / experiment_name
    
    config['paths'] = {
        'experiment_dir': str(base_dir),
        'checkpoints_dir': str(base_dir / "checkpoints"),
        'tensorboard_dir': str(base_dir / "tensorboard_logs"),
        'swanlab_dir': str(base_dir / "swanlab_logs"),
        'results_dir': str(base_dir / "results"),
        'config_file': str(base_dir / "config.yaml")
    }
    
    # 创建目录
    for path in config['paths'].values():
        if path.endswith('.yaml'):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        else:
            Path(path).mkdir(parents=True, exist_ok=True)
    
    return config

def prepare_data(config: Dict[str, Any], force_recreate: bool = False) -> bool:
    """准备训练数据"""
    data_config = config['data']
    
    # 检查prompt文件是否存在
    train_file = Path(data_config['train_prompt_file'])
    val_file = Path(data_config['val_prompt_file'])
    
    if not train_file.exists() or not val_file.exists() or force_recreate:
        print("📊 创建训练/验证数据split...")
        
        # 源数据文件
        source_file = "data_to_lora/cs/arc-challenge/arc-challenge_train_formatted.jsonl"
        if not Path(source_file).exists():
            print(f"❌ 源数据文件不存在: {source_file}")
            return False
        
        # 创建数据目录
        train_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建split
        create_prompt_splits(
            input_file=source_file,
            train_output=str(train_file),
            val_output=str(val_file),
            samples_per_prompt=data_config['samples_per_prompt'],
            test_ratio=1.0 - data_config['train_ratio']
        )
    
    # 验证checkpoint目录
    checkpoint_dir = Path(data_config['checkpoint_dir'])
    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint目录不存在: {checkpoint_dir}")
        return False
    
    # 统计checkpoint文件
    checkpoint_files = []
    for ext in ['*.ckpt', '*.pt', '*.pth']:
        checkpoint_files.extend(list(checkpoint_dir.glob(ext)))
    
    print(f"📊 发现 {len(checkpoint_files)} 个checkpoint文件")
    
    if len(checkpoint_files) == 0:
        print(f"❌ 在 {checkpoint_dir} 中未找到checkpoint文件")
        return False
    
    return True

def run_training(config: Dict[str, Any], dry_run: bool = False) -> bool:
    """运行训练"""
    try:
        # 生成实验名称
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"lora_generator_{timestamp}"
        
        # 设置实验路径
        config = setup_experiment_paths(config, experiment_name)
        
        # 更新实验配置
        config['experiment'] = {
            'name': experiment_name,
            'created_at': datetime.now().isoformat(),
            'description': config.get('experiment', {}).get('description', 
                                                          "LoRA Parameter Generator training"),
            'framework': 'lightning_swanlab',
            'tags': config.get('experiment', {}).get('tags', [])
        }
        
        # 更新SwanLab配置
        if config['logging']['swanlab']['experiment_name'] is None:
            config['logging']['swanlab']['experiment_name'] = experiment_name
        
        print(f"\\n{'=' * 70}")
        print(f"🚀 LoRA Parameter Generator 训练")
        print(f"{'=' * 70}")
        print(f"实验名称: {experiment_name}")
        print(f"实验目录: {config['paths']['experiment_dir']}")
        print(f"Checkpoint源: {config['data']['checkpoint_dir']}")
        print(f"模型配置: {config['model']}")
        print(f"训练配置: batch_size={config['data']['batch_size']}, lr={config['training']['learning_rate']}")
        
        if dry_run:
            print("🏃 Dry Run 模式 - 验证完成!")
            return True
        
        # 保存配置
        with open(config['paths']['config_file'], 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2, allow_unicode=True)
        
        # 准备数据
        if not prepare_data(config):
            return False
        
        # 初始化SwanLab
        print("📊 初始化 SwanLab...")
        swanlab_config = config['logging']['swanlab']
        swanlab_params = {
            "project": swanlab_config['project'],
            "experiment_name": swanlab_config['experiment_name'],
            "config": config,
            "logdir": config['paths']['swanlab_dir']
        }
        
        if swanlab_config.get('workspace'):
            swanlab_params["workspace"] = swanlab_config['workspace']
        
        swanlab_run = swanlab.init(**swanlab_params)
        
        # 创建数据模块
        print("📊 创建数据模块...")
        data_module = LoRAGeneratorDataModule(
            train_prompt_file=config['data']['train_prompt_file'],
            val_prompt_file=config['data']['val_prompt_file'],
            checkpoint_dir=config['data']['checkpoint_dir'],
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers'],
            samples_per_prompt=config['data']['samples_per_prompt'],
            max_checkpoints=config['data']['max_checkpoints'],
            max_seq_len=config['model']['max_seq_len'],
            token_dim=config['model']['output_dim'],
            cache_tokenized=config['data']['cache_tokenized'],
            train_ratio=config['data']['train_ratio']
        )
        
        # 创建Lightning模块
        print("🧠 创建Lightning模块...")
        lightning_module = LoRAGeneratorLightningModule(
            # Model config
            text_encoder_name=config['model']['text_encoder_name'],
            hidden_dim=config['model']['hidden_dim'],
            max_seq_len=config['model']['max_seq_len'],
            num_hyperconv_blocks=config['model']['num_hyperconv_blocks'],
            output_dim=config['model']['output_dim'],
            
            # Training config
            learning_rate=float(config['training']['learning_rate']),
            weight_decay=float(config['training']['weight_decay']),
            warmup_steps=int(config['training']['warmup_steps']),
            
            # Loss config
            mse_weight=float(config['training']['mse_weight']),
            l1_weight=float(config['training']['l1_weight']),
            
            # Optimizer config
            optimizer_type=config['training']['optimizer_type'],
            scheduler_type=config['training']['scheduler_type']
        )
        
        # 将SwanLab run添加到模块
        lightning_module._swanlab_run = swanlab_run
        
        # 设置回调
        callbacks = setup_callbacks(config)
        
        # 设置TensorBoard日志记录器
        tensorboard_logger = TensorBoardLogger(
            save_dir=config['paths']['tensorboard_dir'],
            name=config['logging']['tensorboard']['name'],
            version=config['logging']['tensorboard']['version']
        )
        
        # 创建Trainer
        trainer_config = config['lightning']
        trainer = Trainer(
            max_epochs=config['training']['max_epochs'],
            max_steps=config['training']['max_steps'],
            callbacks=callbacks,
            logger=tensorboard_logger,
            accelerator=trainer_config['accelerator'],
            devices=trainer_config['devices'],
            strategy=trainer_config['strategy'],
            precision=config['training']['precision'],
            gradient_clip_val=config['training']['gradient_clip_val'],
            accumulate_grad_batches=config['training']['accumulate_grad_batches'],
            val_check_interval=config['training']['val_check_interval'],
            check_val_every_n_epoch=config['training']['check_val_every_n_epoch'],
            enable_progress_bar=trainer_config['enable_progress_bar'],
            log_every_n_steps=trainer_config['log_every_n_steps'],
            deterministic=trainer_config['deterministic']
        )
        
        print(f"\\n🏃‍♂️ 开始训练...")
        print(f"📊 训练器配置:")
        print(f"  - 最大epochs: {config['training']['max_epochs']}")
        print(f"  - 精度: {config['training']['precision']}")
        print(f"  - 加速器: {trainer.accelerator}")
        print(f"  - 设备: {trainer.num_devices}")
        
        # 开始训练
        trainer.fit(lightning_module, datamodule=data_module)
        
        # 测试（如果有测试数据）
        if data_module.test_dataloader() is not None:
            print(f"\\n🧪 开始测试...")
            test_results = trainer.test(lightning_module, datamodule=data_module)
            print(f"📊 测试结果: {test_results}")
        
        # 保存最终模型
        final_model_dir = Path(config['paths']['results_dir']) / "final_model"
        final_model_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': lightning_module.generator.state_dict(),
            'config': config,
            'experiment_name': experiment_name
        }, final_model_dir / "generator_model.pt")
        
        print(f"\\n✅ 训练完成!")
        print(f"📁 实验目录: {config['paths']['experiment_dir']}")
        print(f"📁 最终模型: {final_model_dir}")
        
        # 关闭SwanLab
        swanlab.finish()
        
        return True
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        if 'swanlab_run' in locals():
            swanlab.finish()
        return False

def main():
    """主函数"""
    setup_logging()
    
    parser = argparse.ArgumentParser(description="LoRA Parameter Generator 训练脚本")
    parser.add_argument("--config", type=str, default="config/generator_config.yaml",
                       help="配置文件路径")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                       help="Checkpoint目录路径 (覆盖配置文件)")
    parser.add_argument("--dry_run", action="store_true",
                       help="干运行模式: 验证配置但不训练")
    parser.add_argument("--force_recreate_data", action="store_true",
                       help="强制重新创建训练数据")
    
    args = parser.parse_args()
    
    print("🚀 LoRA Parameter Generator 训练脚本")
    print("=" * 70)
    print(f"配置文件: {args.config}")
    print(f"运行模式: {'🏃 Dry Run' if args.dry_run else '🚀 完整训练'}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    try:
        # 加载配置
        config_path = Path(args.config)
        if not config_path.exists():
            # 尝试相对于脚本目录的路径
            config_path = Path(__file__).parent / args.config
        
        if not config_path.exists():
            print(f"❌ 配置文件不存在: {args.config}")
            return False
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 命令行参数覆盖
        if args.checkpoint_dir:
            config['data']['checkpoint_dir'] = args.checkpoint_dir
            print(f"🔄 使用命令行checkpoint目录: {args.checkpoint_dir}")
        
        # 验证配置
        if not validate_config(config):
            return False
        
        # 运行训练
        return run_training(config, dry_run=args.dry_run)
        
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
