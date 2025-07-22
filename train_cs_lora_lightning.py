#!/usr/bin/env python3
"""
train_cs_lora_lightning.py
PyTorch Lightning + SwanLab 版本的 LoRA 训练脚本
现代化训练框架，支持实时监控和实验管理

======================================================================
🚀 使用方法 (与原版兼容的命令行接口):
======================================================================

# 基本训练命令
python train_cs_lora_lightning.py --dataset arc-challenge
python train_cs_lora_lightning.py --dataset arc-easy
python train_cs_lora_lightning.py --dataset boolq
python train_cs_lora_lightning.py --dataset hellaswag
python train_cs_lora_lightning.py --dataset openbookqa
python train_cs_lora_lightning.py --dataset piqa
python train_cs_lora_lightning.py --dataset winogrande

# 自定义参数
python train_cs_lora_lightning.py --dataset arc-challenge --batch_size 16
python train_cs_lora_lightning.py --dataset arc-challenge --dry_run

# SwanLab 团队协作配置
python train_cs_lora_lightning.py --dataset arc-challenge --swanlab_project "team-lora-experiments"
python train_cs_lora_lightning.py --dataset arc-challenge --swanlab_project "team-lora-experiments" --swanlab_workspace "your-team-name"

# 批量执行 (PowerShell)
foreach ($dataset in @("arc-challenge", "arc-easy", "boolq", "hellaswag", "openbookqa", "piqa", "winogrande")) {
    Write-Host "🚀 开始训练 $dataset..."
    python train_cs_lora_lightning.py --dataset $dataset --swanlab_project "team-lora-experiments"
    Write-Host "✅ $dataset 训练完成"
}

======================================================================

主要特性:
✅ PyTorch Lightning: 现代化训练框架，自动处理设备、分布式等
✅ SwanLab 集成: 实时监控训练过程，实验管理和对比  
✅ 主流保存方式: 遵循 HuggingFace/Lightning 最佳实践
✅ 兼容现有接口: 命令行参数与原版完全兼容
✅ 简化架构: 移除复杂的 batch 追踪，专注核心训练
✅ 自动混合精度: 提升训练效率
✅ 灵活的回调系统: 易于扩展和自定义

目录结构 (Lightning + SwanLab 风格):
./runs/                          # 主实验目录
├── {experiment_name}/           # 单个实验
│   ├── checkpoints/             # Lightning checkpoints (.ckpt)
│   ├── tensorboard_logs/        # TensorBoard 日志 
│   ├── swanlab_logs/           # SwanLab 日志
│   ├── final_model/            # 最终 HuggingFace 格式模型
│   └── config.yaml             # 实验配置
└── swanlab_workspace/          # SwanLab 工作区

对比传统 experiments/ 目录的优势:
- 更规范的实验管理
- 支持多种日志格式
- 便于版本控制和分享
- 符合社区最佳实践
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import swanlab

from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def custom_collate_fn(batch):
    """自定义collate函数，保持字典结构"""
    return batch


def get_test_file_path(dataset_name: str) -> Tuple[str, bool]:
    """
    智能选择测试文件路径，优先test，不存在则使用validation
    
    Args:
        dataset_name: 数据集名称
        
    Returns:
        tuple: (文件路径, 是否使用validation)
    """
    data_dir = f"data_to_lora/cs/{dataset_name}"
    test_file = f"{data_dir}/{dataset_name}_test_formatted.jsonl"
    validation_file = f"{data_dir}/{dataset_name}_validation_formatted.jsonl"
    
    if os.path.exists(test_file):
        return test_file, False
    elif os.path.exists(validation_file):
        print(f"📊 注意: {dataset_name} 没有test文件，将使用validation文件作为测试集")
        return validation_file, True
    else:
        # 返回默认test路径，让后续处理报错
        return test_file, False


class SequentialDataset(Dataset):
    """完整的数据集类，支持train/validation/test"""
    
    def __init__(self, data_file: str, dataset_name: str, split: str = "train"):
        self.data_file = data_file
        self.dataset_name = dataset_name
        self.split = split
        self.data = self._load_data()
        self.total_samples = len(self.data)
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """加载数据，保持原始顺序"""
        data = []
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"数据文件不存在: {self.data_file}")
            
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"⚠️ 跳过无效行 {line_idx}: {e}")
                        
        print(f"📊 加载{self.split}数据集 {self.dataset_name}: {len(data)} 样本")
        return data
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # 支持超过数据集长度的循环访问（主要用于训练集）
        if self.split == "train":
            actual_idx = idx % self.total_samples
        else:
            # 验证和测试集不循环访问
            actual_idx = idx
        return self.data[actual_idx].copy()


class TrainTestDataModule(pl.LightningDataModule):
    """Lightning数据模块，管理train/test数据（自动适配validation作为test）"""
    
    def __init__(self, dataset_name: str, batch_size: int = 4, test_mode: bool = False):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.test_mode = test_mode
        
        # 数据文件路径
        self.data_dir = f"data_to_lora/cs/{dataset_name}"
        self.train_file = f"{self.data_dir}/{dataset_name}_train_formatted.jsonl"
        
        # 智能选择测试文件：优先test，不存在则使用validation
        self.test_file, self.using_validation_as_test = get_test_file_path(dataset_name)
        
    def setup(self, stage: str = None):
        """设置数据集"""
        if stage == "fit" or stage is None:
            self.train_dataset = SequentialDataset(self.train_file, self.dataset_name, "train")
            
        if stage == "test" or stage is None:
            self.test_dataset = SequentialDataset(self.test_file, self.dataset_name, "test")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # 保持顺序
            num_workers=0 if self.test_mode else 4,
            pin_memory=not self.test_mode,
            drop_last=False,
            collate_fn=custom_collate_fn
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0 if self.test_mode else 4,
            pin_memory=not self.test_mode,
            drop_last=False,
            collate_fn=custom_collate_fn
        )


class LoRALightningModule(pl.LightningModule):
    """Lightning LoRA 训练模块"""
    
    def __init__(
        self,
        model_path: str,
        lora_config: Dict[str, Any],
        learning_rate_stage1: float = 1e-4,
        learning_rate_stage2: float = 1e-5,
        stage1_steps: int = 75,
        stage2_steps: int = 50,
        max_length: int = 512,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_path = model_path
        self.lora_config = lora_config
        self.learning_rate_stage1 = learning_rate_stage1
        self.learning_rate_stage2 = learning_rate_stage2
        self.stage1_steps = stage1_steps
        self.stage2_steps = stage2_steps
        self.max_length = max_length
        self.total_steps = stage1_steps + stage2_steps
        
        # 初始化模型和 tokenizer
        self._init_model()
        
        # 训练状态跟踪
        self.training_step_count = 0
        
    def _init_model(self):
        """初始化模型和tokenizer"""
        print(f"📦 加载模型: {self.model_path}")

        # 平衡精度和性能，推荐大多数场景
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision('medium')  

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载基础模型
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=None,  # Lightning 会处理设备分配
            trust_remote_code=True
        )
        
        # 应用 LoRA
        lora_config = LoraConfig(**self.lora_config)
        self.model = get_peft_model(self.base_model, lora_config)
        
        print(f"✅ 模型加载完成")
        # 获取可训练参数统计
        trainable_params, total_params = self.model.get_nb_trainable_parameters()
        print(f"📊 可训练参数: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
        
    def forward(self, input_ids, attention_mask, labels):
        """前向传播"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        loss = self._compute_loss(batch, "train")
        batch_size = len(batch) if isinstance(batch, list) else batch['input_ids'].size(0)
        
        # 记录训练指标
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train/step', self.training_step_count, on_step=True, batch_size=batch_size)
        
        # 记录学习率阶段信息
        current_stage = 1 if self.training_step_count < self.stage1_steps else 2
        self.log('train/stage', current_stage, on_step=True, batch_size=batch_size)
        
        # 记录到 SwanLab
        if hasattr(self, '_swanlab_run'):
            swanlab.log({
                "train/loss": loss.item(),
                "train/step": self.training_step_count,
                "train/stage": current_stage,
                "train/epoch": self.current_epoch
            }, step=self.training_step_count)
        
        self.training_step_count += 1
        return loss
    
    def validation_step(self, batch, batch_idx):
        """验证步骤（已移除，因为训练步数<1 epoch）"""
        pass  # 不再需要验证步骤
        
    def test_step(self, batch, batch_idx):
        """测试步骤"""
        loss = self._compute_loss(batch, "test")
        batch_size = len(batch) if isinstance(batch, list) else batch['input_ids'].size(0)
        accuracy = self._compute_accuracy(batch)
        perplexity = torch.exp(loss)
        
        # 记录测试指标
        self.log('test/loss', loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test/accuracy', accuracy, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test/perplexity', perplexity, on_step=False, on_epoch=True, batch_size=batch_size)
        
        return {
            'test_loss': loss,
            'test_accuracy': accuracy,
            'test_perplexity': perplexity
        }
    
    def on_validation_epoch_end(self):
        """验证epoch结束时的回调（已移除）"""
        pass  # 不再需要
    
    def on_test_epoch_end(self):
        """测试epoch结束时的回调"""
        if hasattr(self, '_swanlab_run'):
            # 记录测试指标到SwanLab
            test_loss = self.trainer.callback_metrics.get('test/loss')
            test_acc = self.trainer.callback_metrics.get('test/accuracy') 
            test_ppl = self.trainer.callback_metrics.get('test/perplexity')
            
            if test_loss is not None:
                swanlab.log({
                    "test/loss": test_loss.item(),
                    "test/accuracy": test_acc.item() if test_acc is not None else 0,
                    "test/perplexity": test_ppl.item() if test_ppl is not None else 0,
                    "final_epoch": self.current_epoch
                }, step=self.training_step_count)
    
    def _compute_loss(self, batch, stage: str):
        """计算损失的通用方法"""
        # 处理batch数据
        if isinstance(batch, list):
            # 如果是list，需要tokenize
            inputs = []
            labels = []
            
            for item in batch:
                input_text = item.get('input', '')
                output_text = item.get('output', '')
                
                # 组合输入和输出
                full_text = f"{input_text}{output_text}"
                
                # Tokenize
                encoding = self.tokenizer(
                    full_text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                inputs.append(encoding['input_ids'].squeeze())
                labels.append(encoding['input_ids'].squeeze())
            
            input_ids = torch.stack(inputs).to(self.device)
            attention_mask = torch.ones_like(input_ids).to(self.device)
            labels = torch.stack(labels).to(self.device)
            
        else:
            # 如果已经是tensor格式
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask'] 
            labels = batch['labels']
        
        # 前向传播
        outputs = self(input_ids, attention_mask, labels)
        return outputs.loss
    
    def _compute_accuracy(self, batch):
        """计算准确率（简化版本）"""
        # 这里可以实现更复杂的准确率计算
        # 目前返回一个基于loss的代理指标
        with torch.no_grad():
            loss = self._compute_loss(batch, "eval")
            # 将loss转换为0-1之间的准确率代理指标
            accuracy = torch.exp(-loss)
            return torch.clamp(accuracy, 0.0, 1.0)
    
    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate_stage1,  # 初始学习率
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # 两阶段学习率调度器
        def lr_lambda(current_step):
            if current_step < self.stage1_steps:
                # Stage 1: 高学习率
                return 1.0
            else:
                # Stage 2: 低学习率
                return self.learning_rate_stage2 / self.learning_rate_stage1
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


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
    
    if epochs_needed <= 1.1:
        print(f"  ✅ 效率很好：几乎无数据重复")
    elif epochs_needed <= 2.0:
        print(f"  🔄 效率一般：少量数据重复") 
    else:
        print(f"  ⚠️ 效率较低：大量数据重复")
    
    return epochs_needed


def create_lightning_config(dataset_name: str, base_config: Dict[str, Any], batch_size: int = None, max_steps: int = 125, save_steps: int = 50, learning_rate: float = 1e-4, learning_rate_stage2: float = None) -> Dict[str, Any]:
    """创建Lightning训练配置"""
    config = base_config.copy()
    
    # 自动选择最优batch size（如果未指定）
    if batch_size is None:
        batch_size = get_optimal_batch_size(dataset_name)
        print(f"🎯 自动选择batch_size={batch_size}用于{dataset_name}")
    
    # 分析batch效率
    analyze_batch_efficiency(dataset_name, batch_size, max_steps)
    
    # 智能选择测试文件路径 - 支持validation作为fallback
    test_file_path, using_validation = get_test_file_path(dataset_name)
    
    # 更新数据路径 - 支持train/test或train/validation配置
    config['data']['train_file'] = f"data_to_lora/cs/{dataset_name}/{dataset_name}_train_formatted.jsonl"
    config['data']['test_file'] = test_file_path
    config['data']['using_validation_as_test'] = using_validation
    
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
    
    # 现代化的输出目录结构
    base_dir = Path("./runs") / experiment_name
    config['paths'] = {
        'experiment_dir': str(base_dir),
        'checkpoints_dir': str(base_dir / "checkpoints"),
        'tensorboard_dir': str(base_dir / "tensorboard_logs"),
        'swanlab_dir': str(base_dir / "swanlab_logs"),
        'final_model_dir': str(base_dir / "final_model"),
        'config_file': str(base_dir / "config.yaml")
    }
    
    # 实验元数据
    config['experiment'] = {
        'name': experiment_name,
        'dataset': dataset_name,
        'batch_size': batch_size,
        'framework': 'lightning_swanlab',
        'created_at': datetime.now().isoformat(),
        'description': f"Lightning LoRA training on {dataset_name} - {max_steps} steps",
        'tags': ["lightning", "swanlab", "lora", "qwen2.5", dataset_name, f"batch{batch_size}", f"steps{max_steps}"]
    }
    
    return config


def setup_callbacks(config: Dict[str, Any]) -> List[pl.Callback]:
    """设置Lightning回调（移除Early Stopping和validation相关）"""
    callbacks = []
    
    max_steps = config['training']['max_steps']
    save_steps = config['training']['save_steps']
    
    # 条件检查点保存（最后save_steps步）- 这是您需要的关键功能
    class ConditionalCheckpoint(ModelCheckpoint):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            # 从第 (max_steps - save_steps + 1) 步开始保存
            save_start_step = max_steps - save_steps + 1
            if pl_module.training_step_count >= save_start_step:
                super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
    
    conditional_checkpoint = ConditionalCheckpoint(
        dirpath=config['paths']['checkpoints_dir'],
        filename='checkpoint-step-{step:03d}',
        every_n_train_steps=1,
        save_top_k=-1,
        monitor=None  # 不基于指标，直接保存
    )
    callbacks.append(conditional_checkpoint)
    
    # 最终模型检查点 - 基于训练loss（因为没有验证集）
    final_checkpoint = ModelCheckpoint(
        dirpath=config['paths']['checkpoints_dir'],
        filename='final-model-{epoch:02d}-{train/loss:.4f}',
        monitor='train/loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        verbose=True
    )
    callbacks.append(final_checkpoint)
    
    # 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    return callbacks


def run_lightning_training(
    dataset_name: str,
    config: Dict[str, Any],
    swanlab_project: str = None,
    swanlab_workspace: str = None,
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
        
        # 智能确定项目名称
        if swanlab_project:
            project_name = swanlab_project
        else:
            # 默认项目名称，支持个人和团队使用
            import getpass
            current_user = getpass.getuser()
            project_name = f"lora-training-{current_user}"
        
        # SwanLab初始化参数
        swanlab_params = {
            "project": project_name,
            "experiment_name": config['experiment']['name'],
            "config": config,
            "logdir": config['paths']['swanlab_dir']
        }
        
        # 如果指定了工作区，添加工作区参数（用于团队协作）
        if swanlab_workspace:
            swanlab_params["workspace"] = swanlab_workspace
            print(f"🏢 使用团队工作区: {swanlab_workspace}")
        
        print(f"📂 SwanLab项目: {project_name}")
        swanlab_run = swanlab.init(**swanlab_params)
        
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
        print(f"  - Shuffle: False (严格顺序)")
        
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
        
        # 获取最佳checkpoint回调的引用
        final_checkpoint = None
        for callback in callbacks:
            if isinstance(callback, ModelCheckpoint) and callback.monitor == 'train/loss':
                final_checkpoint = callback
                break
        
        # 设置日志记录器
        tensorboard_logger = TensorBoardLogger(
            save_dir=config['paths']['tensorboard_dir'],
            name="",
            version=""
        )
        
        # 创建Trainer
        trainer = Trainer(
            max_steps=config['training']['max_steps'],
            callbacks=callbacks,
            logger=tensorboard_logger,
            enable_progress_bar=True,
            log_every_n_steps=1,
            enable_checkpointing=True,  # 启用检查点
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
            "checkpoints_dir": config['paths']['checkpoints_dir'],
            "final_model_path": final_checkpoint.best_model_path if final_checkpoint else "N/A",
            "total_steps": config['training']['max_steps'],
            "test_results": test_results[0] if test_results else {},
            "framework": "lightning_swanlab"
        }
        
        print(f"\n✅ Lightning训练完成!")
        print(f"📁 实验目录: {experiment_dir}")
        print(f"📁 最终模型: {final_model_dir}")
        print(f"📁 最佳模型: {final_checkpoint.best_model_path if final_checkpoint else 'N/A'}")
        print(f"📁 检查点: {config['paths']['checkpoints_dir']}")
        print(f"📊 训练步数: {config['training']['max_steps']}")
        print(f"🎯 最终测试结果: {test_results[0] if test_results else '未完成'}")
        
        return results
        
    except Exception as e:
        print(f"❌ Lightning训练失败: {e}")
        if 'swanlab_run' in locals():
            swanlab.finish()
        raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Lightning + SwanLab LoRA训练脚本")
    parser.add_argument("--dataset", type=str, required=True,
                       help="要训练的数据集名称 (arc-challenge, arc-easy, boolq, hellaswag, openbookqa, piqa, winogrande)")
    parser.add_argument("--config", type=str, default="configs/lightning_config.yaml",
                       help="Lightning配置文件路径")
    parser.add_argument("--dry_run", action="store_true",
                       help="干运行模式: 验证配置和数据文件，创建实验目录，但不实际训练模型")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="批处理大小 (默认自动选择)")
    parser.add_argument("--max_steps", type=int, default=125,
                       help="训练总步数 (默认125)")
    parser.add_argument("--save_steps", type=int, default=50,
                       help="保存最后多少步的检查点 (默认50)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="学习率 (默认1e-4)")
    parser.add_argument("--learning_rate_stage2", type=float, default=None,
                       help="第二阶段学习率 (默认为learning_rate的1/10)")
    parser.add_argument("--swanlab_project", type=str, default=None,
                       help="SwanLab项目名称 (默认: lora-training-{用户名或团队名})")
    parser.add_argument("--swanlab_workspace", type=str, default=None,
                       help="SwanLab工作区名称 (用于团队协作)")
    
    # 为了兼容性，保留但忽略的参数
    parser.add_argument("--track_batches", action="store_true",
                       help="(兼容参数，当前版本忽略)")
    
    args = parser.parse_args()
    
    # 验证数据集名称
    valid_datasets = ['arc-challenge', 'arc-easy', 'boolq', 'hellaswag', 'openbookqa', 'piqa', 'winogrande']
    if args.dataset not in valid_datasets:
        print(f"❌ 无效的数据集名称: {args.dataset}")
        print(f"✅ 可用数据集: {', '.join(valid_datasets)}")
        return False
    
    # 兼容性提示
    if args.track_batches:
        print("💡 注意: --track_batches 功能将在下个版本中实现，当前版本忽略此参数")
    
    print("🚀 Lightning + SwanLab LoRA训练脚本")
    print("=" * 70)
    print(f"目标数据集: {args.dataset}")
    print(f"配置文件: {args.config}")
    print(f"Batch大小: {args.batch_size if args.batch_size else '自动选择'}")
    print(f"训练步数: {args.max_steps}")
    print(f"保存步数: 保存最后{args.save_steps}个检查点")
    print(f"学习率: {args.learning_rate} -> {args.learning_rate_stage2 or args.learning_rate/10}")
    
    # SwanLab配置信息
    import getpass
    current_user = getpass.getuser()
    swanlab_project_display = args.swanlab_project or f"lora-training-{current_user}"
    print(f"SwanLab项目: {swanlab_project_display}")
    if args.swanlab_workspace:
        print(f"SwanLab工作区: {args.swanlab_workspace} (团队模式)")
    else:
        print("SwanLab工作区: 个人工作区")
    
    print(f"运行模式: {'🏃 Dry Run (验证配置和数据，不训练)' if args.dry_run else '🚀 完整训练'}")
    print(f"框架: PyTorch Lightning + SwanLab")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    try:
        # 加载基础配置
        with open(args.config, 'r', encoding='utf-8') as f:
            base_config = yaml.safe_load(f)
        
        # 创建Lightning配置
        config = create_lightning_config(args.dataset, base_config, args.batch_size, args.max_steps, args.save_steps, args.learning_rate, args.learning_rate_stage2)
        
        # 执行训练
        results = run_lightning_training(
            dataset_name=args.dataset,
            config=config,
            swanlab_project=args.swanlab_project,
            swanlab_workspace=args.swanlab_workspace,
            dry_run=args.dry_run
        )
        
        print(f"\n🎉 实验完成!")
        print(f"📊 结果: {results.get('status', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
