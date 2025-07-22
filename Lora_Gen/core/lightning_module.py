"""
lightning_module.py
Lightning训练模块，集成SwanLab
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import swanlab
from typing import Dict, List, Any, Optional, Tuple
import os
from pathlib import Path

from .generator import LoRAParameterGenerator, LoRATokenizer
from .data_module import LoRAGeneratorDataModule


class LoRAGeneratorLightningModule(pl.LightningModule):
    """
    LoRA参数生成器的Lightning训练模块
    """
    def __init__(
        self,
        # Model config
        text_encoder_name: str = 'all-MiniLM-L6-v2',
        hidden_dim: int = 384,
        max_seq_len: int = 512,
        num_hyperconv_blocks: int = 3,
        output_dim: int = 384,
        
        # Training config
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        
        # Loss config
        mse_weight: float = 1.0,
        l1_weight: float = 0.1,
        
        # Optimizer config
        optimizer_type: str = 'adamw',
        scheduler_type: str = 'cosine',
        
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # 模型参数
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        
        # 初始化模型
        self.generator = LoRAParameterGenerator(
            text_encoder_name=text_encoder_name,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
            num_hyperconv_blocks=num_hyperconv_blocks,
            output_dim=output_dim
        )
        
        # 初始化tokenizer
        self.tokenizer = LoRATokenizer(
            max_tokens=max_seq_len,
            token_dim=output_dim
        )
        
        # 训练状态跟踪
        self.training_step_count = 0
        
    def forward(self, prompts):
        """前向传播"""
        return self.generator(prompts)
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算损失函数
        
        Args:
            predictions: [B, max_seq_len, output_dim] - 生成的参数
            targets: [B, max_seq_len, output_dim] - 目标参数
            
        Returns:
            loss_dict: 包含各种损失的字典
        """
        # MSE Loss (主要损失)
        mse_loss = F.mse_loss(predictions, targets)
        
        # L1 Loss (稀疏性正则化)
        l1_loss = F.l1_loss(predictions, targets)
        
        # 总损失
        total_loss = self.mse_weight * mse_loss + self.l1_weight * l1_loss
        
        # 计算额外指标
        with torch.no_grad():
            mae = F.l1_loss(predictions, targets)
            rmse = torch.sqrt(mse_loss)
            
            # 相对误差
            target_norm = torch.norm(targets, dim=-1).mean()
            pred_norm = torch.norm(predictions, dim=-1).mean()
            relative_error = torch.abs(pred_norm - target_norm) / (target_norm + 1e-8)
        
        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'l1_loss': l1_loss,
            'mae': mae,
            'rmse': rmse,
            'relative_error': relative_error
        }
    
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        prompts, target_params = batch
        
        # 前向传播
        generated_params = self.generator(prompts)
        
        # 计算损失
        loss_dict = self.compute_loss(generated_params, target_params)
        
        # 记录指标
        self.log_dict({
            f'train/{k}': v for k, v in loss_dict.items()
        }, on_step=True, on_epoch=True, prog_bar=True)
        
        # SwanLab记录
        if hasattr(self, '_swanlab_run'):
            swanlab_metrics = {f'train/{k}': v.item() for k, v in loss_dict.items()}
            swanlab_metrics.update({
                'train/step': self.training_step_count,
                'train/epoch': self.current_epoch,
                'train/lr': self.optimizers().param_groups[0]['lr']
            })
            swanlab.log(swanlab_metrics, step=self.training_step_count)
        
        self.training_step_count += 1
        return loss_dict['total_loss']
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        prompts, target_params = batch
        
        # 前向传播
        generated_params = self.generator(prompts)
        
        # 计算损失
        loss_dict = self.compute_loss(generated_params, target_params)
        
        # 记录指标
        self.log_dict({
            f'val/{k}': v for k, v in loss_dict.items()
        }, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss_dict
    
    def test_step(self, batch, batch_idx):
        """测试步骤"""
        prompts, target_params = batch
        
        # 前向传播
        generated_params = self.generator(prompts)
        
        # 计算损失
        loss_dict = self.compute_loss(generated_params, target_params)
        
        # 记录指标
        self.log_dict({
            f'test/{k}': v for k, v in loss_dict.items()
        }, on_step=False, on_epoch=True)
        
        return loss_dict
    
    def on_validation_epoch_end(self):
        """验证epoch结束时的回调"""
        if hasattr(self, '_swanlab_run'):
            # 记录验证指标到SwanLab
            val_loss = self.trainer.callback_metrics.get('val/total_loss')
            val_mse = self.trainer.callback_metrics.get('val/mse_loss')
            val_mae = self.trainer.callback_metrics.get('val/mae')
            
            if val_loss is not None:
                swanlab.log({
                    'val/total_loss': val_loss.item(),
                    'val/mse_loss': val_mse.item() if val_mse is not None else 0,
                    'val/mae': val_mae.item() if val_mae is not None else 0,
                    'epoch': self.current_epoch
                }, step=self.training_step_count)
    
    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        # 优化器
        if self.optimizer_type.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999)
            )
        elif self.optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")
        
        # 学习率调度器
        if self.scheduler_type.lower() == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs if self.trainer.max_epochs else 100
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                }
            }
        elif self.scheduler_type.lower() == 'linear_warmup':
            def lr_lambda(current_step):
                if current_step < self.warmup_steps:
                    return float(current_step) / float(max(1, self.warmup_steps))
                return max(0.0, float(self.trainer.max_steps - current_step) / 
                          float(max(1, self.trainer.max_steps - self.warmup_steps)))
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                }
            }
        else:
            return optimizer


def setup_callbacks(config: Dict[str, Any]) -> List[pl.Callback]:
    """设置Lightning回调"""
    callbacks = []
    
    # ModelCheckpoint - 保存最佳模型
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['paths']['checkpoints_dir'],
        filename='{epoch:02d}-{val/total_loss:.4f}',
        monitor='val/total_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # EarlyStopping
    early_stopping = EarlyStopping(
        monitor='val/total_loss',
        patience=config['training'].get('early_stopping_patience', 10),
        mode='min',
        verbose=True
    )
    callbacks.append(early_stopping)
    
    # 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    return callbacks


class SwanLabLogger:
    """SwanLab Logger for LoRA Generator"""
    
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
