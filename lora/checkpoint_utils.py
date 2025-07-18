"""
checkpoint_utils.py
LoRA checkpoint 的保存、加载与枚举
增强版本支持条件保存和详细信息记录
"""
import torch
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class CheckpointManager:
    """增强的Checkpoint管理器"""
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 50):
        """
        初始化checkpoint管理器
        
        Args:
            checkpoint_dir: checkpoint保存目录
            max_checkpoints: 最大保存checkpoint数量
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.step_counter = 0
        
        # 创建目录
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # checkpoint信息文件
        self.info_file = os.path.join(checkpoint_dir, "checkpoint_info.json")
        self.checkpoint_info = self._load_checkpoint_info()
        
        logger.info(f"CheckpointManager initialized: {checkpoint_dir}")
    
    def _load_checkpoint_info(self) -> Dict[str, Any]:
        """加载checkpoint信息"""
        if os.path.exists(self.info_file):
            try:
                with open(self.info_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint info: {e}")
        
        return {
            "checkpoints": [],
            "last_step": 0,
            "total_saved": 0
        }
    
    def _save_checkpoint_info(self):
        """保存checkpoint信息"""
        try:
            with open(self.info_file, 'w', encoding='utf-8') as f:
                json.dump(self.checkpoint_info, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save checkpoint info: {e}")
    
    def should_save_checkpoint(self, step: int, force_save: bool = False) -> bool:
        """
        判断是否应该保存checkpoint
        
        Args:
            step: 当前步数
            force_save: 强制保存
            
        Returns:
            bool: 是否保存
        """
        if force_save:
            return True
        
        # 前75步不保存，后50步每步都保存
        return step >= 75
    
    def save_checkpoint(
        self, 
        model, 
        optimizer, 
        step: int, 
        loss: float,
        lr: float,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        保存checkpoint
        
        Args:
            model: 模型
            optimizer: 优化器
            step: 当前步数
            loss: 当前损失
            lr: 当前学习率
            additional_info: 额外信息
            
        Returns:
            str: checkpoint文件路径，如果未保存则返回None
        """
        if not self.should_save_checkpoint(step):
            return None
        
        # 生成checkpoint文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_step_{step:04d}_{timestamp}.pt"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        try:
            # 准备保存数据
            checkpoint_data = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'lr': lr,
                'timestamp': timestamp,
                'additional_info': additional_info or {}
            }
            
            # 保存checkpoint
            torch.save(checkpoint_data, checkpoint_path)
            
            # 更新信息记录
            checkpoint_record = {
                'step': step,
                'path': checkpoint_path,
                'filename': checkpoint_name,
                'loss': loss,
                'lr': lr,
                'timestamp': timestamp,
                'size_mb': os.path.getsize(checkpoint_path) / (1024 * 1024)
            }
            
            self.checkpoint_info['checkpoints'].append(checkpoint_record)
            self.checkpoint_info['last_step'] = step
            self.checkpoint_info['total_saved'] += 1
            
            # 清理旧checkpoint
            self._cleanup_old_checkpoints()
            
            # 保存信息文件
            self._save_checkpoint_info()
            
            logger.info(f"Checkpoint saved: {checkpoint_name} (step={step}, loss={loss:.4f}, lr={lr:.2e})")
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint at step {step}: {e}")
            return None
    
    def _cleanup_old_checkpoints(self):
        """清理旧的checkpoint文件"""
        if len(self.checkpoint_info['checkpoints']) <= self.max_checkpoints:
            return
        
        # 按步数排序，保留最新的
        self.checkpoint_info['checkpoints'].sort(key=lambda x: x['step'])
        
        # 删除最旧的checkpoints
        to_remove = self.checkpoint_info['checkpoints'][:-self.max_checkpoints]
        
        for checkpoint in to_remove:
            try:
                if os.path.exists(checkpoint['path']):
                    os.remove(checkpoint['path'])
                    logger.info(f"Removed old checkpoint: {checkpoint['filename']}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint['filename']}: {e}")
        
        # 更新记录
        self.checkpoint_info['checkpoints'] = self.checkpoint_info['checkpoints'][-self.max_checkpoints:]
    
    def load_latest_checkpoint(self, model, optimizer) -> Optional[Dict[str, Any]]:
        """
        加载最新的checkpoint
        
        Args:
            model: 模型
            optimizer: 优化器
            
        Returns:
            dict: checkpoint信息，如果没有则返回None
        """
        if not self.checkpoint_info['checkpoints']:
            logger.info("No checkpoints found")
            return None
        
        # 获取最新checkpoint
        latest_checkpoint = max(self.checkpoint_info['checkpoints'], key=lambda x: x['step'])
        checkpoint_path = latest_checkpoint['path']
        
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            return None
        
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            
            model.load_state_dict(checkpoint_data['model_state_dict'])
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            
            logger.info(f"Loaded checkpoint: {latest_checkpoint['filename']} (step={checkpoint_data['step']})")
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return None
    
    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """获取checkpoint摘要信息"""
        total_size_mb = sum(ckpt.get('size_mb', 0) for ckpt in self.checkpoint_info['checkpoints'])
        
        return {
            'total_checkpoints': len(self.checkpoint_info['checkpoints']),
            'total_size_mb': total_size_mb,
            'last_step': self.checkpoint_info['last_step'],
            'checkpoint_dir': self.checkpoint_dir,
            'latest_checkpoints': self.checkpoint_info['checkpoints'][-5:] if self.checkpoint_info['checkpoints'] else []
        }

def save_checkpoint(model, path):
    """
    保存模型 checkpoint (保持向后兼容)
    """
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path):
    """
    加载模型 checkpoint (保持向后兼容)
    """
    model.load_state_dict(torch.load(path))
    return model

def list_checkpoints(folder):
    """
    枚举文件夹下所有 .pt checkpoint 路径 (保持向后兼容)
    """
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.pt')]
