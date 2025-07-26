#!/usr/bin/env python3
"""
数据处理模块
包含数据集类、数据模块、数据路径处理等功能
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl


def get_test_file_path(dataset_name: str) -> Tuple[str, bool]:
    """
    智能选择测试文件路径，优先test，不存在则使用validation
    
    Args:
        dataset_name: 数据集名称
        
    Returns:
        tuple: (文件路径, 是否使用validation)
    """
    # 获取项目根目录的绝对路径
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, f"data_to_lora/cs/{dataset_name}")
    test_file = os.path.join(data_dir, f"{dataset_name}_test_formatted.jsonl")
    validation_file = os.path.join(data_dir, f"{dataset_name}_validation_formatted.jsonl")
    
    if os.path.exists(test_file):
        return test_file, False
    elif os.path.exists(validation_file):
        print(f"📊 注意: {dataset_name} 没有test文件，将使用validation文件作为测试集")
        return validation_file, True
    else:
        # 返回默认test路径，让后续处理报错
        return test_file, False


def custom_collate_fn(batch):
    """自定义collate函数，保持字典结构"""
    return batch


class SequentialDataset(Dataset):
    """完整的数据集类，支持train/validation/test"""
    
    def __init__(self, data_file: str, dataset_name: str, split: str = "train"):
        self.data_file = data_file
        self.dataset_name = dataset_name
        self.split = split
        self.data = self._load_data()
        self.total_samples = len(self.data)
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """加载数据"""
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
    """Lightning数据模块，管理train/validation/test数据"""
    
    def __init__(self, dataset_name: str, batch_size: int = 4, test_mode: bool = False):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.test_mode = test_mode
        
        # 数据文件路径
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_dir = os.path.join(project_root, f"data_to_lora/cs/{dataset_name}")
        self.train_file = os.path.join(self.data_dir, f"{dataset_name}_train_formatted.jsonl")
        
        # 验证集文件路径
        self.validation_file = os.path.join(self.data_dir, f"{dataset_name}_validation_formatted.jsonl")
        
        # 智能选择测试文件：优先test，不存在则使用validation
        self.test_file, self.using_validation_as_test = get_test_file_path(dataset_name)
        
    def setup(self, stage: str = None):
        """设置数据集"""
        if stage == "fit" or stage is None:
            self.train_dataset = SequentialDataset(self.train_file, self.dataset_name, "train")
            # 添加验证集
            if os.path.exists(self.validation_file):
                self.val_dataset = SequentialDataset(self.validation_file, self.dataset_name, "validation")
                print(f"📊 使用验证集: {self.validation_file}")
            else:
                print(f"⚠️  验证集文件不存在: {self.validation_file}")
                self.val_dataset = None
            
        if stage == "test" or stage is None:
            self.test_dataset = SequentialDataset(self.test_file, self.dataset_name, "test")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0 if self.test_mode else 4,
            pin_memory=not self.test_mode,
            drop_last=False,
            collate_fn=custom_collate_fn
        )
    
    def val_dataloader(self):
        """验证集数据加载器"""
        if hasattr(self, 'val_dataset') and self.val_dataset is not None:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,  # 验证集不shuffle
                num_workers=0 if self.test_mode else 4,
                pin_memory=not self.test_mode,
                drop_last=False,
                collate_fn=custom_collate_fn
            )
        return None
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # 测试集不shuffle，保证结果可重现
            num_workers=0 if self.test_mode else 4,
            pin_memory=not self.test_mode,
            drop_last=False,
            collate_fn=custom_collate_fn
        )
