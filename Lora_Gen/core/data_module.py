"""
data_module.py
数据模块：处理prompt和checkpoint的配对数据
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import json
import random
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from .generator import LoRATokenizer


class PromptCheckpointDataset(Dataset):
    """
    Prompt-Checkpoint配对数据集
    """
    def __init__(
        self,
        prompt_file: str,
        checkpoint_dir: str,
        tokenizer: LoRATokenizer,
        samples_per_prompt: int = 4,
        max_checkpoints: int = 50,
        cache_tokenized: bool = True
    ):
        self.prompt_file = prompt_file
        self.checkpoint_dir = Path(checkpoint_dir)
        self.tokenizer = tokenizer
        self.samples_per_prompt = samples_per_prompt
        self.max_checkpoints = max_checkpoints
        self.cache_tokenized = cache_tokenized
        
        # 加载数据
        self.prompts = self._load_prompts()
        self.checkpoint_paths = self._load_checkpoint_paths()
        self.tokenized_cache = {} if cache_tokenized else None
        
        print(f"📊 数据集统计:")
        print(f"  - Prompts: {len(self.prompts)}")
        print(f"  - Checkpoints: {len(self.checkpoint_paths)}")
        print(f"  - 总样本数: {len(self)}")
    
    def _load_prompts(self) -> List[str]:
        """加载组合prompts"""
        prompts = []
        with open(self.prompt_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    prompts.append(item['prompt'])
        
        return prompts
    
    def _load_checkpoint_paths(self) -> List[str]:
        """加载checkpoint路径"""
        checkpoint_files = []
        
        # 查找所有.ckpt或.pt文件
        for ext in ['*.ckpt', '*.pt', '*.pth']:
            checkpoint_files.extend(list(self.checkpoint_dir.glob(ext)))
        
        # 限制数量
        if len(checkpoint_files) > self.max_checkpoints:
            checkpoint_files = checkpoint_files[:self.max_checkpoints]
        
        return [str(f) for f in checkpoint_files]
    
    def _tokenize_checkpoint(self, checkpoint_path: str) -> torch.Tensor:
        """tokenize checkpoint，使用缓存"""
        if self.cache_tokenized and checkpoint_path in self.tokenized_cache:
            return self.tokenized_cache[checkpoint_path]
        
        try:
            tokens = self.tokenizer.tokenize_checkpoint(checkpoint_path)
            
            if self.cache_tokenized:
                self.tokenized_cache[checkpoint_path] = tokens
            
            return tokens
        except Exception as e:
            print(f"⚠️ 无法tokenize checkpoint {checkpoint_path}: {e}")
            # 返回零向量作为fallback
            return torch.zeros(self.tokenizer.max_tokens, self.tokenizer.token_dim)
    
    def __len__(self):
        return len(self.prompts) * len(self.checkpoint_paths)
    
    def __getitem__(self, idx):
        # 计算prompt和checkpoint的索引
        prompt_idx = idx // len(self.checkpoint_paths)
        checkpoint_idx = idx % len(self.checkpoint_paths)
        
        # 获取数据
        prompt = self.prompts[prompt_idx]
        checkpoint_path = self.checkpoint_paths[checkpoint_idx]
        
        # Tokenize checkpoint
        tokenized_params = self._tokenize_checkpoint(checkpoint_path)
        
        return {
            'prompt': prompt,
            'target_params': tokenized_params,
            'checkpoint_path': checkpoint_path,
            'prompt_idx': prompt_idx,
            'checkpoint_idx': checkpoint_idx
        }


class LoRAGeneratorDataModule(pl.LightningDataModule):
    """
    Lightning数据模块
    """
    def __init__(
        self,
        train_prompt_file: str,
        val_prompt_file: str,
        checkpoint_dir: str,
        batch_size: int = 8,
        num_workers: int = 4,
        samples_per_prompt: int = 4,
        max_checkpoints: int = 50,
        max_seq_len: int = 512,
        token_dim: int = 384,
        cache_tokenized: bool = True,
        train_ratio: float = 0.9
    ):
        super().__init__()
        self.train_prompt_file = train_prompt_file
        self.val_prompt_file = val_prompt_file
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.samples_per_prompt = samples_per_prompt
        self.max_checkpoints = max_checkpoints
        self.max_seq_len = max_seq_len
        self.token_dim = token_dim
        self.cache_tokenized = cache_tokenized
        self.train_ratio = train_ratio
        
        # 初始化tokenizer
        self.tokenizer = LoRATokenizer(
            max_tokens=max_seq_len,
            token_dim=token_dim
        )
    
    def setup(self, stage: Optional[str] = None):
        """设置数据集"""
        if stage == "fit" or stage is None:
            # 如果只有一个prompt文件，需要分割
            if self.val_prompt_file is None or not os.path.exists(self.val_prompt_file):
                # 使用训练文件创建数据集，然后分割
                full_dataset = PromptCheckpointDataset(
                    prompt_file=self.train_prompt_file,
                    checkpoint_dir=self.checkpoint_dir,
                    tokenizer=self.tokenizer,
                    samples_per_prompt=self.samples_per_prompt,
                    max_checkpoints=self.max_checkpoints,
                    cache_tokenized=self.cache_tokenized
                )
                
                # 分割数据集
                train_size = int(len(full_dataset) * self.train_ratio)
                val_size = len(full_dataset) - train_size
                
                self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                    full_dataset, [train_size, val_size]
                )
            else:
                # 分别创建训练和验证数据集
                self.train_dataset = PromptCheckpointDataset(
                    prompt_file=self.train_prompt_file,
                    checkpoint_dir=self.checkpoint_dir,
                    tokenizer=self.tokenizer,
                    samples_per_prompt=self.samples_per_prompt,
                    max_checkpoints=self.max_checkpoints,
                    cache_tokenized=self.cache_tokenized
                )
                
                self.val_dataset = PromptCheckpointDataset(
                    prompt_file=self.val_prompt_file,
                    checkpoint_dir=self.checkpoint_dir,
                    tokenizer=self.tokenizer,
                    samples_per_prompt=self.samples_per_prompt,
                    max_checkpoints=self.max_checkpoints,
                    cache_tokenized=self.cache_tokenized
                )
        
        if stage == "test" or stage is None:
            self.test_dataset = self.val_dataset if hasattr(self, 'val_dataset') else None
    
    def collate_fn(self, batch):
        """自定义collate函数"""
        prompts = [item['prompt'] for item in batch]
        target_params = torch.stack([item['target_params'] for item in batch])
        
        return prompts, target_params
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )
    
    def test_dataloader(self):
        if self.test_dataset is not None:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                pin_memory=True
            )
        return None


def create_prompt_splits(
    input_file: str,
    train_output: str,
    val_output: str,
    samples_per_prompt: int = 4,
    test_ratio: float = 0.1
):
    """
    从单个数据文件创建训练/验证split
    
    Args:
        input_file: 输入的jsonl文件
        train_output: 训练集输出文件
        val_output: 验证集输出文件
        samples_per_prompt: 每个prompt包含的样本数
        test_ratio: 验证集比例
    """
    # 加载数据
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    
    print(f"📊 加载数据: {len(data)} 样本")
    
    # 随机打乱
    random.shuffle(data)
    
    # 创建non-overlapping的组合
    prompts = []
    for i in range(0, len(data) - samples_per_prompt + 1, samples_per_prompt):
        batch = data[i:i + samples_per_prompt]
        
        # 拼接成组合prompt
        combined_prompt = ""
        for item in batch:
            prompt_text = f"Question: {item['input']}\n"
            for option in item['options']:
                prompt_text += f"{option}\n"
            prompt_text += f"Answer: {item['target']}\n\n"
            combined_prompt += prompt_text
        
        combined_prompt = combined_prompt.strip()
        prompts.append({
            "prompt": combined_prompt,
            "sample_count": len(batch),
            "batch_info": [{"id": item["id"], "target": item["target"]} for item in batch]
        })
    
    print(f"📊 创建组合: {len(prompts)} 个prompt组合")
    
    # 分割训练和验证集
    split_idx = int(len(prompts) * (1 - test_ratio))
    train_prompts = prompts[:split_idx]
    val_prompts = prompts[split_idx:]
    
    # 保存训练集
    with open(train_output, 'w', encoding='utf-8') as f:
        for prompt_item in train_prompts:
            f.write(json.dumps(prompt_item, ensure_ascii=False) + '\n')
    
    # 保存验证集
    with open(val_output, 'w', encoding='utf-8') as f:
        for prompt_item in val_prompts:
            f.write(json.dumps(prompt_item, ensure_ascii=False) + '\n')
    
    print(f"✅ 数据分割完成:")
    print(f"  - 训练集: {len(train_prompts)} 组合 -> {train_output}")
    print(f"  - 验证集: {len(val_prompts)} 组合 -> {val_output}")


if __name__ == "__main__":
    # 示例：创建数据分割
    create_prompt_splits(
        input_file="data_to_lora/cs/arc-challenge/arc-challenge_train_formatted.jsonl",
        train_output="Lora_Gen/data/train_prompts.jsonl",
        val_output="Lora_Gen/data/val_prompts.jsonl",
        samples_per_prompt=4,
        test_ratio=0.1
    )
