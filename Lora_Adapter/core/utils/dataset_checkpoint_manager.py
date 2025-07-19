#!/usr/bin/env python3
"""
数据集Checkpoint管理器
用于按照数据集管理和查找最新的checkpoint

这是一个独立的命令行工具，帮助你管理按数据集组织的训练实验结果。
新的文件结构: experiments/cs/{dataset_name}/{timestamp_lora}/

主要功能:
1. 查看所有数据集的训练状态概览
2. 查看特定数据集的详细实验信息
3. 快速获取最新checkpoint路径(用于继续训练或推理)
4. 快速获取最终模型路径(用于部署或评估)

使用示例:
# 查看所有数据集的概览
python scripts/dataset_checkpoint_manager.py

# 查看arc-challenge数据集的详细信息
python scripts/dataset_checkpoint_manager.py --dataset arc-challenge

# 获取arc-challenge的最新checkpoint路径
python scripts/dataset_checkpoint_manager.py --latest-checkpoint arc-challenge

# 获取arc-challenge的最终训练模型路径  
python scripts/dataset_checkpoint_manager.py --latest-model arc-challenge

# 在脚本中使用(获取模型路径用于推理)
MODEL_PATH=$(python scripts/dataset_checkpoint_manager.py --latest-model arc-challenge)
python inference.py --model_path $MODEL_PATH --input test.jsonl

应用场景:
- 当你训练了多个数据集，每个数据集有多次实验时
- 需要快速找到某个数据集的最佳/最新模型
- 需要继续之前中断的训练(使用最新checkpoint)
- 需要批量处理多个数据集的训练结果
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import argparse


class DatasetCheckpointManager:
    """按数据集管理checkpoint的工具类"""
    
    def __init__(self, experiments_root: str = "./experiments"):
        self.experiments_root = Path(experiments_root)
        self.cs_dir = self.experiments_root / "cs"
        
    def list_available_datasets(self) -> List[str]:
        """列出所有可用的数据集"""
        if not self.cs_dir.exists():
            return []
            
        datasets = []
        for item in self.cs_dir.iterdir():
            if item.is_dir():
                datasets.append(item.name)
        return sorted(datasets)
    
    def list_experiments_for_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        """列出指定数据集的所有实验"""
        dataset_dir = self.cs_dir / dataset_name
        if not dataset_dir.exists():
            return []
        
        experiments = []
        for exp_dir in dataset_dir.iterdir():
            if exp_dir.is_dir():
                metadata_file = exp_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        
                        # 检查checkpoint和model文件
                        checkpoints_dir = exp_dir / "checkpoints"
                        models_dir = exp_dir / "models"
                        
                        checkpoint_count = 0
                        if checkpoints_dir.exists():
                            checkpoint_count = len(list(checkpoints_dir.glob("checkpoint_*")))
                        
                        has_final_model = False
                        if models_dir.exists():
                            has_final_model = any(models_dir.glob("*.safetensors")) or any(models_dir.glob("pytorch_model.bin"))
                        
                        experiments.append({
                            "name": exp_dir.name,
                            "path": str(exp_dir),
                            "created_at": metadata.get("created_at"),
                            "status": metadata.get("status", "unknown"),
                            "description": metadata.get("description", ""),
                            "checkpoint_count": checkpoint_count,
                            "has_final_model": has_final_model,
                            "checkpoints_dir": str(checkpoints_dir) if checkpoints_dir.exists() else None,
                            "models_dir": str(models_dir) if models_dir.exists() else None
                        })
                    except Exception as e:
                        print(f"⚠️ 读取实验元数据失败 {exp_dir}: {e}")
        
        # 按创建时间排序（最新的在前）
        experiments.sort(key=lambda x: x["created_at"] if x["created_at"] else "", reverse=True)
        return experiments
    
    def get_latest_experiment(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """获取指定数据集的最新实验"""
        experiments = self.list_experiments_for_dataset(dataset_name)
        if experiments:
            return experiments[0]  # 已按时间排序，第一个是最新的
        return None
    
    def get_latest_checkpoint_path(self, dataset_name: str) -> Optional[str]:
        """获取指定数据集最新实验的最新checkpoint路径"""
        latest_exp = self.get_latest_experiment(dataset_name)
        if not latest_exp or not latest_exp["checkpoints_dir"]:
            return None
        
        checkpoints_dir = Path(latest_exp["checkpoints_dir"])
        if not checkpoints_dir.exists():
            return None
        
        # 查找最新的checkpoint
        checkpoints = list(checkpoints_dir.glob("checkpoint_*"))
        if not checkpoints:
            return None
        
        # 按时间排序，获取最新的
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return str(checkpoints[0])
    
    def get_final_model_path(self, dataset_name: str) -> Optional[str]:
        """获取指定数据集最新实验的最终模型路径"""
        latest_exp = self.get_latest_experiment(dataset_name)
        if not latest_exp or not latest_exp["models_dir"]:
            return None
        
        models_dir = Path(latest_exp["models_dir"])
        if not models_dir.exists():
            return None
        
        # 查找模型文件
        model_files = list(models_dir.glob("*.safetensors")) + list(models_dir.glob("pytorch_model.bin"))
        if model_files:
            return str(models_dir)  # 返回模型目录路径
        
        return None
    
    def print_dataset_summary(self, dataset_name: str):
        """打印指定数据集的摘要信息"""
        experiments = self.list_experiments_for_dataset(dataset_name)
        
        print(f"\n📊 数据集: {dataset_name}")
        print(f"{'=' * 50}")
        
        if not experiments:
            print("❌ 没有找到实验")
            return
        
        print(f"📈 实验总数: {len(experiments)}")
        
        latest = experiments[0]
        print(f"\n🏆 最新实验:")
        print(f"  名称: {latest['name']}")
        print(f"  状态: {latest['status']}")
        print(f"  创建时间: {latest['created_at']}")
        print(f"  Checkpoint数: {latest['checkpoint_count']}")
        print(f"  有最终模型: {'✅' if latest['has_final_model'] else '❌'}")
        
        # 最新checkpoint路径
        latest_ckpt = self.get_latest_checkpoint_path(dataset_name)
        if latest_ckpt:
            print(f"  最新Checkpoint: {latest_ckpt}")
        
        # 最终模型路径
        final_model = self.get_final_model_path(dataset_name)
        if final_model:
            print(f"  最终模型目录: {final_model}")
        
        print(f"\n📋 所有实验:")
        for exp in experiments:
            status_emoji = {"completed": "✅", "failed": "❌", "running": "🔄"}.get(exp["status"], "❓")
            print(f"  {status_emoji} {exp['name']} ({exp['status']}) - {exp['checkpoint_count']}个checkpoints")
    
    def print_all_datasets_summary(self):
        """打印所有数据集的摘要"""
        datasets = self.list_available_datasets()
        
        print(f"\n🗂️ 所有数据集摘要")
        print(f"{'=' * 70}")
        
        if not datasets:
            print("❌ 没有找到任何数据集实验")
            return
        
        print(f"📈 数据集总数: {len(datasets)}")
        print()
        
        for dataset in datasets:
            experiments = self.list_experiments_for_dataset(dataset)
            if experiments:
                latest = experiments[0]
                status_emoji = {"completed": "✅", "failed": "❌", "running": "🔄"}.get(latest["status"], "❓")
                print(f"{status_emoji} {dataset:<15} | {len(experiments)}个实验 | 最新: {latest['name']} ({latest['status']})")
            else:
                print(f"❓ {dataset:<15} | 无有效实验")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="数据集Checkpoint管理器")
    parser.add_argument("--dataset", type=str, help="查看指定数据集的详细信息")
    parser.add_argument("--list", action="store_true", help="列出所有数据集")
    parser.add_argument("--latest-checkpoint", type=str, help="获取指定数据集的最新checkpoint路径")
    parser.add_argument("--latest-model", type=str, help="获取指定数据集的最终模型路径")
    parser.add_argument("--experiments-root", type=str, default="./experiments", help="实验根目录")
    
    args = parser.parse_args()
    
    manager = DatasetCheckpointManager(args.experiments_root)
    
    if args.list:
        manager.print_all_datasets_summary()
    elif args.dataset:
        manager.print_dataset_summary(args.dataset)
    elif args.latest_checkpoint:
        path = manager.get_latest_checkpoint_path(args.latest_checkpoint)
        if path:
            print(path)
        else:
            print(f"❌ 没有找到数据集 {args.latest_checkpoint} 的checkpoint")
    elif args.latest_model:
        path = manager.get_final_model_path(args.latest_model)
        if path:
            print(path)
        else:
            print(f"❌ 没有找到数据集 {args.latest_model} 的最终模型")
    else:
        # 默认显示所有数据集摘要
        manager.print_all_datasets_summary()


if __name__ == "__main__":
    main()
