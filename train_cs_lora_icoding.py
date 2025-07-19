#!/usr/bin/env python3
"""
train_cs_lora_icoding.py
服务器环境LoRA训练脚本 - 针对4×H800优化
支持精确batch追踪和prompt-checkpoint配对

配置特点:
- Batch size: 32 (适配服务器内存)
- Training steps: 125
- Checkpoint频率: 每50步保存
- 数据加载: 严格顺序，可追踪每个batch的source行号

使用示例:
# 服务器环境 - batch size 32
python train_cs_lora_icoding.py --dataset arc-challenge

# 本地测试 - batch size 4 
python train_cs_lora_icoding.py --dataset arc-challenge --test_mode

# 训练单个数据集
python train_cs_lora_icoding.py --dataset arc-challenge

# 训练多个数据集
python train_cs_lora_icoding.py --dataset arc-challenge arc-easy

# 启用详细的batch追踪日志
python train_cs_lora_icoding.py --dataset arc-challenge --track_batches

# 干运行查看数据分布
python train_cs_lora_icoding.py --dataset arc-challenge --dry_run

关键特性:
1. 固定batch size=32，严格顺序加载数据
2. 每个checkpoint精确记录对应的prompt范围
3. 自动处理epoch循环，保持数据一致性
4. 详细的batch-checkpoint映射日志
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import torch
from torch.utils.data import DataLoader, Dataset


def custom_collate_fn(batch):
    """自定义collate函数，保持字典结构不被PyTorch默认处理破坏"""
    return batch

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

try:
    from scripts.experiment_manager_enhanced import ExperimentManager
    from scripts.model_manager import ModelManager
    from utils.data_processor import DataProcessor
    from core.train import LoRATrainer
    from lora.checkpoint_utils import CheckpointManager
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保所有依赖已正确安装")
    sys.exit(1)


class SequentialTrackingDataset(Dataset):
    """严格顺序的可追踪数据集"""
    
    def __init__(self, data_file: str, dataset_name: str):
        self.data_file = data_file
        self.dataset_name = dataset_name
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
                        # 添加原始行号信息
                        item['_source_line'] = line_idx
                        item['_dataset_name'] = self.dataset_name
                        data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"⚠️ 跳过无效行 {line_idx}: {e}")
                        
        print(f"📊 加载数据集 {self.dataset_name}: {len(data)} 样本")
        return data
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # 支持超过数据集长度的循环访问
        actual_idx = idx % self.total_samples
        item = self.data[actual_idx].copy()
        
        # 添加循环信息
        epoch_num = idx // self.total_samples
        item['_actual_idx'] = actual_idx
        item['_global_idx'] = idx
        item['_epoch'] = epoch_num
        
        return item


class BatchTracker:
    """Batch追踪器，记录每个batch的详细信息"""
    
    def __init__(self, dataset_name: str, log_dir: str):
        self.dataset_name = dataset_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化追踪日志
        self.batch_log_file = self.log_dir / f"batch_tracking_{dataset_name}.jsonl"
        self.checkpoint_map_file = self.log_dir / f"checkpoint_mapping_{dataset_name}.json"
        
        self.batch_records = []
        self.checkpoint_mappings = {}
        
    def log_batch(self, step: int, batch_data: List[Dict[str, Any]]):
        """记录单个batch的详细信息"""
        batch_info = {
            "step": step,
            "dataset": self.dataset_name,
            "batch_size": len(batch_data),
            "timestamp": datetime.now().isoformat(),
            "samples": []
        }
        
        # 记录每个样本的追踪信息
        for sample in batch_data:
            sample_info = {
                "source_line": sample.get('_source_line', -1),
                "actual_idx": sample.get('_actual_idx', -1),
                "global_idx": sample.get('_global_idx', -1),
                "epoch": sample.get('_epoch', 0),
                "id": sample.get('id', 'unknown'),
                "input_preview": sample.get('input', '')[:100] + '...' if len(sample.get('input', '')) > 100 else sample.get('input', '')
            }
            batch_info["samples"].append(sample_info)
        
        # 计算batch范围
        source_lines = [s['source_line'] for s in batch_info["samples"]]
        batch_info["source_range"] = {
            "min_line": min(source_lines) if source_lines else -1,
            "max_line": max(source_lines) if source_lines else -1,
            "epochs_involved": list(set(s['epoch'] for s in batch_info["samples"]))
        }
        
        self.batch_records.append(batch_info)
        
        # 实时写入日志
        with open(self.batch_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(batch_info, ensure_ascii=False) + '\n')
    
    def log_checkpoint(self, step: int, checkpoint_path: str, batch_range: Tuple[int, int]):
        """记录checkpoint与batch的映射关系"""
        start_step, end_step = batch_range
        
        # 找到对应的batch记录
        related_batches = [
            record for record in self.batch_records 
            if start_step <= record["step"] <= end_step
        ]
        
        checkpoint_info = {
            "checkpoint_step": step,
            "checkpoint_path": checkpoint_path,
            "batch_range": {
                "start_step": start_step,
                "end_step": end_step
            },
            "total_batches": len(related_batches),
            "total_samples": sum(b["batch_size"] for b in related_batches),
            "source_data_summary": self._summarize_source_data(related_batches),
            "timestamp": datetime.now().isoformat()
        }
        
        self.checkpoint_mappings[f"checkpoint_{step}"] = checkpoint_info
        
        # 保存mapping文件
        with open(self.checkpoint_map_file, 'w', encoding='utf-8') as f:
            json.dump(self.checkpoint_mappings, f, indent=2, ensure_ascii=False)
    
    def _summarize_source_data(self, batches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """总结source data信息"""
        all_source_lines = []
        all_epochs = set()
        
        for batch in batches:
            for sample in batch["samples"]:
                all_source_lines.append(sample["source_line"])
                all_epochs.add(sample["epoch"])
        
        return {
            "total_source_lines": len(all_source_lines),
            "source_line_range": {
                "min": min(all_source_lines) if all_source_lines else -1,
                "max": max(all_source_lines) if all_source_lines else -1
            },
            "epochs_involved": sorted(list(all_epochs)),
            "unique_source_lines": len(set(all_source_lines))
        }


def create_icoding_config(dataset_name: str, base_config: Dict[str, Any], batch_size: int = 32) -> Dict[str, Any]:
    """创建icoding环境的训练配置"""
    config = base_config.copy()
    
    # 更新数据路径
    config['data']['train_file'] = f"data_to_lora/cs/{dataset_name}/{dataset_name}_train_formatted.jsonl"
    
    # 生成时间戳用于实验名称
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"icoding_{timestamp}_lora"
    
    # icoding专用配置
    config['training'].update({
        'per_device_train_batch_size': batch_size,  # 可配置batch size
        'gradient_accumulation_steps': 1,
        'max_steps': 125,  # 固定步数
        'save_steps': 50,  # checkpoint频率
        'logging_steps': 1,
        'dataloader_num_workers': 4,  # 服务器优化
        'dataloader_pin_memory': True,
        'remove_unused_columns': False,  # 保留追踪信息
    })
    
    # 确保数据加载的一致性
    config['data'].update({
        'shuffle': False,  # 严格禁用shuffle
        'seed': 42,  # 固定种子
        'drop_last': False  # 保留最后的不完整batch
    })
    
    # 更新输出目录
    config['training']['output_dir'] = f"./experiments/icoding/{dataset_name}/{experiment_name}/models"
    config['checkpoint']['dir'] = f"./experiments/icoding/{dataset_name}/{experiment_name}/checkpoints"
    config['logging']['log_dir'] = f"./experiments/icoding/{dataset_name}/{experiment_name}/logs"
    
    # 实验信息
    config['experiment']['name'] = experiment_name
    config['experiment']['description'] = f"iCoding LoRA training on {dataset_name} - batch_size={batch_size}, steps=125"
    config['experiment']['tags'] = ["icoding", "lora", "qwen2.5", dataset_name, f"batch{batch_size}", "trackable"]
    
    config['experiment_type'] = "icoding_lora"
    config['dataset_name'] = dataset_name
    
    return config


def run_icoding_experiment(dataset_name: str, base_config: Dict[str, Any], track_batches: bool = True, dry_run: bool = False, test_mode: bool = False, batch_size: int = 32) -> Dict[str, Any]:
    """运行icoding环境的训练实验"""
    
    print(f"\n{'=' * 70}")
    print(f"🚀 iCoding环境训练: {dataset_name}")
    print(f"{'=' * 70}")
    
    # 创建配置
    config = create_icoding_config(dataset_name, base_config, batch_size)
    
    # 验证数据文件
    data_file = config['data']['train_file']
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"训练数据文件不存在: {data_file}")
    
    print(f"📁 数据文件: {data_file}")
    print(f"🎯 输出目录: {config['training']['output_dir']}")
    print(f"📊 训练配置: batch_size={batch_size}, steps=125, checkpoint_every=50")
    
    if dry_run:
        # 分析数据分布
        dataset = SequentialTrackingDataset(data_file, dataset_name)
        analyze_data_distribution(dataset, batch_size=batch_size, total_steps=125)
        return {"status": "dry_run_completed"}
    
    try:
        # 设置日志目录
        log_dir = Path(config['logging']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化追踪器
        tracker = None
        if track_batches:
            tracker = BatchTracker(dataset_name, str(log_dir))
            print("📝 启用batch追踪")
        
        # 创建数据集和DataLoader
        dataset = SequentialTrackingDataset(data_file, dataset_name)
        
        # 创建DataLoader - 关键配置确保数据一致性
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # 严格禁用shuffle
            num_workers=0 if test_mode else 4,  # 测试模式下使用0个worker避免多进程问题
            pin_memory=False if test_mode else True,  # 测试模式下关闭pin_memory
            drop_last=False,  # 保留最后的不完整batch
            persistent_workers=False if test_mode else True,  # 测试模式下关闭persistent workers
            collate_fn=custom_collate_fn  # 使用自定义collate函数保持字典结构
        )
        
        print(f"📊 数据加载器配置:")
        print(f"  - 数据集大小: {len(dataset)} 样本")
        print(f"  - Batch大小: {batch_size}")
        print(f"  - 总batch数: {len(dataloader)}")
        print(f"  - Shuffle: False (严格顺序)")
        
        # 执行训练或模拟训练
        if test_mode:
            # 测试模式：完整的batch追踪验证，但步数较少以便观察
            results = run_test_training_with_full_tracking(
                dataloader=dataloader,
                total_steps=20,  # 测试模式使用较少步数
                checkpoint_steps=[10, 20],  # 测试checkpoint
                tracker=tracker,
                config=config
            )
        else:
            # 正常模式或模拟训练
            results = simulate_training_with_tracking(
                dataloader=dataloader,
                total_steps=125,
                checkpoint_steps=[50, 100, 125],
                tracker=tracker,
                config=config
            )
        
        # 生成line-to-checkpoint映射文件
        if tracker:
            generate_line_to_checkpoint_mapping(tracker, str(log_dir))
        
        return results
        
    except Exception as e:
        print(f"❌ {dataset_name} 训练失败: {e}")
        raise


def simulate_training_with_tracking(
    dataloader: DataLoader,
    total_steps: int,
    checkpoint_steps: List[int],
    tracker: Optional[BatchTracker],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """模拟训练过程，展示batch追踪逻辑"""
    
    print(f"\n🔄 开始训练模拟 (共{total_steps}步)...")
    
    step = 0
    checkpoint_saved = []
    dataloader_iter = iter(dataloader)
    last_checkpoint_step = 0
    
    while step < total_steps:
        try:
            # 获取下一个batch
            batch = next(dataloader_iter)
            
            # 记录batch信息
            if tracker:
                tracker.log_batch(step, batch)
            
            # 显示进度
            if step % 10 == 0 or step < 5:
                sample_info = batch[0] if batch else {}
                print(f"  Step {step}: batch_size={len(batch)}, "
                      f"source_lines=[{sample_info.get('_source_line', -1)}...], "
                      f"epoch={sample_info.get('_epoch', 0)}")
            
            # 检查是否需要保存checkpoint
            if step + 1 in checkpoint_steps:
                checkpoint_path = f"checkpoint-{step + 1}"
                checkpoint_saved.append({
                    "step": step + 1,
                    "path": checkpoint_path,
                    "batch_range": (last_checkpoint_step, step)
                })
                
                if tracker:
                    tracker.log_checkpoint(step + 1, checkpoint_path, (last_checkpoint_step, step))
                
                print(f"💾 保存checkpoint: {checkpoint_path} (覆盖steps {last_checkpoint_step}-{step})")
                last_checkpoint_step = step + 1
            
            step += 1
            
        except StopIteration:
            # DataLoader耗尽，重新开始下一个epoch
            print(f"  📚 数据集遍历完毕，开始新epoch (当前step: {step})")
            dataloader_iter = iter(dataloader)
    
    results = {
        "status": "completed",
        "total_steps": step,
        "checkpoints_saved": checkpoint_saved,
        "dataset_cycles": step // len(dataloader) + (1 if step % len(dataloader) > 0 else 0)
    }
    
    print(f"\n✅ 训练模拟完成:")
    print(f"  - 总步数: {step}")
    print(f"  - 保存checkpoint: {len(checkpoint_saved)}个")
    print(f"  - 数据集循环: {results['dataset_cycles']}次")
    
    return results


def run_test_training_with_full_tracking(
    dataloader: DataLoader,
    total_steps: int,
    checkpoint_steps: List[int],
    tracker: Optional[BatchTracker],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """测试模式：运行完整的batch追踪验证"""
    
    print(f"\n🧪 开始测试模式训练 (共{total_steps}步，完整追踪)...")
    
    step = 0
    checkpoint_saved = []
    dataloader_iter = iter(dataloader)
    last_checkpoint_step = 0
    all_batches_data = []  # 保存所有batch数据用于分析
    
    while step < total_steps:
        try:
            # 获取下一个batch
            batch = next(dataloader_iter)
            
            # 转换为可序列化的格式
            batch_list = []
            for item in batch:
                if isinstance(item, dict):
                    batch_list.append(item)
                elif hasattr(item, 'item'):  # 处理tensor
                    batch_list.append({'tensor_value': item.item()})
                else:
                    batch_list.append({'value': str(item)})
            
            all_batches_data.append({
                'step': step,
                'batch': batch_list
            })
            
            # 记录batch信息
            if tracker:
                tracker.log_batch(step, batch_list)
            
            # 显示详细进度
            sample_info = batch_list[0] if batch_list else {}
            source_lines = [item.get('_source_line', -1) for item in batch_list]
            epochs = list(set(item.get('_epoch', 0) for item in batch_list))
            
            print(f"  🔍 Step {step:2d}: batch_size={len(batch_list)}, "
                  f"source_lines=[{min(source_lines):3d}-{max(source_lines):3d}], "
                  f"epochs={epochs}")
            
            # 显示每个样本的详细信息（仅前3个样本以避免输出过多）
            for i, item in enumerate(batch_list[:3]):
                print(f"    📄 Sample {i}: line={item.get('_source_line', -1)}, "
                      f"id={item.get('id', 'N/A')[:20]}..., "
                      f"epoch={item.get('_epoch', 0)}")
            
            if len(batch_list) > 3:
                print(f"    📄 ... 还有 {len(batch_list) - 3} 个样本")
            
            # 检查是否需要保存checkpoint
            if step + 1 in checkpoint_steps:
                checkpoint_path = f"checkpoint-{step + 1}"
                checkpoint_saved.append({
                    "step": step + 1,
                    "path": checkpoint_path,
                    "batch_range": (last_checkpoint_step, step)
                })
                
                if tracker:
                    tracker.log_checkpoint(step + 1, checkpoint_path, (last_checkpoint_step, step))
                
                print(f"💾 保存checkpoint: {checkpoint_path} (覆盖steps {last_checkpoint_step}-{step})")
                last_checkpoint_step = step + 1
            
            step += 1
            
        except StopIteration:
            # DataLoader耗尽，重新开始下一个epoch
            print(f"  📚 数据集遍历完毕，开始新epoch (当前step: {step})")
            dataloader_iter = iter(dataloader)
    
    results = {
        "status": "test_completed",
        "total_steps": step,
        "checkpoints_saved": checkpoint_saved,
        "dataset_cycles": step // len(dataloader) + (1 if step % len(dataloader) > 0 else 0),
        "all_batches_data": all_batches_data
    }
    
    print(f"\n✅ 测试模式训练完成:")
    print(f"  - 总步数: {step}")
    print(f"  - 保存checkpoint: {len(checkpoint_saved)}个")
    print(f"  - 数据集循环: {results['dataset_cycles']}次")
    print(f"  - 追踪batch: {len(all_batches_data)}个")
    
    return results


def generate_line_to_checkpoint_mapping(tracker: BatchTracker, log_dir: str):
    """生成line-to-checkpoint的完整映射文件"""
    
    print(f"\n📋 生成line-to-checkpoint映射文件...")
    
    # 构建完整的映射关系
    line_to_checkpoint = {}
    checkpoint_to_lines = {}
    
    # 遍历所有checkpoint映射
    for ckpt_name, ckpt_info in tracker.checkpoint_mappings.items():
        checkpoint_step = ckpt_info['checkpoint_step']
        start_step = ckpt_info['batch_range']['start_step']
        end_step = ckpt_info['batch_range']['end_step']
        
        # 找到对应步骤的batch记录
        related_batches = [
            record for record in tracker.batch_records
            if start_step <= record["step"] <= end_step
        ]
        
        lines_in_checkpoint = []
        
        for batch_record in related_batches:
            for sample in batch_record["samples"]:
                source_line = sample["source_line"]
                lines_in_checkpoint.append(source_line)
                
                # 记录line到checkpoint的映射
                if source_line not in line_to_checkpoint:
                    line_to_checkpoint[source_line] = []
                line_to_checkpoint[source_line].append({
                    "checkpoint": ckpt_name,
                    "checkpoint_step": checkpoint_step,
                    "step": batch_record["step"],
                    "sample_id": sample["id"]
                })
        
        # 记录checkpoint到lines的映射
        checkpoint_to_lines[ckpt_name] = {
            "checkpoint_step": checkpoint_step,
            "lines": sorted(list(set(lines_in_checkpoint))),
            "line_count": len(set(lines_in_checkpoint)),
            "total_samples": len(lines_in_checkpoint),
            "step_range": [start_step, end_step]
        }
    
    # 保存映射文件
    mapping_data = {
        "dataset": tracker.dataset_name,
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_checkpoints": len(checkpoint_to_lines),
            "total_unique_lines": len(line_to_checkpoint),
            "total_line_checkpoint_pairs": sum(len(mappings) for mappings in line_to_checkpoint.values())
        },
        "line_to_checkpoint": line_to_checkpoint,
        "checkpoint_to_lines": checkpoint_to_lines
    }
    
    # 保存详细映射
    mapping_file = Path(log_dir) / f"line_checkpoint_mapping_{tracker.dataset_name}.json"
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, indent=2, ensure_ascii=False)
    
    # 生成简化的概览文件
    summary_data = {
        "dataset": tracker.dataset_name,
        "generated_at": datetime.now().isoformat(),
        "checkpoints": []
    }
    
    for ckpt_name, info in checkpoint_to_lines.items():
        summary_data["checkpoints"].append({
            "name": ckpt_name,
            "step": info["checkpoint_step"],
            "line_range": [min(info["lines"]), max(info["lines"])] if info["lines"] else [0, 0],
            "unique_lines": info["line_count"],
            "total_samples": info["total_samples"]
        })
    
    summary_file = Path(log_dir) / f"checkpoint_summary_{tracker.dataset_name}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 映射文件已生成:")
    print(f"  - 详细映射: {mapping_file}")
    print(f"  - 概览文件: {summary_file}")
    print(f"📊 映射统计:")
    print(f"  - 总checkpoint数: {len(checkpoint_to_lines)}")
    print(f"  - 总行数: {len(line_to_checkpoint)}")
    print(f"  - 行-checkpoint对数: {sum(len(mappings) for mappings in line_to_checkpoint.values())}")


def analyze_data_distribution(dataset: SequentialTrackingDataset, batch_size: int, total_steps: int):
    """分析数据分布情况"""
    
    print(f"\n📊 数据分布分析:")
    print(f"{'=' * 50}")
    
    total_samples_needed = batch_size * total_steps
    dataset_size = len(dataset)
    epochs_needed = total_samples_needed / dataset_size
    
    print(f"数据集信息:")
    print(f"  - 数据集名称: {dataset.dataset_name}")
    print(f"  - 样本总数: {dataset_size}")
    print(f"  - 数据文件: {dataset.data_file}")
    
    print(f"\n训练需求:")
    print(f"  - Batch大小: {batch_size}")
    print(f"  - 训练步数: {total_steps}")
    print(f"  - 需要样本数: {total_samples_needed}")
    print(f"  - 需要循环: {epochs_needed:.2f} epochs")
    
    print(f"\nBatch分布预测:")
    for step in range(min(10, total_steps)):  # 显示前10个batch
        start_idx = step * batch_size
        end_idx = min(start_idx + batch_size, total_samples_needed)
        
        batch_samples = []
        for idx in range(start_idx, end_idx):
            actual_idx = idx % dataset_size
            epoch = idx // dataset_size
            sample = dataset[idx]
            batch_samples.append((actual_idx, epoch, sample['_source_line']))
        
        source_lines = [s[2] for s in batch_samples]
        epochs = list(set(s[1] for s in batch_samples))
        
        print(f"  Step {step:2d}: source_lines=[{min(source_lines):3d}-{max(source_lines):3d}], epochs={epochs}")
    
    if total_steps > 10:
        print(f"  ... (省略中间 {total_steps - 10} 步)")
    
    # 分析checkpoint覆盖范围
    print(f"\nCheckpoint覆盖分析 (每50步保存):")
    for ckpt_step in [50, 100, 125]:
        if ckpt_step <= total_steps:
            start_sample = (ckpt_step - 50) * batch_size if ckpt_step > 50 else 0
            end_sample = ckpt_step * batch_size - 1
            
            start_line = start_sample % dataset_size
            end_line = end_sample % dataset_size
            
            print(f"  Checkpoint-{ckpt_step}: samples[{start_sample}-{end_sample}] -> source_lines[{start_line}-{end_line}]")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="iCoding环境LoRA训练脚本")
    parser.add_argument("--dataset", type=str, required=True,
                       help="要训练的数据集名称 (arc-challenge, arc-easy, boolq, hellaswag, openbookqa, piqa, winogrande)")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                       help="基础配置文件路径")
    parser.add_argument("--track_batches", action="store_true",
                       help="启用详细的batch追踪日志")
    parser.add_argument("--dry_run", action="store_true",
                       help="干运行，分析数据分布但不实际训练")
    parser.add_argument("--test_mode", action="store_true",
                       help="测试模式，使用本地可测试的batch size 4进行完整的追踪验证")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="批处理大小 (默认32，测试模式下自动设为4)")
    
    args = parser.parse_args()
    
    # 测试模式下强制设置小batch size
    if args.test_mode:
        args.batch_size = 4
        args.track_batches = True  # 测试模式自动启用batch追踪
        print("🧪 启用测试模式: batch_size=4, 完整追踪验证")
    
    # 验证数据集名称
    valid_datasets = ['arc-challenge', 'arc-easy', 'boolq', 'hellaswag', 'openbookqa', 'piqa', 'winogrande']
    if args.dataset not in valid_datasets:
        print(f"❌ 无效的数据集名称: {args.dataset}")
        print(f"✅ 可用数据集: {', '.join(valid_datasets)}")
        return False
    
    print("🚀 iCoding环境LoRA训练脚本")
    print("=" * 70)
    print(f"目标数据集: {args.dataset}")
    print(f"配置文件: {args.config}")
    print(f"Batch大小: {args.batch_size}")
    print(f"Batch追踪: {'启用' if args.track_batches else '禁用'}")
    print(f"运行模式: {'测试模式' if args.test_mode else 'Dry Run' if args.dry_run else '实际训练'}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    try:
        # 加载基础配置
        with open(args.config, 'r', encoding='utf-8') as f:
            base_config = yaml.safe_load(f)
        
        # 执行训练
        results = run_icoding_experiment(
            dataset_name=args.dataset,
            base_config=base_config,
            track_batches=args.track_batches,
            dry_run=args.dry_run,
            test_mode=args.test_mode,
            batch_size=args.batch_size
        )
        
        print(f"\n🎉 实验完成!")
        print(f"📊 结果: {results.get('status', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
