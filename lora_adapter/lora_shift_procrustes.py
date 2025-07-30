#!/usr/bin/env python3
"""
基于简化策略的LoRA迁移算法 - 无需源模型版本
实现策略：直接LoRA迁移 + 维度适配

核心思想：
1. 重构阶段：ΔW_source = B_source @ A_source
2. 适配阶段：根据目标模型维度调整LoRA权重
3. 分解阶段：ΔW_adapted = B_target @ A_target (SVD + 截断)
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
import gc

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent / "src"))

from model_utils import ModelWeightLoader, save_transferred_lora

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_timestamp():
    """生成时间戳格式: YYMMDD_HHMMSS"""
    now = datetime.now()
    return now.strftime("%y%m%d_%H%M%S")


def extract_model_name(model_path: str) -> str:
    """从模型路径中提取模型名称"""
    model_path = Path(model_path)
    return model_path.name


def extract_dataset_name(lora_path: str) -> str:
    """从LoRA路径中提取数据集名称"""
    lora_path = Path(lora_path)
    # 从路径中查找数据集名称，通常在文件夹名中包含
    for part in lora_path.parts:
        if 'arc-challenge' in part:
            return 'arc-challenge'
        elif 'hellaswag' in part:
            return 'hellaswag'
        elif 'mmlu' in part:
            return 'mmlu'
        elif 'truthfulqa' in part:
            return 'truthfulqa'
        elif 'winogrande' in part:
            return 'winogrande'
        elif 'gsm8k' in part:
            return 'gsm8k'
    
    # 如果没有找到已知数据集，尝试从文件夹名中提取
    for part in lora_path.parts:
        if '_lora_' in part:
            # 提取 _lora_ 前面的部分作为数据集名
            dataset_part = part.split('_lora_')[0]
            if dataset_part:
                return dataset_part
    
    return 'unknown'


def generate_output_path(source_lora: str, target_model: str, output_base: str, timestamp: str) -> str:
    """生成输出路径，格式: output_base/dataset/source_model_to_target_model/timestamp"""
    dataset_name = extract_dataset_name(source_lora)
    
    # 从LoRA路径中提取源模型名称
    lora_path = Path(source_lora)
    source_model_name = None
    for part in lora_path.parts:
        if any(model in part for model in ['Llama', 'Qwen', 'gemma', 'mistral', 'phi']):
            source_model_name = part
            break
    
    if not source_model_name:
        source_model_name = "unknown"
    
    target_model_name = extract_model_name(target_model)
    
    # 构建路径: output_base/dataset/source_to_target/timestamp
    transfer_name = f"{source_model_name}_to_{target_model_name}"
    output_path = os.path.join(output_base, dataset_name, transfer_name, timestamp)
    
    return output_path


class SimpleLoRATransfer:
    """简化的LoRA迁移核心类 - 无需源模型"""
    
    def __init__(self, lora_rank=16):
        self.lora_rank = lora_rank
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def reconstruct_full_lora(self, lora_A: torch.Tensor, lora_B: torch.Tensor) -> torch.Tensor:
        """重构完整的LoRA权重矩阵"""
        return torch.mm(lora_B, lora_A)
    
    def decompose_to_lora(self, full_weight: torch.Tensor, rank: int = None) -> tuple:
        """将完整权重矩阵分解为LoRA形式"""
        if rank is None:
            rank = self.lora_rank
            
        # SVD分解
        U, S, Vh = torch.linalg.svd(full_weight.float(), full_matrices=False)
        
        # 截断到指定rank
        effective_rank = min(rank, min(full_weight.shape), len(S))
        U_truncated = U[:, :effective_rank]
        S_truncated = S[:effective_rank]
        Vh_truncated = Vh[:effective_rank, :]
        
        # 构造LoRA权重
        # LoRA_A = sqrt(S) * V^T
        # LoRA_B = U * sqrt(S)
        sqrt_S = torch.sqrt(S_truncated)
        lora_A = torch.mm(torch.diag(sqrt_S), Vh_truncated)
        lora_B = torch.mm(U_truncated, torch.diag(sqrt_S))
        
        return lora_A, lora_B
    
    def adapt_lora_dimensions(self, lora_A: torch.Tensor, lora_B: torch.Tensor, 
                             target_shape: tuple) -> tuple:
        """
        适配LoRA权重到目标维度
        
        Args:
            lora_A: 源LoRA_A权重
            lora_B: 源LoRA_B权重
            target_shape: 目标权重形状 (out_features, in_features)
            
        Returns:
            (adapted_lora_A, adapted_lora_B): 适配后的LoRA权重对
        """
        # 重构完整权重
        full_lora = self.reconstruct_full_lora(lora_A, lora_B)
        
        # 获取当前和目标维度
        current_shape = full_lora.shape
        target_out, target_in = target_shape
        
        # 创建目标大小的零矩阵
        adapted_full = torch.zeros(target_shape, device=self.device, dtype=full_lora.dtype)
        
        # 策略1: 直接复制可匹配的部分
        min_out = min(current_shape[0], target_out)
        min_in = min(current_shape[1], target_in)
        adapted_full[:min_out, :min_in] = full_lora[:min_out, :min_in]
        
        # 策略2: 如果目标维度更大，使用插值或重复填充
        if target_out > current_shape[0] or target_in > current_shape[1]:
            # 对于超出部分，使用较小的随机初始化
            if target_out > current_shape[0]:
                # 行维度扩展：使用最后几行的平均值
                if current_shape[0] > 0:
                    avg_rows = full_lora[-min(3, current_shape[0]):, :min_in].mean(dim=0, keepdim=True)
                    for i in range(min_out, target_out):
                        adapted_full[i, :min_in] = avg_rows * 0.1  # 缩小幅度
            
            if target_in > current_shape[1]:
                # 列维度扩展：使用最后几列的平均值
                if current_shape[1] > 0:
                    avg_cols = full_lora[:min_out, -min(3, current_shape[1]):].mean(dim=1, keepdim=True)
                    for j in range(min_in, target_in):
                        adapted_full[:min_out, j] = avg_cols.squeeze() * 0.1  # 缩小幅度
        
        # 重新分解为LoRA形式
        adapted_lora_A, adapted_lora_B = self.decompose_to_lora(adapted_full, self.lora_rank)
        
        return adapted_lora_A, adapted_lora_B
    
    def transfer_lora_weights(self, source_lora: dict, target_base_weights: dict) -> dict:
        """执行简化的LoRA权重迁移"""
        transferred_lora = {}
        stats = {
            'total_pairs': 0,
            'transferred_pairs': 0,
            'skipped_pairs': 0,
            'transferred_list': [],
            'skipped_list': [],
            'skipped_reasons': {},
            'layer_types': {},
            'processing_times': []
        }
        
        print(f"🚀 简化LoRA迁移启动 - 处理{len(source_lora)}个权重")
        
        # 将LoRA权重按A/B配对
        lora_pairs = self._group_lora_pairs(source_lora)
        stats['total_pairs'] = len(lora_pairs)
        
        print(f"📊 找到{len(lora_pairs)}个LoRA权重对")
        
        for i, (base_name, pair_info) in enumerate(lora_pairs.items()):
            if i % 10 == 0:
                print(f"   迁移进度: {i}/{len(lora_pairs)}")
            
            try:
                start_time = datetime.now()
                
                # 获取权重
                lora_A = pair_info['lora_A']
                lora_B = pair_info['lora_B']
                base_key = self._map_lora_to_base_key(pair_info['lora_A_key'])
                
                if base_key not in target_base_weights:
                    reason = "找不到对应的目标基础权重"
                    stats['skipped_pairs'] += 1
                    stats['skipped_list'].append(base_name)
                    stats['skipped_reasons'][base_name] = reason
                    continue
                
                target_base = target_base_weights[base_key]
                target_shape = target_base.shape
                
                # 记录统计信息
                layer_type = self._classify_layer_type(pair_info['lora_A_key'])
                if layer_type not in stats['layer_types']:
                    stats['layer_types'][layer_type] = {'total': 0, 'transferred': 0}
                stats['layer_types'][layer_type]['total'] += 1
                
                # 执行维度适配
                lora_A_target, lora_B_target = self.adapt_lora_dimensions(
                    lora_A.to(self.device), lora_B.to(self.device), target_shape
                )
                
                # 保存迁移结果
                transferred_lora[pair_info['lora_A_key']] = lora_A_target.cpu()
                transferred_lora[pair_info['lora_B_key']] = lora_B_target.cpu()
                
                stats['transferred_pairs'] += 1
                stats['transferred_list'].append(base_name)
                stats['layer_types'][layer_type]['transferred'] += 1
                
                # 记录处理时间
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                stats['processing_times'].append(processing_time)
                
                # 清理GPU内存
                del lora_A_target, lora_B_target, target_base
                
            except Exception as e:
                reason = f"迁移失败: {str(e)}"
                stats['skipped_pairs'] += 1
                stats['skipped_list'].append(base_name)
                stats['skipped_reasons'][base_name] = reason
                logger.warning(f"迁移失败 {base_name}: {e}")
            
            # 定期清理内存
            if i % 5 == 0:
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
        
        # 计算统计信息
        if stats['processing_times']:
            stats['avg_processing_time'] = sum(stats['processing_times']) / len(stats['processing_times'])
            stats['total_processing_time'] = sum(stats['processing_times'])
        
        self._print_transfer_stats(stats)
        return transferred_lora, stats
    
    def _group_lora_pairs(self, source_lora: dict) -> dict:
        """将LoRA权重按A/B配对"""
        pairs = {}
        
        for key, weight in source_lora.items():
            if not key.endswith('.weight'):
                continue
                
            if '.lora_A.' in key:
                base_name = key.replace('.lora_A.weight', '')
                if base_name not in pairs:
                    pairs[base_name] = {}
                pairs[base_name]['lora_A'] = weight
                pairs[base_name]['lora_A_key'] = key
            elif '.lora_B.' in key:
                base_name = key.replace('.lora_B.weight', '')
                if base_name not in pairs:
                    pairs[base_name] = {}
                pairs[base_name]['lora_B'] = weight
                pairs[base_name]['lora_B_key'] = key
        
        # 只保留完整的A/B对
        complete_pairs = {}
        for base_name, pair_info in pairs.items():
            if 'lora_A' in pair_info and 'lora_B' in pair_info:
                complete_pairs[base_name] = pair_info
        
        return complete_pairs
    
    def _map_lora_to_base_key(self, lora_key: str) -> str:
        """将LoRA键映射到基础模型键"""
        # 移除LoRA特定的部分
        base_key = lora_key.replace('.lora_A.weight', '.weight')
        base_key = base_key.replace('.lora_B.weight', '.weight')
        
        # 移除 base_model.model. 前缀，这是LoRA特有的
        if base_key.startswith('base_model.model.'):
            base_key = base_key[len('base_model.model.'):]
        
        return base_key
    
    def _classify_layer_type(self, layer_name: str) -> str:
        """分类层类型"""
        layer_name_lower = layer_name.lower()
        
        if 'q_proj' in layer_name_lower:
            return 'q_proj'
        elif 'k_proj' in layer_name_lower:
            return 'k_proj'
        elif 'v_proj' in layer_name_lower:
            return 'v_proj'
        elif 'o_proj' in layer_name_lower:
            return 'o_proj'
        elif 'gate_proj' in layer_name_lower:
            return 'gate_proj'
        elif 'up_proj' in layer_name_lower:
            return 'up_proj'
        elif 'down_proj' in layer_name_lower:
            return 'down_proj'
        elif 'attn' in layer_name_lower:
            return 'attention'
        elif 'mlp' in layer_name_lower:
            return 'mlp'
        else:
            return 'other'
    
    def _print_transfer_stats(self, stats: dict):
        """打印迁移统计信息"""
        print(f"\n{'🎉'*20} 简化LoRA迁移完成统计 {'🎉'*20}")
        print(f"{'='*80}")
        print(f"📊 LoRA权重对统计:")
        print(f"  总权重对数: {stats['total_pairs']}")
        print(f"  成功迁移: {stats['transferred_pairs']}")
        print(f"  跳过对数: {stats['skipped_pairs']}")
        
        if stats['transferred_pairs'] > 0:
            success_rate = (stats['transferred_pairs'] / stats['total_pairs']) * 100
            print(f"  迁移成功率: {success_rate:.1f}%")
        
        # 按层类型统计
        if stats['layer_types']:
            print(f"\n📋 按层类型统计:")
            for layer_type, type_stats in stats['layer_types'].items():
                total = type_stats['total']
                transferred = type_stats['transferred']
                rate = (transferred / total * 100) if total > 0 else 0
                print(f"  {layer_type:12s}: {transferred:2d}/{total:2d} ({rate:5.1f}%)")
        
        # 性能统计
        if 'avg_processing_time' in stats:
            print(f"\n⚡ 性能统计:")
            print(f"  平均处理时间: {stats['avg_processing_time']:.3f}秒/对")
            print(f"  总处理时间: {stats['total_processing_time']:.1f}秒")
        
        # 跳过原因统计
        if stats['skipped_reasons']:
            print(f"\n❌ 跳过原因统计:")
            reason_counts = {}
            for reason in stats['skipped_reasons'].values():
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            for reason, count in reason_counts.items():
                print(f"  {reason}: {count}个")
        
        print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="简化LoRA迁移脚本 - 无需源模型")
    parser.add_argument("--source_lora", type=str, required=True,
                       help="源LoRA模型路径")
    parser.add_argument("--target_model", type=str, required=True,
                       help="目标基础模型路径")
    parser.add_argument("--output_base", type=str,
                       default="/root/autodl-tmp/shifted",
                       help="输出基础路径")
    parser.add_argument("--lora_rank", type=int, default=16,
                       help="LoRA秩")
    
    args = parser.parse_args()
    
    # 生成时间戳和输出路径
    timestamp = generate_timestamp()
    output_path = generate_output_path(args.source_lora, args.target_model, args.output_base, timestamp)
    
    print(f"🚀 简化LoRA迁移 - 无需源模型版本")
    print(f"📂 源LoRA: {args.source_lora}")
    print(f"📂 目标模型: {args.target_model}")
    print(f"📂 输出: {output_path}")
    print(f"⚙️ 参数: lora_rank={args.lora_rank}")
    
    try:
        # 检查路径
        for path in [args.source_lora, args.target_model]:
            if not os.path.exists(path):
                print(f"❌ 路径不存在: {path}")
                return False
        
        # 初始化组件
        lora_transfer = SimpleLoRATransfer(lora_rank=args.lora_rank)
        loader = ModelWeightLoader()
        
        # 加载权重
        print("📥 加载LoRA权重...")
        source_lora_weights, lora_config = loader.load_lora_weights(args.source_lora)
        
        print("📥 加载目标模型权重...")
        target_base_weights = loader.load_base_model_weights(args.target_model)
        
        # 执行迁移
        start_time = datetime.now()
        transferred_lora, stats = lora_transfer.transfer_lora_weights(
            source_lora=source_lora_weights,
            target_base_weights=target_base_weights
        )
        end_time = datetime.now()
        
        if not transferred_lora:
            print("❌ 迁移失败：没有成功迁移任何权重对")
            return False
        
        # 保存结果
        os.makedirs(output_path, exist_ok=True)
        save_transferred_lora(transferred_lora, lora_config, output_path)
        
        duration = (end_time - start_time).total_seconds()
        print(f"🎉 简化LoRA迁移完成！用时: {duration:.1f}秒")
        print(f"📂 结果: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ 迁移失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)