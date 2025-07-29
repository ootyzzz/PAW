#!/usr/bin/env python3
"""
基于Procrustes最优对齐的LoRA迁移算法
实现策略A：结构保持的LoRA迁移 + Procrustes最优对齐

核心思想：
1. 重构阶段：ΔW_source = B_source @ A_source
2. 对齐阶段：使用Procrustes找到最优变换 Q*
3. 变换阶段：ΔW_aligned = Q* @ ΔW_source @ Q*^T
4. 分解阶段：ΔW_aligned = B_target @ A_target (SVD + 截断)
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

from lora_x_core import LoRAXCore
from model_utils import ModelWeightLoader, save_transferred_lora

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_timestamp():
    """生成时间戳格式: YYMMDD_HHMMSS"""
    now = datetime.now()
    return now.strftime("%y%m%d_%H%M%S")


def infer_source_model_path(lora_path: str) -> str:
    """根据LoRA路径推断源模型路径"""
    lora_path = Path(lora_path)
    
    if lora_path.name == "final_model":
        model_name = lora_path.parent.parent.name  
    else:
        model_name = lora_path.parent.name  
    
    source_model_path = f"/root/autodl-tmp/models/{model_name}"
    return source_model_path


class ProcrustesLoRACore(LoRAXCore):
    """基于Procrustes对齐的LoRA迁移核心类"""
    
    def __init__(self, rank=64, similarity_threshold=0.1, lora_rank=16):
        super().__init__(rank, similarity_threshold)
        self.lora_rank = lora_rank
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def procrustes_alignment(self, source_matrix: torch.Tensor, target_matrix: torch.Tensor) -> torch.Tensor:
        """
        计算Procrustes最优对齐变换矩阵
        
        Args:
            source_matrix: 源矩阵 A
            target_matrix: 目标矩阵 B
            
        Returns:
            Q: 最优正交变换矩阵，使得 ||AQ - B||_F^2 最小
        """
        # 计算 A^T B
        AtB = torch.mm(source_matrix.T, target_matrix)
        
        # SVD分解: A^T B = U Σ V^T
        U, S, Vh = torch.linalg.svd(AtB, full_matrices=False)
        
        # 最优正交变换: Q* = V U^T
        Q_optimal = torch.mm(Vh.T, U.T)
        
        return Q_optimal
    
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
    
    def transfer_lora_pair(self, lora_A_source: torch.Tensor, lora_B_source: torch.Tensor,
                          source_base: torch.Tensor, target_base: torch.Tensor) -> tuple:
        """
        使用Procrustes对齐迁移一对LoRA权重
        
        Returns:
            (lora_A_target, lora_B_target): 迁移后的LoRA权重对
        """
        # 确保所有张量在同一设备上
        device = self.device
        lora_A_source = lora_A_source.to(device)
        lora_B_source = lora_B_source.to(device)
        source_base = source_base.to(device)
        target_base = target_base.to(device)
        
        # 检查维度兼容性
        if source_base.shape != target_base.shape:
            # 维度不匹配，使用简化的投影方法
            return self._handle_dimension_mismatch(lora_A_source, lora_B_source, source_base, target_base)
        
        # 步骤1: 重构完整的LoRA权重
        full_lora_source = self.reconstruct_full_lora(lora_A_source, lora_B_source)
        
        # 步骤2: 计算源模型和目标模型的子空间
        U_source, _, Vh_source = self.compute_svd_subspace(source_base)
        U_target, _, Vh_target = self.compute_svd_subspace(target_base)
        
        # 步骤3: Procrustes对齐
        try:
            # 对行空间进行对齐
            Q_row = self.procrustes_alignment(U_source, U_target)
            # 对列空间进行对齐
            Q_col = self.procrustes_alignment(Vh_source.T, Vh_target.T)
            
            # 步骤4: 应用对齐变换
            # ΔW_aligned = Q_row^T @ ΔW_source @ Q_col
            full_lora_aligned = torch.mm(torch.mm(Q_row.T, full_lora_source), Q_col)
            
            # 步骤5: 重新分解为LoRA形式
            lora_A_target, lora_B_target = self.decompose_to_lora(full_lora_aligned, self.lora_rank)
            
            return lora_A_target, lora_B_target
            
        except Exception as e:
            # 如果Procrustes对齐失败，回退到简单方法
            logger.warning(f"Procrustes对齐失败，使用简化方法: {e}")
            return self._handle_dimension_mismatch(lora_A_source, lora_B_source, source_base, target_base)
    
    def _handle_dimension_mismatch(self, lora_A_source: torch.Tensor, lora_B_source: torch.Tensor,
                                  source_base: torch.Tensor, target_base: torch.Tensor) -> tuple:
        """处理维度不匹配的情况"""
        # 获取目标维度
        target_shape = target_base.shape
        
        # 重构源LoRA
        full_lora_source = self.reconstruct_full_lora(lora_A_source, lora_B_source)
        
        # 创建目标大小的零矩阵
        full_lora_target = torch.zeros(target_shape, device=self.device, dtype=full_lora_source.dtype)
        
        # 复制可以匹配的部分
        min_rows = min(full_lora_source.shape[0], target_shape[0])
        min_cols = min(full_lora_source.shape[1], target_shape[1])
        full_lora_target[:min_rows, :min_cols] = full_lora_source[:min_rows, :min_cols]
        
        # 重新分解为LoRA形式
        lora_A_target, lora_B_target = self.decompose_to_lora(full_lora_target, self.lora_rank)
        
        return lora_A_target, lora_B_target
    
    def transfer_lora_weights(self, source_lora: dict, target_base_weights: dict, 
                            source_base_weights: dict) -> dict:
        """执行完整的LoRA权重迁移"""
        transferred_lora = {}
        stats = {
            'total_pairs': 0,
            'transferred_pairs': 0,
            'skipped_pairs': 0,
            'transferred_list': [],
            'skipped_list': [],
            'skipped_reasons': {},
            'similarity_stats': [],
            'layer_types': {},
            'processing_times': []
        }
        
        print(f"🚀 Procrustes LoRA迁移启动 - 处理{len(source_lora)}个权重")
        
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
                
                if base_key not in source_base_weights or base_key not in target_base_weights:
                    reason = "找不到对应的基础权重"
                    stats['skipped_pairs'] += 1
                    stats['skipped_list'].append(base_name)
                    stats['skipped_reasons'][base_name] = reason
                    continue
                
                source_base = source_base_weights[base_key]
                target_base = target_base_weights[base_key]
                
                # 计算相似度
                U_s, _, _ = self.compute_svd_subspace(source_base)
                U_t, _, _ = self.compute_svd_subspace(target_base)
                similarity = self.compute_subspace_similarity(U_s, U_t)
                
                # 记录统计信息
                layer_type = self._classify_layer_type(pair_info['lora_A_key'])
                stats['similarity_stats'].append({
                    'layer': base_name,
                    'similarity': similarity,
                    'layer_type': layer_type
                })
                
                if layer_type not in stats['layer_types']:
                    stats['layer_types'][layer_type] = {'total': 0, 'transferred': 0}
                stats['layer_types'][layer_type]['total'] += 1
                
                # 相似度过滤
                if similarity < self.similarity_threshold:
                    reason = f"相似度过低 ({similarity:.6f} < {self.similarity_threshold})"
                    stats['skipped_pairs'] += 1
                    stats['skipped_list'].append(base_name)
                    stats['skipped_reasons'][base_name] = reason
                    continue
                
                # 执行Procrustes迁移
                lora_A_target, lora_B_target = self.transfer_lora_pair(
                    lora_A, lora_B, source_base, target_base
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
                del lora_A_target, lora_B_target, source_base, target_base
                
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
        
        self._print_procrustes_stats(stats)
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
    
    def _print_procrustes_stats(self, stats: dict):
        """打印Procrustes迁移统计信息"""
        print(f"\n{'🎉'*20} Procrustes迁移完成统计 {'🎉'*20}")
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
        
        # 相似度统计
        if stats['similarity_stats']:
            similarities = [s['similarity'] for s in stats['similarity_stats']]
            print(f"\n📊 相似度统计:")
            print(f"  平均相似度: {sum(similarities)/len(similarities):.6f}")
            print(f"  最高相似度: {max(similarities):.6f}")
            print(f"  最低相似度: {min(similarities):.6f}")
        
        # 性能统计
        if 'avg_processing_time' in stats:
            print(f"\n⚡ 性能统计:")
            print(f"  平均处理时间: {stats['avg_processing_time']:.3f}秒/对")
            print(f"  总处理时间: {stats['total_processing_time']:.1f}秒")
        
        print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="基于Procrustes对齐的LoRA迁移脚本")
    parser.add_argument("--source_lora", type=str, required=True,
                       help="源LoRA模型路径")
    parser.add_argument("--target_model", type=str, required=True,
                       help="目标基础模型路径")
    parser.add_argument("--output_base", type=str, 
                       default="/root/autodl-tmp/shifted/procrustes",
                       help="输出基础路径")
    parser.add_argument("--rank", type=int, default=64,
                       help="SVD截断秩")
    parser.add_argument("--similarity_threshold", type=float, default=0.1,
                       help="相似性阈值 (提高以确保质量)")
    parser.add_argument("--lora_rank", type=int, default=16,
                       help="LoRA秩")
    
    args = parser.parse_args()
    
    # 生成时间戳和输出路径
    timestamp = generate_timestamp()
    output_path = os.path.join(args.output_base, timestamp)
    
    # 推断源模型路径
    source_model_path = infer_source_model_path(args.source_lora)
    
    print(f"🚀 Procrustes LoRA迁移 - 结构保持对齐算法")
    print(f"📂 源LoRA: {args.source_lora}")
    print(f"📂 源模型: {source_model_path}")
    print(f"📂 目标模型: {args.target_model}")
    print(f"📂 输出: {output_path}")
    print(f"⚙️ 参数: rank={args.rank}, threshold={args.similarity_threshold}, lora_rank={args.lora_rank}")
    
    try:
        # 检查路径
        for path in [args.source_lora, source_model_path, args.target_model]:
            if not os.path.exists(path):
                print(f"❌ 路径不存在: {path}")
                return False
        
        # 初始化组件
        lora_core = ProcrustesLoRACore(
            rank=args.rank, 
            similarity_threshold=args.similarity_threshold,
            lora_rank=args.lora_rank
        )
        loader = ModelWeightLoader()
        
        # 加载权重
        print("📥 加载LoRA权重...")
        source_lora_weights, lora_config = loader.load_lora_weights(args.source_lora)
        
        print("📥 加载源模型权重...")
        source_base_weights = loader.load_base_model_weights(source_model_path)
        
        print("📥 加载目标模型权重...")
        target_base_weights = loader.load_base_model_weights(args.target_model)
        
        # 执行迁移
        start_time = datetime.now()
        transferred_lora, stats = lora_core.transfer_lora_weights(
            source_lora=source_lora_weights,
            target_base_weights=target_base_weights,
            source_base_weights=source_base_weights
        )
        end_time = datetime.now()
        
        if not transferred_lora:
            print("❌ 迁移失败：没有成功迁移任何权重对")
            return False
        
        # 保存结果
        os.makedirs(output_path, exist_ok=True)
        save_transferred_lora(transferred_lora, lora_config, output_path)
        
        duration = (end_time - start_time).total_seconds()
        print(f"🎉 Procrustes迁移完成！用时: {duration:.1f}秒")
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