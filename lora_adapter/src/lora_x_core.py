#!/usr/bin/env python3
"""
LoRA-X核心实现模块
基于论文: LoRA-X: Bridging Foundation Models with Training-Free Cross-Model Adaptation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List, Optional
from safetensors import safe_open
import logging

logger = logging.getLogger(__name__)


class LoRAXCore:
    """LoRA-X核心算法实现"""
    
    def __init__(self, rank: int = 320, similarity_threshold: float = 0.3):
        """
        Args:
            rank: SVD截断秩，论文中推荐320
            similarity_threshold: 子空间相似性阈值
        """
        self.rank = rank
        self.similarity_threshold = similarity_threshold
        
    def compute_svd_subspace(self, weight_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算权重矩阵的SVD子空间分解 - CUDA加速版本
        
        Args:
            weight_matrix: 权重矩阵 W ∈ R^(m×n)
            
        Returns:
            U_truncated: 截断的左奇异矩阵 Ũ ∈ R^(m×r)
            S_truncated: 截断的奇异值 s̃ ∈ R^r  
            Vh_truncated: 截断的右奇异矩阵 Ṽ^T ∈ R^(r×n)
        """
        # 确保在CUDA设备上计算
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        weight_matrix = weight_matrix.to(device)
        
        # 执行SVD分解
        U, S, Vh = torch.linalg.svd(weight_matrix.float(), full_matrices=False)
        
        # 截断到指定rank
        effective_rank = min(self.rank, min(weight_matrix.shape))
        U_truncated = U[:, :effective_rank]
        S_truncated = S[:effective_rank] 
        Vh_truncated = Vh[:effective_rank, :]
        
        return U_truncated, S_truncated, Vh_truncated
    
    def compute_subspace_similarity(self, U_source: torch.Tensor, U_target: torch.Tensor) -> float:
        """
        计算子空间相似性 - CUDA加速版本
        使用Frobenius内积: ||U_s^T U_t||_F^2 / (||U_s||_F^2 * ||U_t||_F^2)
            
        Returns:
            similarity: 相似性分数 [0, 1]
        """
        # 确保在CUDA设备上计算
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        U_source = U_source.to(device)
        U_target = U_target.to(device)
        
        # 处理维度完全不匹配的情况
        if U_source.shape[0] != U_target.shape[0]:
            # 如果行维度不同，使用投影方法计算相似性
            min_rows = min(U_source.shape[0], U_target.shape[0])
            min_cols = min(U_source.shape[1], U_target.shape[1])
            
            # 截断到最小维度
            U_s = U_source[:min_rows, :min_cols]
            U_t = U_target[:min_rows, :min_cols]
        else:
            # 行维度相同，只处理列维度
            min_dim = min(U_source.shape[1], U_target.shape[1])
            U_s = U_source[:, :min_dim]
            U_t = U_target[:, :min_dim]
        
        # 计算内积矩阵
        inner_product = torch.mm(U_s.T, U_t)
        
        # 计算Frobenius范数平方作为相似性度量
        similarity = torch.norm(inner_product, p='fro')**2 / (U_s.shape[1] * U_t.shape[1])
        
        return similarity.item()
    
    def transfer_lora_weights(self, 
                            source_lora: Dict[str, torch.Tensor],
                            target_base_weights: Dict[str, torch.Tensor],
                            source_base_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        执行LoRA-X迁移
        
        Args:
            source_lora: 源模型LoRA权重字典
            target_base_weights: 目标模型基础权重  
            source_base_weights: 源模型基础权重
            
        Returns:
            transferred_lora: 迁移后的LoRA权重字典
        """
        transferred_lora = {}
        transfer_stats = {
            'total_layers': 0,
            'transferred_layers': 0, 
            'skipped_layers': []
        }
        
        # 预计算所有相似度（并行化）
        logger.info("预计算所有层的相似度...")
        similarities = self._precompute_similarities_parallel(source_lora, source_base_weights, target_base_weights)
        
        # 遍历源LoRA权重
        for lora_key in source_lora.keys():
            if not lora_key.endswith('.weight'):
                continue

            transfer_stats['total_layers'] += 1

            # 找到对应的基础权重
            base_key = self._map_lora_to_base_key(lora_key)

            if base_key not in source_base_weights or base_key not in target_base_weights:
                logger.warning(f"跳过层 {lora_key}: 找不到对应的基础权重")
                transfer_stats['skipped_layers'].append(lora_key)
                continue

            # 获取权重
            source_base = source_base_weights[base_key]
            target_base = target_base_weights[base_key]
            lora_weight = source_lora[lora_key]

            # 检查维度兼容性
            if not self._check_dimension_compatibility(source_base, target_base, lora_weight):
                logger.info(f"层 {lora_key} 维度不兼容，采用Frobenius最小化投影")
                try:
                    projected_weight = self._frobenius_projection(lora_weight, source_base, target_base)
                    transferred_lora[lora_key] = projected_weight
                    transfer_stats['transferred_layers'] += 1
                    logger.info(f"成功迁移层 {lora_key} (Frobenius投影)")
                except Exception as e:
                    logger.warning(f"Frobenius投影失败: {e}")
                    transfer_stats['skipped_layers'].append(lora_key)
                continue

            # 使用预计算的相似度
            similarity = similarities.get(base_key, 0.0)
            # 相似性过滤
            if similarity < self.similarity_threshold:
                logger.info(f"跳过层 {lora_key}: 相似性过低 ({similarity:.3f} < {self.similarity_threshold})")
                transfer_stats['skipped_layers'].append(lora_key)
                continue

            # 执行迁移
            transferred_weight = self._transfer_single_layer(lora_weight, source_base, target_base)
            transferred_lora[lora_key] = transferred_weight
            transfer_stats['transferred_layers'] += 1
            logger.info(f"成功迁移层 {lora_key}: 相似性={similarity:.3f}")
        
        self._log_transfer_stats(transfer_stats)
        return transferred_lora
    
    def _precompute_similarities_parallel(self, source_lora: Dict[str, torch.Tensor], 
                                        source_base_weights: Dict[str, torch.Tensor],
                                        target_base_weights: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """并行预计算所有层的相似度"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        similarities = {}
        
        # 收集所有需要计算的层
        valid_keys = []
        for lora_key in source_lora.keys():
            if not lora_key.endswith('.weight'):
                continue
            base_key = self._map_lora_to_base_key(lora_key)
            if base_key in source_base_weights and base_key in target_base_weights:
                valid_keys.append(base_key)
        
        # 去重
        unique_keys = list(set(valid_keys))
        logger.info(f"需要计算相似度的唯一层数: {len(unique_keys)}")
        
        # 批量计算SVD和相似度
        for i, base_key in enumerate(unique_keys):
            if i % 10 == 0:
                logger.info(f"计算进度: {i}/{len(unique_keys)}")
            
            try:
                source_base = source_base_weights[base_key].to(device)
                target_base = target_base_weights[base_key].to(device)
                
                U_s, _, _ = self.compute_svd_subspace(source_base)
                U_t, _, _ = self.compute_svd_subspace(target_base)
                similarity = self.compute_subspace_similarity(U_s, U_t)
                similarities[base_key] = similarity
            except Exception as e:
                logger.warning(f"计算相似度失败 {base_key}: {e}")
                similarities[base_key] = 0.0
        
        return similarities
    
    def _transfer_single_layer(self, 
                              lora_weight: torch.Tensor,
                              source_base: torch.Tensor, 
                              target_base: torch.Tensor) -> torch.Tensor:
        """
        单层LoRA权重迁移
        实现论文公式3: ∆W_t←s = U_t U_t^T ∆W_s V_t V_t^T
        """
        # 确保所有张量在同一设备上
        device = lora_weight.device
        target_base = target_base.to(device)
        
        # 获取目标模型子空间
        U_t, _, Vh_t = self.compute_svd_subspace(target_base)
        V_t = Vh_t.T
        
        # 检查维度兼容性，如果不兼容则使用Frobenius投影
        try:
            # 子空间投影迁移
            # ∆W_t←s = U_t U_t^T ∆W_s V_t V_t^T
            projected_weight = torch.mm(torch.mm(U_t, U_t.T), torch.mm(lora_weight, torch.mm(V_t, V_t.T)))
        except RuntimeError as e:
            if "cannot be multiplied" in str(e):
                # 维度不匹配，使用Frobenius投影
                logger.info(f"维度不匹配，使用Frobenius投影: {lora_weight.shape} vs 目标子空间")
                projected_weight = self._frobenius_projection(lora_weight, lora_weight, target_base)
            else:
                raise e
        
        return projected_weight
    
    def _map_lora_to_base_key(self, lora_key: str) -> str:
        """映射LoRA权重名到基础模型权重名"""
        # 移除LoRA特定的前缀/后缀
        # 例如: base_model.model.model.layers.0.mlp.down_proj.lora_A.weight
        # 映射为: model.layers.0.mlp.down_proj.weight
        base_key = lora_key.replace('base_model.model.', '').replace('.lora_A', '').replace('.lora_B', '')
        return base_key
    
    def _check_dimension_compatibility(self, 
                                     source_base: torch.Tensor, 
                                     target_base: torch.Tensor, 
                                     lora_weight: torch.Tensor) -> bool:
        """检查维度兼容性"""
        # 基础检查
        if source_base.dim() != 2 or target_base.dim() != 2:
            return False
        
        # LoRA权重应该与某个维度匹配
        s_shape = source_base.shape
        t_shape = target_base.shape
        l_shape = lora_weight.shape
        
        # 简化的兼容性检查
        return len(l_shape) == 2
    
    def _frobenius_projection(self, lora_weight: torch.Tensor, source_base: torch.Tensor, target_base: torch.Tensor) -> torch.Tensor:
        """
        对于维度不兼容的层，采用LoRA结构感知的投影
        保持LoRA的低秩结构 (rank=16)
        """
        # 确保所有张量在同一设备上
        device = lora_weight.device
        source_base = source_base.to(device)
        target_base = target_base.to(device)
        
        # 获取目标形状和LoRA rank
        target_shape = target_base.shape
        lora_shape = lora_weight.shape
        lora_rank = 16  # 固定rank=16
        
        # 如果形状完全匹配，直接返回
        if lora_shape == target_shape:
            return lora_weight
        
        # 判断是lora_A还是lora_B权重
        if len(lora_shape) == 2 and len(target_shape) == 2:
            if lora_shape[0] == lora_rank:
                # 这是lora_A权重: [16, input_dim] -> [16, target_input_dim]
                target_lora_shape = (lora_rank, target_shape[1])
                logger.info(f"处理lora_A权重: {lora_shape} -> {target_lora_shape}")
                
                # 使用SVD投影保持rank=16
                U, S, Vh = torch.linalg.svd(lora_weight.float(), full_matrices=False)
                # 截断到rank=16
                rank = min(lora_rank, U.shape[1], Vh.shape[0])
                U_truncated = U[:, :rank]
                S_truncated = S[:rank]
                Vh_truncated = Vh[:rank, :]
                
                # 重构到目标维度
                if target_lora_shape[1] <= lora_shape[1]:
                    # 目标维度较小，直接截断
                    projected = U_truncated @ torch.diag(S_truncated) @ Vh_truncated[:, :target_lora_shape[1]]
                else:
                    # 目标维度较大，零填充
                    projected = torch.zeros(target_lora_shape, device=device, dtype=lora_weight.dtype)
                    reconstructed = U_truncated @ torch.diag(S_truncated) @ Vh_truncated
                    projected[:, :reconstructed.shape[1]] = reconstructed
                
                return projected
                
            elif lora_shape[1] == lora_rank:
                # 这是lora_B权重: [output_dim, 16] -> [target_output_dim, 16]
                target_lora_shape = (target_shape[0], lora_rank)
                logger.info(f"处理lora_B权重: {lora_shape} -> {target_lora_shape}")
                
                # 使用SVD投影保持rank=16
                U, S, Vh = torch.linalg.svd(lora_weight.float(), full_matrices=False)
                # 截断到rank=16
                rank = min(lora_rank, U.shape[1], Vh.shape[0])
                U_truncated = U[:, :rank]
                S_truncated = S[:rank]
                Vh_truncated = Vh[:rank, :]
                
                # 重构到目标维度
                if target_lora_shape[0] <= lora_shape[0]:
                    # 目标维度较小，直接截断
                    projected = U_truncated[:target_lora_shape[0], :] @ torch.diag(S_truncated) @ Vh_truncated
                else:
                    # 目标维度较大，零填充
                    projected = torch.zeros(target_lora_shape, device=device, dtype=lora_weight.dtype)
                    reconstructed = U_truncated @ torch.diag(S_truncated) @ Vh_truncated
                    projected[:reconstructed.shape[0], :] = reconstructed
                
                return projected
            else:
                # 非标准LoRA形状，使用通用投影
                logger.warning(f"非标准LoRA形状，使用通用投影: {lora_shape} -> {target_shape}")
                projected = torch.zeros(target_shape, device=device, dtype=lora_weight.dtype)
                min_rows = min(lora_shape[0], target_shape[0])
                min_cols = min(lora_shape[1], target_shape[1])
                projected[:min_rows, :min_cols] = lora_weight[:min_rows, :min_cols]
                return projected
        else:
            # 对于其他情况，返回随机初始化的权重
            logger.warning(f"形状不兼容，使用随机初始化: {lora_shape} -> {target_shape}")
            return torch.randn(target_shape, device=device, dtype=lora_weight.dtype) * 0.01

    def _log_transfer_stats(self, stats: Dict):
        """记录迁移统计信息"""
        logger.info(f"LoRA-X迁移完成:")
        logger.info(f"  总层数: {stats['total_layers']}")
        logger.info(f"  成功迁移: {stats['transferred_layers']}")
        logger.info(f"  跳过层数: {len(stats['skipped_layers'])}")
        if stats['skipped_layers']:
            logger.info(f"  跳过的层: {stats['skipped_layers'][:5]}...")  # 只显示前5个
