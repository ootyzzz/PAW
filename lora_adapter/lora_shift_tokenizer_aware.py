#!/usr/bin/env python3
"""
Tokenizer-Aware LoRA迁移算法
实现基于tokenizer对齐的子空间映射 + Procrustes最优对齐

核心思想：
1. 分析tokenizer差异，建立token映射关系
2. 对每个layer pair，先进行tokenizer-aware的映射
3. 然后使用Procrustes最优对齐算法
4. 最终实现tokenizer-aware的子空间对齐
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
from transformers import AutoTokenizer

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent / "tokenlizer"))

from lora_x_core import LoRAXCore
from model_utils import ModelWeightLoader, save_transferred_lora
from tokenizer_analyzer import TokenizerAnalyzer

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


class TokenizerAwareLoRACore(LoRAXCore):
    """基于Tokenizer感知的LoRA迁移核心类"""
    
    def __init__(self, rank=64, similarity_threshold=0.1, lora_rank=16, 
                 source_tokenizer_path=None, target_tokenizer_path=None):
        super().__init__(rank, similarity_threshold)
        self.lora_rank = lora_rank
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化tokenizer分析器
        if source_tokenizer_path and target_tokenizer_path:
            print("🔍 初始化Tokenizer分析器...")
            self.tokenizer_analyzer = TokenizerAnalyzer(source_tokenizer_path, target_tokenizer_path)
            self.token_mapping = self._build_token_mapping()
            print(f"📊 Token映射建立完成: {len(self.token_mapping)}个映射")
        else:
            self.tokenizer_analyzer = None
            self.token_mapping = None
            print("⚠️ 未提供tokenizer路径，将使用标准Procrustes对齐")
    
    def _build_token_mapping(self) -> dict:
        """建立从源tokenizer到目标tokenizer的映射"""
        source_vocab = self.tokenizer_analyzer.qwen_tokenizer.get_vocab()
        target_vocab = self.tokenizer_analyzer.llama_tokenizer.get_vocab()
        
        mapping = {}
        
        # 方法1: 精确匹配
        for src_token, src_id in source_vocab.items():
            if src_token in target_vocab:
                mapping[src_id] = target_vocab[src_token]
        
        print(f"  精确匹配: {len(mapping)}个token")
        
        # 方法2: 相似度匹配（为未匹配的token）
        unmatched_count = 0
        for src_token, src_id in source_vocab.items():
            if src_id not in mapping:
                # 寻找最相似的token
                best_match = self._find_similar_token(src_token, target_vocab)
                if best_match:
                    mapping[src_id] = target_vocab[best_match]
                    unmatched_count += 1
                else:
                    # 映射到UNK token
                    mapping[src_id] = target_vocab.get('<unk>', 0)
        
        print(f"  相似度匹配: {unmatched_count}个token")
        print(f"  总映射覆盖率: {len(mapping)/len(source_vocab)*100:.1f}%")
        
        return mapping
    
    def _find_similar_token(self, source_token: str, target_vocab: dict) -> str:
        """寻找最相似的token"""
        # 简单的字符串相似度匹配
        best_match = None
        best_score = 0
        
        for target_token in target_vocab.keys():
            # 计算编辑距离相似度
            score = self._string_similarity(source_token, target_token)
            if score > best_score and score > 0.7:  # 相似度阈值
                best_score = score
                best_match = target_token
        
        return best_match
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """计算字符串相似度"""
        if not s1 or not s2:
            return 0.0
        
        # 简单的Jaccard相似度
        set1 = set(s1.lower())
        set2 = set(s2.lower())
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def tokenizer_aware_alignment(self, source_embedding: torch.Tensor, 
                                target_embedding: torch.Tensor) -> torch.Tensor:
        """
        基于tokenizer映射的嵌入对齐
        
        Args:
            source_embedding: 源模型的嵌入表示 [seq_len, hidden_dim]
            target_embedding: 目标模型的嵌入表示 [seq_len, hidden_dim]
            
        Returns:
            aligned_embedding: 对齐后的嵌入表示
        """
        if self.token_mapping is None:
            # 如果没有token映射，直接返回目标嵌入
            return target_embedding
        
        # 创建对齐后的嵌入矩阵
        aligned_embedding = torch.zeros_like(target_embedding)
        
        # 应用token映射
        for src_id, tgt_id in self.token_mapping.items():
            if src_id < source_embedding.shape[0] and tgt_id < target_embedding.shape[0]:
                # 使用加权平均来融合源和目标的表示
                alpha = 0.7  # 源模型权重
                aligned_embedding[tgt_id] = (alpha * source_embedding[src_id] + 
                                           (1-alpha) * target_embedding[tgt_id])
        
        return aligned_embedding
    
    def procrustes_alignment(self, source_matrix: torch.Tensor, target_matrix: torch.Tensor) -> torch.Tensor:
        """
        计算Procrustes最优对齐变换矩阵
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
        sqrt_S = torch.sqrt(S_truncated)
        lora_A = torch.mm(torch.diag(sqrt_S), Vh_truncated)
        lora_B = torch.mm(U_truncated, torch.diag(sqrt_S))
        
        return lora_A, lora_B
    
    def tokenizer_aware_transfer_lora_pair(self, lora_A_source: torch.Tensor, lora_B_source: torch.Tensor,
                                         source_base: torch.Tensor, target_base: torch.Tensor) -> tuple:
        """
        使用Tokenizer-Aware + Procrustes对齐迁移一对LoRA权重
        
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
            return self._handle_dimension_mismatch(lora_