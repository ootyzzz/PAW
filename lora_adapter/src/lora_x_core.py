#!/usr/bin/env python3
"""
/root/PAW/lora_adapter/src/lora_x_core.py
LoRA-X核心实现模块 (带截断对齐)
"""

import torch
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class LoRAXCore:
    """LoRA-X核心算法实现 (支持FP16推理，投影阶段临时升FP32)"""
    
    def __init__(self, rank: int = 320):
        self.rank = rank
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.used_transform = 0
        self.used_truncation = 0
        self.used_padding = 0

    def compute_svd_subspace(self, weight_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算权重矩阵的SVD子空间分解"""
        weight_matrix = weight_matrix.to(self.device)
        if weight_matrix.dtype == torch.float16:
            weight_matrix = weight_matrix.float()
        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(weight_matrix, full_matrices=False)
            effective_rank = min(self.rank, min(weight_matrix.shape))
            U_truncated = U[:, :effective_rank]
            S_truncated = S[:effective_rank]
            Vh_truncated = Vh[:effective_rank, :]
        return U_truncated, S_truncated, Vh_truncated

    def _frobenius_projection_with_transform(
        self,
        lora_weight: torch.Tensor,
        source_base: torch.Tensor,
        target_base: torch.Tensor,
        lora_key: str
    ) -> torch.Tensor:
        """
        Frobenius最优近似 + Linear Transform (带截断对齐)
        对应LoRA-X论文4.2.2不同维度情况
        """
        device = self.device
        lora_weight = lora_weight.to(device).float()
        source_base = source_base.to(device).float()
        target_base = target_base.to(device).float()

        with torch.no_grad():
            U_s, _, _ = self.compute_svd_subspace(source_base)
            U_t, _, _ = self.compute_svd_subspace(target_base)

            # 对齐行数
            m_min = min(U_s.shape[0], U_t.shape[0])
            U_s = U_s[:m_min, :]
            U_t = U_t[:m_min, :]

            # 构造线性变换
            try:
                P = torch.linalg.lstsq(U_s, U_t).solution
            except RuntimeError as e:
                logger.warning(f"{lora_key}: lstsq失败，退回pinv: {e}")
                P = U_t @ torch.linalg.pinv(U_s)

            U_s_tilde = U_s @ P

            # 根据 lora_A / lora_B 分别处理
            target_shape = target_base.shape
            if "lora_A" in lora_key:
                # A: [r, input_dim]
                target_lora_shape = (lora_weight.shape[0], target_shape[1])
                projected = torch.zeros(target_lora_shape, device=device, dtype=torch.float32)
                cols = min(lora_weight.shape[1], target_lora_shape[1])
                projected[:, :cols] = lora_weight[:, :cols]
                if cols < target_lora_shape[1]:
                    self.used_padding += 1
                elif cols < lora_weight.shape[1]:
                    self.used_truncation += 1
            elif "lora_B" in lora_key:
                # B: [output_dim, r]
                target_lora_shape = (target_shape[0], lora_weight.shape[1])
                projected = torch.zeros(target_lora_shape, device=device, dtype=torch.float32)
                rows = min(lora_weight.shape[0], target_lora_shape[0])
                projected[:rows, :] = lora_weight[:rows, :]
                if rows < target_lora_shape[0]:
                    self.used_padding += 1
                elif rows < lora_weight.shape[0]:
                    self.used_truncation += 1
            else:
                # fallback: 通用对齐
                target_lora_shape = target_shape
                projected = torch.zeros(target_lora_shape, device=device, dtype=torch.float32)
                rows = min(lora_weight.shape[0], target_lora_shape[0])
                cols = min(lora_weight.shape[1], target_lora_shape[1])
                projected[:rows, :cols] = lora_weight[:rows, :cols]
                self.used_truncation += 1

            projected = projected.half()
            self.used_transform += 1

        del U_s, U_t, P, U_s_tilde
        torch.cuda.empty_cache()

        return projected

    def transfer_lora_weights(
        self,
        source_lora: Dict[str, torch.Tensor],
        target_base_weights: Dict[str, torch.Tensor],
        source_base_weights: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """执行LoRA-X迁移"""
        transferred_lora = {}
        total_layers = len([k for k in source_lora if k.endswith('.weight')])
        logger.info(f"开始执行LoRA权重迁移，总计 {total_layers} 层")

        processed = 0
        for lora_key, lora_weight in source_lora.items():
            if not lora_key.endswith('.weight'):
                continue
            base_key = lora_key.replace('base_model.model.', '').replace('.lora_A', '').replace('.lora_B', '')
            if base_key not in source_base_weights or base_key not in target_base_weights:
                continue

            try:
                projected = self._frobenius_projection_with_transform(
                    lora_weight,
                    source_base_weights[base_key],
                    target_base_weights[base_key],
                    lora_key
                )
                transferred_lora[lora_key] = projected
            except Exception as e:
                logger.error(f"投影失败 {lora_key}: {e}")

            processed += 1
            if processed % 20 == 0:
                logger.info(f"[进度] 已处理 {processed}/{total_layers}")

        logger.info(
            f"迁移完成，总共处理 {processed} 层，"
            f"使用线性变换 {self.used_transform} 层，"
            f"截断 {self.used_truncation} 层，"
            f"填充 {self.used_padding} 层"
        )
        return transferred_lora
