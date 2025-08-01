"""
/root/PAW/lora_adapter/src/interpo_core.py
LoRA Linear Interpolation Baseline (直接插值到 rank=16)
"""

import torch
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class InterpoCore:
    """LoRA Linear Interpolation Baseline"""

    def __init__(self, alpha: float = 0.5, rank: int = 16):
        """
        alpha: 插值系数 (0=源, 1=目标)
        rank: 强制输出的LoRA秩
        """
        self.alpha = alpha
        self.rank = rank
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.used_padding = 0
        self.used_truncation = 0

    def _resize_weight(self, weight: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        """
        使用线性插值/裁剪填充将 weight 调整到 target_shape
        """
        src_h, src_w = weight.shape
        tgt_h, tgt_w = target_shape
        resized = torch.zeros((tgt_h, tgt_w), device=self.device, dtype=torch.float32)

        # 行采样/插值
        row_idx = torch.linspace(0, src_h - 1, steps=tgt_h).long()
        # 列采样/插值
        col_idx = torch.linspace(0, src_w - 1, steps=tgt_w).long()

        resized = weight[row_idx][:, col_idx]

        if tgt_h > src_h or tgt_w > src_w:
            self.used_padding += 1
        if tgt_h < src_h or tgt_w < src_w:
            self.used_truncation += 1

        return resized

    def transfer_lora_weights(
        self,
        source_lora: Dict[str, torch.Tensor],
        target_base_weights: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """执行线性插值迁移，强制输出目标模型可加载的 rank=16 LoRA"""
        transferred_lora = {}
        total_layers = len([k for k in source_lora if k.endswith(".weight")])
        logger.info(f"开始执行LoRA线性插值，总计 {total_layers} 层")

        processed = 0
        for lora_key, lora_weight in source_lora.items():
            if not lora_key.endswith(".weight"):
                continue

            base_key = (
                lora_key.replace("base_model.model.", "")
                .replace(".lora_A", "")
                .replace(".lora_B", "")
            )
            if base_key not in target_base_weights:
                continue

            target_shape = target_base_weights[base_key].shape

            if "lora_A" in lora_key:
                # A: [r, in_dim]
                target_shape = (self.rank, target_shape[1])
            elif "lora_B" in lora_key:
                # B: [out_dim, r]
                target_shape = (target_shape[0], self.rank)
            else:
                continue

            logger.info(
                f"🔍 [Layer: {lora_key}]\n"
                f"  源LoRA维度: {tuple(lora_weight.shape)}\n"
                f"  目标LoRA维度: {target_shape}"
            )

            projected = self._resize_weight(lora_weight.float(), target_shape)

            transferred_lora[lora_key] = projected.half()
            processed += 1

            if processed % 20 == 0:
                logger.info(f"[进度] 已处理 {processed}/{total_layers}")

        logger.info(
            f"迁移完成，总共处理 {processed} 层，"
            f"截断 {self.used_truncation} 层，"
            f"填充 {self.used_padding} 层"
        )
        return transferred_lora
