#!/usr/bin/env python3
"""
LoRA-X迁移执行脚本
从Qwen2.5-1.5B迁移LoRA权重到Gemma-2-2B
"""

import argparse
import logging
import sys
from pathlib import Path

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent.parent / "src"))

from lora_x_core import LoRAXCore
from model_utils import ModelWeightLoader, save_transferred_lora

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="LoRA-X跨模型迁移")
    parser.add_argument("--source_lora", type=str, required=True,
                       help="源LoRA模型路径")
    parser.add_argument("--source_model", type=str, required=True,
                       help="源基础模型路径")
    parser.add_argument("--target_model", type=str, required=True,
                       help="目标基础模型路径")
    parser.add_argument("--output", type=str, required=True,
                       help="输出LoRA路径")
    parser.add_argument("--rank", type=int, default=320,
                       help="SVD截断秩")
    parser.add_argument("--similarity_threshold", type=float, default=0.3,
                       help="子空间相似性阈值")
    parser.add_argument("--attention_only", action="store_true",
                       help="仅迁移attention层")
    
    args = parser.parse_args()
    
    logger.info("开始LoRA-X迁移")
    logger.info(f"源LoRA: {args.source_lora}")
    logger.info(f"源模型: {args.source_model}")
    logger.info(f"目标模型: {args.target_model}")
    logger.info(f"输出路径: {args.output}")
    
    try:
        # 初始化组件
        lora_x = LoRAXCore(rank=args.rank, similarity_threshold=args.similarity_threshold)
        loader = ModelWeightLoader()
        
        # 加载源LoRA权重
        logger.info("加载源LoRA权重...")
        source_lora_weights, lora_config = loader.load_lora_weights(args.source_lora)
        
        # 确定目标层
        target_layers = None
        if args.attention_only:
            logger.info("获取attention层名称...")
            source_attn_layers = loader.get_attention_layer_names(args.source_model)
            target_attn_layers = loader.get_attention_layer_names(args.target_model)
            target_layers = source_attn_layers + target_attn_layers
        
        # 加载基础模型权重
        logger.info("加载源模型权重...")
        source_base_weights = loader.load_base_model_weights(args.source_model, target_layers)
        
        logger.info("加载目标模型权重...")
        target_base_weights = loader.load_base_model_weights(args.target_model, target_layers)
        
        # 执行迁移
        logger.info("执行LoRA-X迁移...")
        transferred_lora = lora_x.transfer_lora_weights(
            source_lora=source_lora_weights,
            target_base_weights=target_base_weights,
            source_base_weights=source_base_weights
        )
        
        if not transferred_lora:
            logger.error("迁移失败：没有成功迁移任何层")
            return False
        
        # 保存结果
        logger.info("保存迁移结果...")
        save_transferred_lora(transferred_lora, lora_config, args.output)
        
        logger.info("LoRA-X迁移完成！")
        return True
        
    except Exception as e:
        logger.error(f"迁移过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
