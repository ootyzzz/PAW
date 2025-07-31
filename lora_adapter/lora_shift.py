#!/usr/bin/env python3
"""
方法2 LoRA-X的粗糙模仿
用例 从Meta-Llama-3.1-8B-Instruct迁移LoRA到Qwen2.5-7B-Instruct
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import torch

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent / "src"))

from src.lora_x_core import LoRAXCore
from src.model_utils import ModelWeightLoader, save_transferred_lora

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_timestamp():
    """生成时间戳格式: YYMMDD_HHMMSS"""
    now = datetime.now()
    return now.strftime("%y%m%d_%H%M%S")

def infer_source_model_path(lora_path: str) -> str:
    lora_path = Path(lora_path)
    if lora_path.name == "final_model":
        model_name = lora_path.parent.parent.name  
    else:
        model_name = lora_path.parent.name  
    return f"/root/autodl-tmp/models/{model_name}"

def main():
    parser = argparse.ArgumentParser(description="LoRA-X Frobenius最优近似迁移")
    parser.add_argument("--source_lora", type=str, required=True,
                       help="源LoRA模型路径")
    parser.add_argument("--target_model", type=str, required=True,
                       help="目标基础模型路径")
    parser.add_argument("--output_base", type=str,
                       default="/root/autodl-tmp/shifted/arc-challenge/Llama-to-Qwen",
                       help="输出基础路径")
    parser.add_argument("--rank", type=int, default=320,
                       help="SVD截断秩")
    args = parser.parse_args()

    timestamp = generate_timestamp()
    output_path = os.path.join(args.output_base, timestamp)
    source_model_path = infer_source_model_path(args.source_lora)

    print(f"\n🚀 LoRA迁移开始 🚀")
    print("="*80)
    print(f"📂 源LoRA路径: {args.source_lora}")
    print(f"📂 源模型路径: {source_model_path}")
    print(f"📂 目标模型路径: {args.target_model}")
    print(f"📂 输出路径: {output_path}")
    print(f"⚙️  SVD截断秩: {args.rank}")
    print(f"🕐 时间戳: {timestamp}")
    print("="*80)

    try:
        if not os.path.exists(args.source_lora):
            raise FileNotFoundError(f"源LoRA路径不存在: {args.source_lora}")
        if not os.path.exists(source_model_path):
            raise FileNotFoundError(f"源模型路径不存在: {source_model_path}")
        if not os.path.exists(args.target_model):
            raise FileNotFoundError(f"目标模型路径不存在: {args.target_model}")

        lora_x = LoRAXCore(rank=args.rank)
        loader = ModelWeightLoader()

        print("\n📥 加载源LoRA权重...")
        source_lora_weights, lora_config = loader.load_lora_weights(args.source_lora)

        print("\n📥 加载源模型权重...")
        source_base_weights = loader.load_base_model_weights(source_model_path)

        print("\n📥 加载目标模型权重...")
        target_base_weights = loader.load_base_model_weights(args.target_model)

        print("\n🔄 执行LoRA-X Frobenius最优近似迁移...")
        with torch.no_grad():
            transferred_lora = lora_x.transfer_lora_weights(
                source_lora=source_lora_weights,
                target_base_weights=target_base_weights,
                source_base_weights=source_base_weights
            )

        if not transferred_lora:
            print("❌ 迁移失败：没有成功迁移任何层")
            return False

        os.makedirs(output_path, exist_ok=True)
        print(f"\n💾 保存迁移结果到: {output_path}")
        save_transferred_lora(transferred_lora, lora_config, output_path)

        print("\n🎉 LoRA迁移完成！")
        return True

    except Exception as e:
        logger.error(f"迁移过程出错: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
