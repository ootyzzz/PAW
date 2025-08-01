#!/usr/bin/env python3
"""
LoRA Linear Interpolation Baseline
用例: 从源模型的LoRA直接线性插值到目标模型
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

from interpo_core import InterpoCore
from model_utils import ModelWeightLoader, save_transferred_lora

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
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
    parser = argparse.ArgumentParser(description="LoRA Linear Interpolation Baseline")
    parser.add_argument("--source_lora", type=str, required=True,
                       help="源LoRA模型路径")
    parser.add_argument("--target_model", type=str, required=True,
                       help="目标基础模型路径")
    parser.add_argument("--output_base", type=str,
                       default="/root/autodl-tmp/interpolated",
                       help="输出基础路径")
    parser.add_argument("--alpha", type=float, default=1.0,
                       help="插值系数 α (0=仅源, 1=仅插值缩放)")
    args = parser.parse_args()

    timestamp = generate_timestamp()
    output_path = os.path.join(args.output_base, timestamp)
    source_model_path = infer_source_model_path(args.source_lora)

    print(f"\n🚀 LoRA线性插值迁移开始 🚀")
    print("="*80)
    print(f"📂 源LoRA路径: {args.source_lora}")
    print(f"📂 源模型路径: {source_model_path}")
    print(f"📂 目标模型路径: {args.target_model}")
    print(f"📂 输出路径: {output_path}")
    print(f"⚙️ 插值系数: {args.alpha}")
    print(f"🕐 时间戳: {timestamp}")
    print("="*80)

    try:
        if not os.path.exists(args.source_lora):
            raise FileNotFoundError(f"源LoRA路径不存在: {args.source_lora}")
        if not os.path.exists(source_model_path):
            raise FileNotFoundError(f"源模型路径不存在: {source_model_path}")
        if not os.path.exists(args.target_model):
            raise FileNotFoundError(f"目标模型路径不存在: {args.target_model}")

        interpo = InterpoCore(alpha=args.alpha)
        loader = ModelWeightLoader()

        print("\n📥 加载源LoRA权重...")
        source_lora_weights, lora_config = loader.load_lora_weights(args.source_lora)

        print("\n📥 加载目标模型权重...")
        target_base_weights = loader.load_base_model_weights(args.target_model)

        print("\n🔄 执行LoRA线性插值迁移...")
        with torch.no_grad():
            transferred_lora = interpo.transfer_lora_weights(
                source_lora=source_lora_weights,
                target_base_weights=target_base_weights
            )

        if not transferred_lora:
            print("❌ 迁移失败：没有成功迁移任何层")
            return False

        os.makedirs(output_path, exist_ok=True)
        print(f"\n💾 保存迁移结果到: {output_path}")
        save_transferred_lora(transferred_lora, lora_config, output_path)

        print("\n🎉 LoRA线性插值迁移完成！")
        return True

    except Exception as e:
        logger.error(f"迁移过程出错: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
