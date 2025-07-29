#!/usr/bin/env python3
"""
Lightning风格的快速评估脚本 - 基于PyTorch Lightning优化的评估方法
支持同时评估多个模型，包括基础模型和LoRA模型
使用Lightning的数据加载和并行处理机制，显著提高评估速度

使用示例:
python eval/lightning_eval.py --models_list \\
    /root/autodl-tmp/models/Qwen_Qwen2.5-1.5B \\
    /root/autodl-tmp/models/gemma-2-2b-it \\
    /root/PAW/runs/Qwen_Qwen2.5-1.5B/arc-challenge_lora_20250723_191421/final_model \\
    --dataset arc-challenge
"""

import sys
import argparse
import os
from datetime import datetime

# 导入核心模块
from core.batch_eval import evaluate_models


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Lightning风格的快速模型评估工具")
    parser.add_argument("--lora", type=str, nargs="*", default=None,
                       help="要评估的LoRA模型路径列表 (可选，如果不提供则评测base_model)")
    parser.add_argument("--dataset", type=str, default="arc-challenge",
                       help="数据集名称 (默认: arc-challenge)")
    parser.add_argument("--output_dir", type=str, default="eval/results",
                       help="评估结果输出目录 (默认: eval/results)")
    parser.add_argument("--base_model", type=str, default=None,
                       help="基础模型路径 (当不提供lora时必需，或用于加载LoRA模型)")
    parser.add_argument("--sample_ratio", type=float, default=1.0,
                       help="数据采样比例，加速评估 (默认: 1.0 = 100%%)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="批处理大小 (默认: 8)")
    
    args = parser.parse_args()
    
    # 验证参数逻辑
    if not args.lora and not args.base_model:
        parser.error("必须提供 --lora 或 --base_model 参数之一")
    
    print("🔬 Lightning模型评估工具")
    print("=" * 50)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 确定要评估的模型列表
    if args.lora:
        # 如果提供了lora参数，过滤掉无效的参数（如单独的反斜杠）
        filtered_models = []
        for model_path in args.lora:
            # 过滤掉空字符串、单独的反斜杠等无效参数
            if model_path and model_path.strip() and model_path.strip() not in ['\\', '/', '']:
                filtered_models.append(model_path.strip())
    else:
        # 如果没有提供lora参数，使用base_model
        filtered_models = [args.base_model]
        print(f"📝 评估模式: 直接评测基础模型")
    
    # 验证模型路径
    valid_models = []
    for model_path in filtered_models:
        if os.path.exists(model_path):
            valid_models.append(model_path)
            print(f"✅ 找到模型: {model_path}")
        else:
            print(f"⚠️ 警告: 模型路径不存在: {model_path}")
            print(f"将尝试作为HuggingFace模型名称加载")
            valid_models.append(model_path)  # 仍然添加，让下游处理
    
    if len(valid_models) == 0:
        print("❌ 错误: 没有有效的模型路径")
        return False
    
    # 检测LoRA模型和基础模型
    lora_models = []
    for model_path in valid_models:
        if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "adapter_config.json")):
            lora_models.append(model_path)
    
    if lora_models and not args.base_model:
        print(f"ℹ️ 检测到{len(lora_models)}个LoRA模型:")
        for lora in lora_models:
            print(f"  - {lora}")
        print(f"💡 如果加载失败，请使用 --base_model 参数指定基础模型")
    
    try:
        # 运行评估
        results = evaluate_models(
            models_list=valid_models,
            dataset_name=args.dataset,
            output_dir=args.output_dir,
            base_model_path=args.base_model,
            sample_ratio=args.sample_ratio,
            batch_size=args.batch_size
        )
        
        print("✅ 评估完成")
        return True
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
