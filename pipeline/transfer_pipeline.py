#!/usr/bin/env python3
"""
transfer_pipeline.py
自动化LoRA训练和迁移管道

主要功能:
1. 训练 source model + LoRA
2. 迁移 source LoRA → target model
3. 评估目标基础模型
4. 评估迁移LoRA
5. 训练 target model + LoRA (可选)
6. 评估源基础模型
7. 生成详细的结果报告

使用方法:
python transfer_pipeline.py \
  --source_model Llama-3.2-3B-Instruct \
  --target_model Qwen_Qwen2.5-1.5B \
  --dataset arc-challenge

python transfer_pipeline.py \
  --source_model gemma-2-2b-it \
  --target_model Qwen_Qwen2.5-1.5B \
  --dataset arc-challenge

python pipeline/experiments/run_single_experiment.py --base_model /root/autodl-tmp/models/Qwen_Qwen2.5-1.5B --target_model /root/autodl-tmp/models/Llama-3.2-3B-Instruct --dataset arc-challenge --eval_only
    
快速测试:
python transfer_pipeline.py --quick_test
"""

import os
import sys
import argparse
import warnings

# 添加pipeline模块到路径
pipeline_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipeline')
if pipeline_root not in sys.path:
    sys.path.insert(0, pipeline_root)

# 修复MKL线程层冲突 - 必须在导入numpy/pandas之前设置
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'

# 屏蔽 Transformers 警告，但保留重要信息
warnings.filterwarnings("ignore", message=".*cache_implementation.*")
warnings.filterwarnings("ignore", message=".*generation flags are not valid.*")
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

from core.pipeline import TransferPipeline


def main():
    parser = argparse.ArgumentParser(
        description="LoRA训练和迁移自动化管道",
        epilog="""
使用示例:
  # 使用配置文件默认值
  python transfer_pipeline.py
  
  # 快速测试 (0.5B→1.5B, 20步训练, 5%评估)
  python transfer_pipeline.py --quick_test
  
  # 自定义模型
  python transfer_pipeline.py --source_model gemma-2-2b-it --target_model Qwen_Qwen2.5-1.5B --dataset arc-challenge
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source_model", type=str, default=None,
                       help="源模型路径或名称 (默认使用配置文件)")
    parser.add_argument("--target_model", type=str, default=None,
                       help="目标模型路径或名称 (默认使用配置文件)")
    parser.add_argument("--dataset", type=str, default=None,
                       help="数据集名称 (默认使用配置文件)")
    parser.add_argument("--config", type=str, 
                       default=os.path.join(os.path.dirname(__file__), "config", "pipeline_config.yaml"),
                       help="配置文件路径")
    parser.add_argument("--eval_only", action="store_true",
                       help="仅运行评估，跳过训练和迁移")
    parser.add_argument("--quick_test", action="store_true",
                       help="快速测试模式：自动使用1.5B→0.5B配置")
    
    args = parser.parse_args()
    
    # Quick test mode: use preset configuration
    if args.quick_test:
        print("Quick Test Mode: 0.5B → 1.5B")
        
        # Create pipeline instance (using quick test configuration)
        pipeline = TransferPipeline(quick_test=True)
        
        # Use recommended configuration
        if not args.source_model:
            args.source_model = pipeline.config.get('recommended_models.source')
        if not args.target_model:
            args.target_model = pipeline.config.get('recommended_models.target')
        if not args.dataset:
            args.dataset = pipeline.config.get('recommended_models.dataset')
            
        print(f"Source Model: {args.source_model}")
        print(f"Target Model: {args.target_model}")
        print(f"Dataset: {args.dataset}")
        print(f"Training Steps: 20, Evaluation Ratio: 5%")
        print("")
    else:
        # Create pipeline instance (using normal configuration)
        pipeline = TransferPipeline(args.config)
        
        # Use default values from config if not provided
        if not args.source_model:
            args.source_model = pipeline.config.get('default_experiment.source_model')
        if not args.target_model:
            args.target_model = pipeline.config.get('default_experiment.target_model')
        if not args.dataset:
            args.dataset = pipeline.config.get('default_experiment.dataset')
        if not args.eval_only:
            args.eval_only = pipeline.config.get('default_experiment.eval_only', False)
        
        # Validate that we have all required parameters
        if not all([args.source_model, args.target_model, args.dataset]):
            print("ERROR: Missing required parameters. Check configuration file or provide:")
            print("  --source_model [model_name]")
            print("  --target_model [model_name]") 
            print("  --dataset [dataset_name]")
            print("NOTE: Or use --quick_test for automatic configuration")
            return False
        
        print(f"Using Configuration: {args.config}")
        print(f"Source Model: {args.source_model}")
        print(f"Target Model: {args.target_model}")
        print(f"Dataset: {args.dataset}")
        print(f"Eval Only: {args.eval_only}")
        print("")
    
    # 处理模型路径
    if not args.source_model.startswith('/'):
        args.source_model = pipeline.config.get_model_path(args.source_model)
    if not args.target_model.startswith('/'):
        args.target_model = pipeline.config.get_model_path(args.target_model)
    
    # 验证模型存在
    if not os.path.exists(args.source_model):
        print(f"❌ 源模型不存在: {args.source_model}")
        return False
    if not os.path.exists(args.target_model):
        print(f"❌ 目标模型不存在: {args.target_model}")
        return False
    
    # 验证数据集
    supported_datasets = pipeline.config.get('training.datasets', [])
    if args.dataset not in supported_datasets:
        print(f"❌ 不支持的数据集: {args.dataset}")
        print(f"✅ 支持的数据集: {', '.join(supported_datasets)}")
        return False
    
    # 运行管道
    success = pipeline.run_pipeline(
        args.source_model, 
        args.target_model, 
        args.dataset,
        eval_only=args.eval_only
    )
    
    if success:
        print(f"\n🎉 管道执行成功!")
        return True
    else:
        print(f"\n❌ 管道执行失败!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
