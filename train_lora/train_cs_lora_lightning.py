#!/usr/bin/env python3
"""
train_cs_lora_lightning.py
PyTorch Lightning + SwanLab 版本的 LoRA 训练脚本
现代化训练框架，支持实时监控和实验管理
======================================================================

# 数据集
python train_cs_lora_lightning.py --dataset arc-challenge arc-easy boolq hellaswag openbookqa piqa winogrande

# 自定义参数
python train_cs_lora_lightning.py --dataset arc-challenge --bs 16
python train_cs_lora_lightning.py --dataset arc-challenge --dry_run

# 批量执行 (PowerShell)
foreach ($dataset in @("arc-challenge", "arc-easy", "boolq", "hellaswag", "openbookqa", "piqa", "winogrande")) {
    Write-Host "🚀 开始训练 $dataset..."
    python train_cs_lora_lightning.py --dataset $dataset
    Write-Host "✅ $dataset 训练完成"
}

======================================================================

主要特性:
✅ PyTorch Lightning: 现代化训练框架，自动处理设备、分布式等
✅ SwanLab 集成: 实时监控训练过程，实验管理和对比  
✅ 主流保存方式: 遵循 HuggingFace/Lightning
✅ 自动混合精度: 提升训练效率
✅ 模块化架构: 代码按功能拆分，易于维护和扩展

目录结构 (Lightning + SwanLab 风格，按模型分组):
./runs/                          # 主实验目录
├── {model_name}/                # 按模型名称分组 (如 Qwen2.5-1.5B, Qwen2.5-0.5B)
│   ├── {experiment_name}/       # 单个实验 (如 arc-challenge_lora_20250723_143022)
│   │   ├── checkpoints/         # Lightning checkpoints (.ckpt) 
│   │   ├── tensorboard_logs/    # TensorBoard 日志
│   │   ├── swanlab_logs/       # SwanLab 日志
│   │   ├── final_model/        # 最终 HuggingFace 格式模型
│   │   └── config.yaml         # 实验配置
│   └── {another_experiment}/    # 同一模型的其他实验
└── {another_model}/             # 其他模型的实验
    └── {experiment_name}/
        └── ...
"""

import os
import sys
import yaml
import argparse
import warnings
from datetime import datetime
from pathlib import Path

# 屏蔽 Transformers 和 Lightning 警告
warnings.filterwarnings("ignore", message=".*cache_implementation.*")
warnings.filterwarnings("ignore", message=".*generation flags are not valid.*")
warnings.filterwarnings("ignore", message=".*sync_dist.*")
warnings.filterwarnings("ignore", message=".*recommended.*")
warnings.filterwarnings("ignore", message=".*Progress bar.*")
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['PYTORCH_LIGHTNING_VERBOSITY'] = 'ERROR'  # 只显示错误信息
os.environ['TQDM_DISABLE'] = '1'  # 禁用tqdm进度条

# 强制使用单GPU，避免多进程启动（4卡A800环境）
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 只使用第一张GPU
os.environ['WORLD_SIZE'] = '1'  # 强制单节点
os.environ['LOCAL_RANK'] = '0'
os.environ['RANK'] = '0'

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, '..'))  # 添加上级目录

# 导入核心模块
from core import (
    setup_environment,
    create_lightning_config,
    run_lightning_training
)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Lightning + SwanLab LoRA训练脚本")
    parser.add_argument("--dataset", type=str, required=True,
                       help="要训练的数据集名称 (arc-challenge, arc-easy, boolq, hellaswag, openbookqa, piqa, winogrande)")
    parser.add_argument("--config", type=str, default="./config/lightning_config.yaml",
                       help="Lightning配置文件路径")
    parser.add_argument("--dry_run", action="store_true",
                       help="干运行模式: 验证配置和数据文件，创建实验目录，但不实际训练模型")
    parser.add_argument("--bs", type=int, default=None,
                       help="批处理大小 (默认自动选择)")
    parser.add_argument("--max_steps", type=int, default=125,
                       help="训练总步数 (默认125)")
    parser.add_argument("--save_steps", type=int, default=1,
                       help="保存最后多少步的检查点 (默认1)")
    parser.add_argument("--lr", type=float, default=1e-5,
                       help="学习率")
    parser.add_argument("--lr2", type=float, default=None,
                       help="第二阶段学习率 (默认和learning_rate相同")
    parser.add_argument("--base_model", type=str, default="../autodl-tmp/models/gemma-2-2b-it",
                       help="基础模型路径或huggingface模型名 (默认=../autodl-tmp/models/gemma-2-2b-it)")
    # 为了兼容性，保留但忽略的参数
    parser.add_argument("--track_batches", action="store_true",
                       help="(兼容参数，当前版本忽略)")

    args = parser.parse_args()
    if args.lr2 is None:
        args.lr2 = args.lr
    
    # 验证数据集名称
    valid_datasets = ['arc-challenge', 'arc-easy', 'boolq', 'hellaswag', 'openbookqa', 'piqa', 'winogrande']
    if args.dataset not in valid_datasets:
        print(f"❌ 无效的数据集名称: {args.dataset}")
        print(f"✅ 可用数据集: {', '.join(valid_datasets)}")
        return False
    
    # 兼容性提示
    if args.track_batches:
        print("💡 注意: --track_batches 功能将在下个版本中实现，当前版本忽略此参数")
    
    print("🚀 Lightning + SwanLab LoRA训练脚本 (模块化版本)")
    print("=" * 70)
    print(f"目标数据集: {args.dataset}")
    print(f"配置文件: {args.config}")
    print(f"Batch大小: {args.bs if args.bs else '自动选择'}")
    print(f"训练步数: {args.max_steps}")
    print(f"保存步数: 保存最后{args.save_steps}个检查点")
    print(f"学习率: {args.lr} -> {args.lr2 or args.lr/10}")
    print(f"基础模型: {args.base_model}")
    print(f"运行模式: {'🏃 Dry Run (验证配置和数据，不训练)' if args.dry_run else '🚀 完整训练'}")
    print(f"框架: PyTorch Lightning + SwanLab")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    try:
        # 设置环境
        setup_environment()
        
        # 加载基础配置
        config_path = os.path.join(os.path.dirname(__file__), args.config)
        with open(config_path, 'r', encoding='utf-8') as f:
            base_config = yaml.safe_load(f)
        
        # 创建Lightning配置
        config = create_lightning_config(
            dataset_name=args.dataset, 
            base_config=base_config, 
            base_model_path=args.base_model,  # 传递base_model路径
            batch_size=args.bs, 
            max_steps=args.max_steps, 
            save_steps=args.save_steps, 
            learning_rate=args.lr, 
            learning_rate_stage2=args.lr2
        )
        
        # 注入 base model 路径
        if 'model' not in config:
            config['model'] = {}
        config['model']['path'] = args.base_model
        config['model']['name'] = args.base_model

        # 执行训练
        results = run_lightning_training(
            dataset_name=args.dataset,
            config=config,
            dry_run=args.dry_run
        )
        
        print(f"\n🎉 实验完成!")
        print(f"📊 结果: {results.get('status', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
