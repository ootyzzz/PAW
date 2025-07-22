#!/usr/bin/env python3
"""
scripts/prepare_data.py
数据准备脚本

使用方法：
python scripts/prepare_data.py --source data_to_lora/cs/arc-challenge/arc-challenge_train_formatted.jsonl --output_dir Lora_Gen/data
python scripts/prepare_data.py --checkpoint_dir runs/arc-challenge_lora_20250721_005053/checkpoints --analyze
"""
import os
import sys
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any
import torch

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from Lora_Gen.core.data_module import create_prompt_splits
from Lora_Gen.core.generator import LoRATokenizer


def analyze_checkpoints(checkpoint_dir: str):
    """分析checkpoint文件"""
    checkpoint_dir = Path(checkpoint_dir)
    
    print(f"📊 分析Checkpoint目录: {checkpoint_dir}")
    print("=" * 60)
    
    # 查找checkpoint文件
    checkpoint_files = []
    for ext in ['*.ckpt', '*.pt', '*.pth']:
        checkpoint_files.extend(list(checkpoint_dir.glob(ext)))
    
    if not checkpoint_files:
        print("❌ 未找到checkpoint文件")
        return
    
    print(f"📁 发现 {len(checkpoint_files)} 个checkpoint文件:")
    
    # 分析每个checkpoint
    total_params = 0
    lora_params = 0
    file_sizes = []
    
    for i, ckpt_file in enumerate(checkpoint_files[:10]):  # 只分析前10个
        try:
            # 检查文件大小
            file_size = ckpt_file.stat().st_size / (1024**3)  # GB
            file_sizes.append(file_size)
            
            # 加载checkpoint - 处理PyTorch 2.6
            try:
                checkpoint = torch.load(ckpt_file, map_location='cpu', weights_only=False)
            except Exception:
                checkpoint = torch.load(ckpt_file, map_location='cpu')
            
            # 统计参数
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    params = checkpoint['state_dict']
                else:
                    params = checkpoint
                
                file_total = 0
                file_lora = 0
                
                for name, param in params.items():
                    param_count = param.numel()
                    file_total += param_count
                    
                    if 'lora' in name.lower():
                        file_lora += param_count
                
                total_params += file_total
                lora_params += file_lora
                
                print(f"  {i+1:2d}. {ckpt_file.name}")
                print(f"      大小: {file_size:.2f} GB")
                print(f"      总参数: {file_total:,}")
                print(f"      LoRA参数: {file_lora:,}")
                
        except Exception as e:
            print(f"  {i+1:2d}. {ckpt_file.name} - ❌ 加载失败: {e}")
    
    print(f"\\n📈 统计摘要:")
    print(f"  - 文件数量: {len(checkpoint_files)}")
    print(f"  - 平均文件大小: {sum(file_sizes)/len(file_sizes):.2f} GB")
    print(f"  - 平均总参数: {total_params//len(file_sizes):,}")
    print(f"  - 平均LoRA参数: {lora_params//len(file_sizes):,}")


def test_tokenization(checkpoint_file: str, max_tokens: int = 512, token_dim: int = 384):
    """测试checkpoint的tokenization"""
    print(f"🧪 测试Checkpoint Tokenization")
    print(f"文件: {checkpoint_file}")
    print(f"配置: max_tokens={max_tokens}, token_dim={token_dim}")
    print("=" * 50)
    
    try:
        tokenizer = LoRATokenizer(max_tokens=max_tokens, token_dim=token_dim)
        
        # Tokenize
        tokens = tokenizer.tokenize_checkpoint(checkpoint_file)
        
        print(f"✅ Tokenization成功!")
        print(f"  - Token形状: {tokens.shape}")
        print(f"  - Token统计:")
        print(f"    Mean: {tokens.mean():.6f}")
        print(f"    Std: {tokens.std():.6f}")
        print(f"    Min: {tokens.min():.6f}")
        print(f"    Max: {tokens.max():.6f}")
        
        # 测试detokenization
        param_vec = tokenizer.detokenize(tokens)
        print(f"  - Detokenization形状: {param_vec.shape}")
        
    except Exception as e:
        print(f"❌ Tokenization失败: {e}")
        import traceback
        traceback.print_exc()


def prepare_training_data(
    source_file: str,
    output_dir: str,
    samples_per_prompt: int = 4,
    test_ratio: float = 0.1
):
    """准备训练数据"""
    print(f"📊 准备训练数据")
    print(f"源文件: {source_file}")
    print(f"输出目录: {output_dir}")
    print(f"每个prompt样本数: {samples_per_prompt}")
    print(f"测试比例: {test_ratio}")
    print("=" * 50)
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 输出文件路径
    train_file = output_dir / "train_prompts.jsonl"
    val_file = output_dir / "val_prompts.jsonl"
    
    try:
        # 创建数据分割
        create_prompt_splits(
            input_file=source_file,
            train_output=str(train_file),
            val_output=str(val_file),
            samples_per_prompt=samples_per_prompt,
            test_ratio=test_ratio
        )
        
        # 验证生成的文件
        train_count = 0
        val_count = 0
        
        with open(train_file, 'r', encoding='utf-8') as f:
            train_count = sum(1 for line in f if line.strip())
        
        with open(val_file, 'r', encoding='utf-8') as f:
            val_count = sum(1 for line in f if line.strip())
        
        print(f"\\n✅ 数据准备完成!")
        print(f"  - 训练样本: {train_count}")
        print(f"  - 验证样本: {val_count}")
        print(f"  - 训练文件: {train_file}")
        print(f"  - 验证文件: {val_file}")
        
    except Exception as e:
        print(f"❌ 数据准备失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="数据准备工具")
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 准备训练数据
    prep_parser = subparsers.add_parser('prepare', help='准备训练数据')
    prep_parser.add_argument('--source', type=str, required=True,
                            help='源数据文件路径')
    prep_parser.add_argument('--output_dir', type=str, default='Lora_Gen/data',
                            help='输出目录')
    prep_parser.add_argument('--samples_per_prompt', type=int, default=4,
                            help='每个prompt的样本数')
    prep_parser.add_argument('--test_ratio', type=float, default=0.1,
                            help='测试集比例')
    
    # 分析checkpoint
    analyze_parser = subparsers.add_parser('analyze', help='分析checkpoint文件')
    analyze_parser.add_argument('--checkpoint_dir', type=str, required=True,
                               help='Checkpoint目录路径')
    
    # 测试tokenization
    test_parser = subparsers.add_parser('test', help='测试tokenization')
    test_parser.add_argument('--checkpoint_file', type=str, required=True,
                            help='单个checkpoint文件路径')
    test_parser.add_argument('--max_tokens', type=int, default=512,
                            help='最大token数')
    test_parser.add_argument('--token_dim', type=int, default=384,
                            help='Token维度')
    
    # 兼容旧的命令行参数格式
    parser.add_argument('--source', type=str, help='源数据文件路径')
    parser.add_argument('--output_dir', type=str, default='Lora_Gen/data',
                       help='输出目录')
    parser.add_argument('--checkpoint_dir', type=str, help='Checkpoint目录路径')
    parser.add_argument('--analyze', action='store_true', help='分析checkpoint文件')
    
    args = parser.parse_args()
    
    print("🔧 LoRA Generator 数据准备工具")
    print("=" * 50)
    
    try:
        if args.command == 'prepare':
            prepare_training_data(
                source_file=args.source,
                output_dir=args.output_dir,
                samples_per_prompt=args.samples_per_prompt,
                test_ratio=args.test_ratio
            )
        elif args.command == 'analyze':
            analyze_checkpoints(args.checkpoint_dir)
        elif args.command == 'test':
            test_tokenization(
                checkpoint_file=args.checkpoint_file,
                max_tokens=args.max_tokens,
                token_dim=args.token_dim
            )
        else:
            # 兼容旧格式
            if args.analyze and args.checkpoint_dir:
                analyze_checkpoints(args.checkpoint_dir)
            elif args.source:
                prepare_training_data(args.source, args.output_dir)
            else:
                parser.print_help()
        
        return True
        
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
