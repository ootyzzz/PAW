#!/usr/bin/env python3
"""
数据集完整处理流程
1. 混合并标准化7个commonsense数据集
2. 最终清理和格式化
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command, description):
    """运行命令并显示进度"""
    print(f"\n🔄 {description}")
    print(f"执行命令: {command}")
    print("-" * 60)
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=False, text=True)
        print(f"✅ {description} 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="数据集完整处理流程")
    parser.add_argument("--datasets_dir", type=str, default="raw_datasets",
                       help="数据集根目录")
    parser.add_argument("--output_dir", type=str, default="raw_datasets/commonsense",
                       help="输出目录")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="每个数据集最大样本数")
    parser.add_argument("--skip_mix", action="store_true",
                       help="跳过数据集混合步骤")
    parser.add_argument("--skip_clean", action="store_true",
                       help="跳过数据清理步骤")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 数据集完整处理流程")
    print("=" * 60)
    
    # 确保在正确的工作目录
    script_dir = Path(__file__).parent.parent.parent
    os.chdir(script_dir)
    print(f"工作目录: {os.getcwd()}")
    
    success = True
    
    # 步骤1: 混合数据集
    if not args.skip_mix:
        mix_cmd = f"python utils/dataset/cs_mixer.py"
        mix_cmd += f" --datasets_dir {args.datasets_dir}"
        mix_cmd += f" --output_dir {args.output_dir}"
        mix_cmd += f" --seed {args.seed}"
        if args.max_samples:
            mix_cmd += f" --max_samples {args.max_samples}"
        
        success = run_command(mix_cmd, "步骤1: 混合并标准化数据集")
        
        if not success:
            print("\n❌ 数据集混合失败，停止处理")
            return
    else:
        print("\n⏭️  跳过数据集混合步骤")
    
    # 步骤2: 最终数据清理
    if not args.skip_clean:
        clean_cmd = "python utils/dataset/cs_formatter.py"
        success = run_command(clean_cmd, "步骤2: 最终数据清理和格式化")
        
        if not success:
            print("\n❌ 数据清理失败")
            return
    else:
        print("\n⏭️  跳过数据清理步骤")
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 数据集处理流程完成！")
        print("\n📁 输出文件:")
        print(f"   - {args.output_dir}/cs_mixed.jsonl (合并后数据)")
        print(f"   - {args.output_dir}/cs_mixed_formatted.jsonl (格式化后数据)")
    else:
        print("❌ 处理流程未完全成功")
    print("=" * 60)

if __name__ == "__main__":
    main()
