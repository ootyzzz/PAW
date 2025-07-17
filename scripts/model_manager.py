#!/usr/bin/env python3
"""
model_manager.py 已废弃，仅保留底层工具支持
所有模型管理操作请通过 baidu_gpu_lora_training.ipynb notebook 完成
如需底层命令行调用，可在 notebook 中使用 !python scripts/model_manager.py 或 subprocess
"""

import os
import argparse
from pathlib import Path

def download_model(model_name, cache_dir="./models"):
    """下载模型到本地"""
    print(f"📥 开始下载模型: {model_name}")
    print(f"📁 保存路径: {cache_dir}")
    
    # 创建目录
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print("📦 下载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        print("🤖 下载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            torch_dtype="auto"
        )
        
        print("✅ 模型下载完成!")
        
        # 显示模型信息
        print(f"📊 模型信息:")
        print(f"  - 名称: {model_name}")
        print(f"  - 参数量: {model.num_parameters():,}")
        print(f"  - 缓存路径: {cache_dir}")
        
        return True
        
    except ImportError:
        print("❌ 请先安装transformers: pip install transformers")
        return False
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def list_models(cache_dir="./models"):
    """列出已下载的模型"""
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        print("📁 模型缓存目录不存在")
        return
    
    print(f"📚 已下载的模型 ({cache_dir}):")
    
    # 查找模型目录
    model_dirs = []
    for item in cache_path.iterdir():
        if item.is_dir():
            # 检查是否包含模型文件
            if any(item.glob("*.bin")) or any(item.glob("*.safetensors")):
                model_dirs.append(item)
    
    if not model_dirs:
        print("  暂无已下载的模型")
        return
    
    for i, model_dir in enumerate(model_dirs, 1):
        print(f"  {i}. {model_dir.name}")
        
        # 显示模型文件大小
        total_size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
        size_mb = total_size / (1024 * 1024)
        print(f"     大小: {size_mb:.1f} MB")

def main():
    parser = argparse.ArgumentParser(description="模型管理工具")
    parser.add_argument("--action", choices=["download", "list"], default="list",
                       help="操作类型")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B",
                       help="模型名称")
    parser.add_argument("--cache_dir", type=str, default="./models",
                       help="模型缓存目录")
    
    args = parser.parse_args()
    
    print("🏗️  Foundation Model Manager")
    print("=" * 50)
    
    if args.action == "download":
        download_model(args.model, args.cache_dir)
    elif args.action == "list":
        list_models(args.cache_dir)

if __name__ == "__main__":
    main()
