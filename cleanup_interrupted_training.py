#!/usr/bin/env python3
"""
清理被打断训练的痕迹
每次手动打断训练后运行此脚本
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
LOGS_DIR = PROJECT_ROOT / "logs"

def find_interrupted_experiments():
    """找到被打断的实验"""
    interrupted = []
    
    if not EXPERIMENTS_DIR.exists():
        return interrupted
    
    for exp_dir in EXPERIMENTS_DIR.iterdir():
        if exp_dir.is_dir() and exp_dir.name.startswith('commonsense_lora_'):
            metadata_file = exp_dir / 'metadata.json'
            
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # 检查是否为被打断的实验
                    status = metadata.get('status', '')
                    if status in ['running', 'created', 'failed']:
                        interrupted.append({
                            'dir': exp_dir,
                            'name': exp_dir.name,
                            'status': status,
                            'created': metadata.get('created_at', 'unknown')
                        })
                        
                except Exception as e:
                    print(f"无法读取 {metadata_file}: {e}")
                    # 如果metadata损坏，也认为是被打断的
                    interrupted.append({
                        'dir': exp_dir,
                        'name': exp_dir.name,
                        'status': 'corrupted',
                        'created': 'unknown'
                    })
    
    return interrupted

def cleanup_interrupted_training(dry_run=True):
    """清理被打断的训练"""
    print("🧹 清理被打断的训练...")
    
    # 1. 清理不完整的实验
    interrupted = find_interrupted_experiments()
    total_size = 0
    
    if interrupted:
        print(f"\n找到 {len(interrupted)} 个被打断的实验:")
        
        for exp in interrupted:
            exp_dir = exp['dir']
            dir_size = sum(f.stat().st_size for f in exp_dir.rglob('*') if f.is_file())
            size_mb = dir_size / (1024 * 1024)
            total_size += size_mb
            
            print(f"  - {exp['name']} (状态: {exp['status']}, 大小: {size_mb:.1f}MB)")
            
            if not dry_run:
                try:
                    shutil.rmtree(exp_dir)
                    print(f"    ✅ 已删除")
                except Exception as e:
                    print(f"    ❌ 删除失败: {e}")
            else:
                print(f"    [DRY RUN] 将被删除")
    
    # 2. 清理旧的训练日志 (保留最新的5个)
    if LOGS_DIR.exists():
        log_files = list(LOGS_DIR.glob('training_*.log'))
        log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        if len(log_files) > 5:
            old_logs = log_files[5:]  # 保留最新5个
            print(f"\n清理旧日志文件 ({len(old_logs)} 个):")
            
            for log_file in old_logs:
                size_mb = log_file.stat().st_size / (1024 * 1024)
                print(f"  - {log_file.name} ({size_mb:.1f}MB)")
                
                if not dry_run:
                    try:
                        log_file.unlink()
                        print(f"    ✅ 已删除")
                    except Exception as e:
                        print(f"    ❌ 删除失败: {e}")
                else:
                    print(f"    [DRY RUN] 将被删除")
    
    # 3. 清理缓存文件
    cache_dirs = list(PROJECT_ROOT.rglob('__pycache__'))
    if cache_dirs:
        print(f"\n清理Python缓存 ({len(cache_dirs)} 个目录):")
        for cache_dir in cache_dirs:
            print(f"  - {cache_dir.relative_to(PROJECT_ROOT)}")
            if not dry_run:
                try:
                    shutil.rmtree(cache_dir)
                    print(f"    ✅ 已删除")
                except Exception as e:
                    print(f"    ❌ 删除失败: {e}")
            else:
                print(f"    [DRY RUN] 将被删除")
    
    print(f"\n总计节省空间: {total_size:.1f}MB")
    
    if dry_run:
        print(f"\n⚠️ 这是预演模式，没有实际删除文件")
        print(f"运行 python cleanup_interrupted_training.py --execute 来实际执行清理")
    else:
        print(f"\n✅ 清理完成!")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="清理被打断的训练痕迹")
    parser.add_argument("--execute", action="store_true", help="实际执行清理（默认为预演模式）")
    parser.add_argument("--list-only", action="store_true", help="仅列出需要清理的文件")
    
    args = parser.parse_args()
    
    print("🧹 清理被打断训练工具")
    print("=" * 50)
    
    if args.list_only:
        interrupted = find_interrupted_experiments()
        if interrupted:
            print("被打断的实验:")
            for exp in interrupted:
                print(f"  - {exp['name']} ({exp['status']})")
        else:
            print("没有找到被打断的实验")
        return
    
    dry_run = not args.execute
    cleanup_interrupted_training(dry_run)

if __name__ == "__main__":
    main()
