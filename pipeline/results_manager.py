#!/usr/bin/env python3
"""
results_manager.py
实验结果管理工具

功能:
- 查看实验历史
- 搜索特定实验
- 生成实验报告
- 清理过期实验
"""

import os
import sys
import yaml
import pandas as pd
import argparse
from datetime import datetime, timedelta
from pathlib import Path

class ResultsManager:
    def __init__(self, config_path="config/pipeline_config.yaml"):
        """初始化结果管理器"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.results_dir = self.config['paths']['results_dir']
        self.csv_path = os.path.join(self.results_dir, self.config['results']['csv_file'])
    
    def load_results(self):
        """加载实验结果"""
        if not os.path.exists(self.csv_path):
            print("📊 还没有任何实验结果")
            return pd.DataFrame()
        
        try:
            return pd.read_csv(self.csv_path)
        except Exception as e:
            print(f"❌ 加载结果失败: {e}")
            return pd.DataFrame()
    
    def list_experiments(self, limit=10):
        """列出最近的实验"""
        df = self.load_results()
        if df.empty:
            return
        
        print(f"📊 最近 {min(limit, len(df))} 个实验:")
        print("=" * 80)
        
        # 按时间排序
        df_sorted = df.sort_values('timestamp', ascending=False).head(limit)
        
        for _, row in df_sorted.iterrows():
            print(f"🆔 {row['experiment_id']}")
            print(f"   📅 时间: {row['timestamp']}")
            print(f"   🎯 {self._get_model_name(row['source_model'])} → {self._get_model_name(row['target_model'])}")
            print(f"   📚 数据集: {row['dataset']}")
            
            # 显示关键结果
            if pd.notna(row['transferred_acc']) and pd.notna(row['target_acc']):
                improvement = (row['transferred_acc'] - row['target_acc']) * 100
                print(f"   📈 迁移效果: {row['transferred_acc']:.4f} (+{improvement:.2f}%)")
            
            print()
    
    def search_experiments(self, **filters):
        """搜索实验"""
        df = self.load_results()
        if df.empty:
            return
        
        print(f"🔍 搜索条件: {filters}")
        
        # 应用过滤器
        filtered_df = df.copy()
        for key, value in filters.items():
            if key in df.columns and value:
                if key in ['source_model', 'target_model']:
                    # 模型名支持部分匹配
                    filtered_df = filtered_df[filtered_df[key].str.contains(value, case=False, na=False)]
                else:
                    filtered_df = filtered_df[filtered_df[key] == value]
        
        if filtered_df.empty:
            print("🚫 未找到匹配的实验")
            return
        
        print(f"✅ 找到 {len(filtered_df)} 个匹配实验:")
        print("=" * 60)
        
        for _, row in filtered_df.iterrows():
            print(f"🆔 {row['experiment_id']}")
            print(f"   📅 {row['timestamp']}")
            print(f"   🎯 {self._get_model_name(row['source_model'])} → {self._get_model_name(row['target_model'])}")
            print(f"   📚 {row['dataset']}")
            
            if pd.notna(row['source_acc']):
                print(f"   📊 源模型: {row['source_acc']:.4f}")
            if pd.notna(row['target_acc']):
                print(f"   📊 目标模型: {row['target_acc']:.4f}")
            if pd.notna(row['transferred_acc']):
                improvement = (row['transferred_acc'] - row['target_acc']) * 100 if pd.notna(row['target_acc']) else 0
                print(f"   📊 迁移结果: {row['transferred_acc']:.4f} (+{improvement:.2f}%)")
            print()
    
    def generate_report(self, dataset=None):
        """生成实验报告"""
        df = self.load_results()
        if df.empty:
            return
        
        if dataset:
            df = df[df['dataset'] == dataset]
            if df.empty:
                print(f"🚫 数据集 {dataset} 没有实验记录")
                return
            print(f"📊 数据集 {dataset} 实验报告")
        else:
            print("📊 全部实验报告")
        
        print("=" * 60)
        
        # 总体统计
        print(f"📈 总实验数: {len(df)}")
        print(f"📅 时间范围: {df['timestamp'].min()} - {df['timestamp'].max()}")
        print(f"🎯 涉及模型: {', '.join(df['source_model'].apply(self._get_model_name).unique())}")
        print(f"📚 涉及数据集: {', '.join(df['dataset'].unique())}")
        
        # 性能统计
        if not df['transferred_acc'].isna().all() and not df['target_acc'].isna().all():
            improvements = (df['transferred_acc'] - df['target_acc']) * 100
            improvements = improvements.dropna()
            
            if not improvements.empty:
                print(f"\n📈 迁移效果统计:")
                print(f"   平均提升: {improvements.mean():.2f}%")
                print(f"   最大提升: {improvements.max():.2f}%")
                print(f"   最小提升: {improvements.min():.2f}%")
                print(f"   成功率: {(improvements > 0).sum()}/{len(improvements)} ({(improvements > 0).mean()*100:.1f}%)")
        
        # 按数据集分组统计
        if not dataset:
            print(f"\n📚 各数据集统计:")
            for ds in df['dataset'].unique():
                ds_df = df[df['dataset'] == ds]
                print(f"   {ds}: {len(ds_df)} 个实验")
        
        print()
    
    def cleanup_old_experiments(self, days=30, dry_run=True):
        """清理旧实验文件"""
        df = self.load_results()
        if df.empty:
            return
        
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff_date.strftime("%Y%m%d")
        
        old_experiments = df[df['timestamp'] < cutoff_str]
        
        if old_experiments.empty:
            print(f"✅ 没有超过 {days} 天的实验")
            return
        
        print(f"🗑️ 发现 {len(old_experiments)} 个超过 {days} 天的实验:")
        
        for _, row in old_experiments.iterrows():
            print(f"   📅 {row['timestamp']} - {row['experiment_id']}")
            
            # 检查相关文件
            paths_to_check = []
            if pd.notna(row['source_lora_path']):
                paths_to_check.append(row['source_lora_path'])
            if pd.notna(row['target_lora_path']):
                paths_to_check.append(row['target_lora_path'])
            if pd.notna(row['transferred_lora_path']):
                paths_to_check.append(row['transferred_lora_path'])
            
            for path in paths_to_check:
                if os.path.exists(path):
                    print(f"     📁 {path}")
                    if not dry_run:
                        try:
                            import shutil
                            shutil.rmtree(path)
                            print(f"     ✅ 已删除")
                        except Exception as e:
                            print(f"     ❌ 删除失败: {e}")
        
        if dry_run:
            print(f"\n🔍 这是预演模式，实际文件未删除")
            print(f"💡 使用 --no-dry-run 执行实际删除")
        else:
            print(f"\n✅ 清理完成")
    
    def _get_model_name(self, model_path):
        """获取模型简称"""
        return os.path.basename(model_path.strip())

def main():
    parser = argparse.ArgumentParser(description="实验结果管理工具")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 列出实验
    list_parser = subparsers.add_parser('list', help='列出最近的实验')
    list_parser.add_argument('--limit', type=int, default=10, help='显示数量限制')
    
    # 搜索实验
    search_parser = subparsers.add_parser('search', help='搜索实验')
    search_parser.add_argument('--dataset', type=str, help='数据集名称')
    search_parser.add_argument('--source_model', type=str, help='源模型名称(支持部分匹配)')
    search_parser.add_argument('--target_model', type=str, help='目标模型名称(支持部分匹配)')
    
    # 生成报告
    report_parser = subparsers.add_parser('report', help='生成实验报告')
    report_parser.add_argument('--dataset', type=str, help='特定数据集的报告')
    
    # 清理实验
    cleanup_parser = subparsers.add_parser('cleanup', help='清理旧实验')
    cleanup_parser.add_argument('--days', type=int, default=30, help='保留天数')
    cleanup_parser.add_argument('--no-dry-run', action='store_true', help='执行实际删除')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = ResultsManager()
    
    if args.command == 'list':
        manager.list_experiments(args.limit)
    elif args.command == 'search':
        filters = {}
        if args.dataset:
            filters['dataset'] = args.dataset
        if args.source_model:
            filters['source_model'] = args.source_model
        if args.target_model:
            filters['target_model'] = args.target_model
        
        manager.search_experiments(**filters)
    elif args.command == 'report':
        manager.generate_report(args.dataset)
    elif args.command == 'cleanup':
        manager.cleanup_old_experiments(args.days, dry_run=not args.no_dry_run)

if __name__ == "__main__":
    main()
