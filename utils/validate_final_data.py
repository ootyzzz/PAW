#!/usr/bin/env python3
"""
验证最终清理后的数据集
"""

import json
from pathlib import Path
from collections import defaultdict

def main():
    """
    验证数据集
    """
    input_file = Path("raw_datasets/commonsense/cs_all_final.jsonl")
    
    if not input_file.exists():
        print(f"文件不存在: {input_file}")
        return
    
    print(f"验证文件: {input_file}")
    
    # 统计信息
    dataset_stats = defaultdict(int)
    task_type_stats = defaultdict(int)
    total_count = 0
    error_count = 0
    
    # 收集每个数据集的样本
    dataset_samples = defaultdict(list)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                sample = json.loads(line)
                total_count += 1
                
                # 统计数据集和任务类型
                dataset = sample.get('dataset', 'unknown')
                task_type = sample.get('task_type', 'unknown')
                dataset_stats[dataset] += 1
                task_type_stats[task_type] += 1
                
                # 验证必要字段
                required_fields = ['id', 'dataset', 'task_type', 'input', 'options', 'target', 'target_idx']
                missing_fields = [field for field in required_fields if field not in sample]
                
                if missing_fields:
                    print(f"第{line_num}行缺少字段: {missing_fields}")
                    error_count += 1
                    continue
                
                # 验证数据完整性
                if not sample['input'].strip():
                    print(f"第{line_num}行input为空")
                    error_count += 1
                    continue
                
                if not isinstance(sample['options'], list):
                    print(f"第{line_num}行options不是列表")
                    error_count += 1
                    continue
                
                # 收集样本用于展示
                if len(dataset_samples[dataset]) < 2:
                    dataset_samples[dataset].append(sample)
                
            except Exception as e:
                print(f"第{line_num}行解析错误: {e}")
                error_count += 1
    
    print(f"\n=== 验证结果 ===")
    print(f"总样本数: {total_count}")
    print(f"错误样本数: {error_count}")
    print(f"成功率: {(total_count - error_count) / total_count * 100:.2f}%")
    
    print(f"\n=== 数据集统计 ===")
    for dataset, count in sorted(dataset_stats.items()):
        print(f"{dataset}: {count}")
    
    print(f"\n=== 任务类型统计 ===")
    for task_type, count in sorted(task_type_stats.items()):
        print(f"{task_type}: {count}")
    
    print(f"\n=== 各数据集样本展示 ===")
    for dataset, samples in dataset_samples.items():
        print(f"\n--- {dataset} ---")
        for i, sample in enumerate(samples, 1):
            print(f"样本{i}:")
            print(f"  ID: {sample['id']}")
            print(f"  任务类型: {sample['task_type']}")
            print(f"  输入: {sample['input'][:100]}...")
            print(f"  选项数量: {len(sample['options'])}")
            if sample['options']:
                print(f"  第一个选项: {sample['options'][0][:50]}...")
            print(f"  答案: {sample['target'][:50]}...")
            print(f"  答案索引: {sample['target_idx']}")
            print()

if __name__ == "__main__":
    main()
