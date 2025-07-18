"""
create_balanced_commonsense.py
创建平衡的commonsense数据集，每个数据集的样本数量相等
"""

import json
import random
import os
from pathlib import Path
from typing import List, Dict, Any

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载JSONL文件，支持不同格式（JSON和Python字典格式）"""
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        # 先尝试标准JSON解析
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        try:
                            # 如果失败，尝试eval解析（处理单引号格式，如ARC数据集）
                            data.append(eval(line))
                        except Exception as e:
                            print(f"⚠️  警告: 无法解析第{line_num}行: {str(e)[:100]}...")
                            continue
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """保存数据到JSONL文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def standardize_format(data: List[Dict[str, Any]], dataset_name: str) -> List[Dict[str, Any]]:
    """
    标准化不同数据集的格式
    统一转换为包含以下字段的格式：
    - dataset_source: 数据集来源
    - question: 问题文本
    - choices: 选择项列表（如果有）
    - answer: 答案
    - original_data: 原始数据（保留完整信息）
    """
    standardized = []
    
    for item in data:
        standard_item = {
            "dataset_source": dataset_name,
            "original_data": item.copy()
        }
        
        if dataset_name in ['arc-challenge', 'arc-easy']:
            # ARC格式: question, choices{text, label}, answerKey
            standard_item.update({
                "question": item["question"],
                "choices": item["choices"]["text"],
                "answer": item["answerKey"],
                "answer_index": ord(item["answerKey"]) - ord('A')
            })
            
        elif dataset_name == 'boolq':
            # BoolQ格式: question, answer (true/false), passage
            standard_item.update({
                "question": item["question"],
                "choices": ["False", "True"],
                "answer": "True" if item["answer"] else "False",
                "answer_index": 1 if item["answer"] else 0,
                "passage": item.get("passage", "")
            })
            
        elif dataset_name == 'hellaswag':
            # HellaSwag格式: ctx, endings, label
            standard_item.update({
                "question": item["ctx"],
                "choices": item["endings"],
                "answer_index": int(item["label"]) if item["label"] != "" else -1,
                "answer": item["endings"][int(item["label"])] if item["label"] != "" and item["label"].isdigit() else ""
            })
            
        elif dataset_name == 'openbookqa':
            # OpenBookQA格式: question_stem, choices{text, label}, answerKey
            standard_item.update({
                "question": item["question_stem"],
                "choices": item["choices"]["text"],
                "answer": item["answerKey"],
                "answer_index": ord(item["answerKey"]) - ord('A')
            })
            
        elif dataset_name == 'piqa':
            # PIQA格式: goal, sol1, sol2 (没有label，这是一个选择题但没有标准答案)
            standard_item.update({
                "question": item["goal"],
                "choices": [item["sol1"], item["sol2"]],
                "answer_index": -1,  # PIQA没有标准答案标签
                "answer": ""  # 没有预定义答案
            })
            
        elif dataset_name == 'winogrande':
            # Winogrande格式: sentence, option1, option2, answer
            standard_item.update({
                "question": item["sentence"],
                "choices": [item["option1"], item["option2"]],
                "answer": item["answer"],
                "answer_index": 0 if item["answer"] == "1" else 1 if item["answer"] == "2" else -1
            })
        
        standardized.append(standard_item)
    
    return standardized

def create_balanced_dataset(
    datasets_dir: str = "raw_datasets",
    output_dir: str = "raw_datasets/commonsense",
    samples_per_dataset: int = None,
    seed: int = 42
) -> Dict[str, int]:
    """
    创建平衡的commonsense数据集，每个数据集的样本数量相等
    
    Args:
        datasets_dir: 数据集根目录
        output_dir: 输出目录
        samples_per_dataset: 每个数据集的样本数（None表示使用最小数据集的大小）
        seed: 随机种子
    
    Returns:
        统计信息字典
    """
    
    # 设置随机种子
    random.seed(seed)
    
    # 定义数据集列表
    datasets = [
        'arc-challenge', 'arc-easy', 'boolq', 'hellaswag', 
        'openbookqa', 'piqa', 'winogrande'
    ]
    
    dataset_sizes = {}
    all_datasets_data = {}
    
    print("🔄 分析各数据集大小...")
    
    # 首先分析所有数据集的大小
    for dataset_name in datasets:
        train_file = os.path.join(datasets_dir, dataset_name, f"{dataset_name}_train.jsonl")
        data = load_jsonl(train_file)
        
        if not data:
            print(f"⚠️  警告: {dataset_name} 训练集为空或文件不存在")
            continue
            
        dataset_sizes[dataset_name] = len(data)
        all_datasets_data[dataset_name] = data
        print(f"📂 {dataset_name}: {len(data)} 样本")
    
    # 确定每个数据集的采样数量
    if samples_per_dataset is None:
        # 使用最小数据集的大小
        samples_per_dataset = min(dataset_sizes.values())
        print(f"\n🎯 自动设置每个数据集样本数为: {samples_per_dataset} (最小数据集大小)")
    else:
        print(f"\n🎯 设置每个数据集样本数为: {samples_per_dataset}")
    
    balanced_data = []
    balanced_stats = {}
    
    print("\n🔄 创建平衡数据集...")
    
    for dataset_name in datasets:
        if dataset_name not in all_datasets_data:
            balanced_stats[dataset_name] = 0
            continue
            
        data = all_datasets_data[dataset_name]
        
        # 如果数据量大于目标数量，随机采样
        if len(data) > samples_per_dataset:
            sampled_data = random.sample(data, samples_per_dataset)
            print(f"📊 {dataset_name}: 从 {len(data)} 个样本中随机采样 {samples_per_dataset} 个")
        else:
            sampled_data = data
            print(f"📊 {dataset_name}: 使用全部 {len(data)} 个样本")
        
        # 标准化格式
        standardized_data = standardize_format(sampled_data, dataset_name)
        balanced_data.extend(standardized_data)
        balanced_stats[dataset_name] = len(standardized_data)
    
    # 打乱数据
    print("🔀 打乱数据顺序...")
    random.shuffle(balanced_data)
    
    # 保存平衡数据集
    output_file = os.path.join(output_dir, "cs_balanced.jsonl")
    print(f"💾 保存到: {output_file}")
    save_jsonl(balanced_data, output_file)
    
    # 统计信息
    total_samples = len(balanced_data)
    balanced_stats['total'] = total_samples
    
    print("\n📊 平衡数据集创建完成!")
    print("=" * 50)
    for dataset_name, count in balanced_stats.items():
        if dataset_name != 'total':
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            print(f"{dataset_name:15}: {count:6,} 样本 ({percentage:5.1f}%)")
    print("-" * 50)
    print(f"{'总计':15}: {total_samples:6,} 样本 (100.0%)")
    
    return balanced_stats

def update_stats_file(output_dir: str, all_stats: Dict[str, int], balanced_stats: Dict[str, int]):
    """更新统计文件，包含全量和平衡数据集的统计信息"""
    
    combined_stats = {
        "cs_all": all_stats,
        "cs_balanced": balanced_stats,
        "description": {
            "cs_all": "包含所有7个数据集的完整训练数据",
            "cs_balanced": "每个数据集样本数量相等的平衡数据集"
        }
    }
    
    stats_file = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(combined_stats, f, indent=2, ensure_ascii=False)
    
    print(f"📈 统计信息已更新到: {stats_file}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="创建平衡的commonsense数据集")
    parser.add_argument("--datasets_dir", type=str, default="raw_datasets",
                       help="数据集根目录")
    parser.add_argument("--output_dir", type=str, default="raw_datasets/commonsense",
                       help="输出目录")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--samples_per_dataset", type=int, default=None,
                       help="每个数据集的样本数（None表示使用最小数据集的大小）")
    
    args = parser.parse_args()
    
    # 创建平衡数据集
    balanced_stats = create_balanced_dataset(
        datasets_dir=args.datasets_dir,
        output_dir=args.output_dir,
        samples_per_dataset=args.samples_per_dataset,
        seed=args.seed
    )
    
    # 读取现有的全量数据集统计信息
    stats_file = os.path.join(args.output_dir, "dataset_stats.json")
    if os.path.exists(stats_file):
        with open(stats_file, 'r', encoding='utf-8') as f:
            all_stats = json.load(f)
    else:
        # 如果没有现有统计信息，创建一个默认的
        all_stats = {
            "arc-challenge": 1119,
            "arc-easy": 2251,
            "boolq": 9427,
            "hellaswag": 39905,
            "openbookqa": 4957,
            "piqa": 16113,
            "winogrande": 40398,
            "total": 114170
        }
    
    # 更新统计文件
    update_stats_file(args.output_dir, all_stats, balanced_stats)

if __name__ == "__main__":
    main()
