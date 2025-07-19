"""
mix_commonsense_datasets.py
合并并打乱7个commonsense数据集的工具
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

def mix_commonsense_datasets(
    datasets_dir: str = "raw_datasets",
    output_dir: str = "raw_datasets/commonsense",
    seed: int = 42,
    max_samples_per_dataset: int = None
) -> Dict[str, int]:
    """
    合并并打乱7个commonsense数据集
    
    Args:
        datasets_dir: 数据集根目录
        output_dir: 输出目录
        seed: 随机种子
        max_samples_per_dataset: 每个数据集最大样本数（None表示不限制）
    
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
    
    all_data = []
    stats = {}
    
    print("🔄 开始合并数据集...")
    
    for dataset_name in datasets:
        # print(f"📂 处理数据集: {dataset_name}")
        
        # 构建训练集文件路径
        train_file = os.path.join(datasets_dir, dataset_name, f"{dataset_name}_train.jsonl")
        
        # 加载数据
        data = load_jsonl(train_file)
        
        if not data:
            print(f"⚠️  警告: {dataset_name} 训练集为空或文件不存在")
            stats[dataset_name] = 0
            continue
        
        # print(f"   加载了 {len(data)} 个样本")
        
        # 限制样本数量（如果指定）
        if max_samples_per_dataset and len(data) > max_samples_per_dataset:
            data = random.sample(data, max_samples_per_dataset)
            print(f"   随机采样 {max_samples_per_dataset} 个样本")
        
        # 标准化格式
        standardized_data = standardize_format(data, dataset_name)
        
        all_data.extend(standardized_data)
        stats[dataset_name] = len(standardized_data)
    
    # 打乱数据
    print("🔀 打乱数据顺序...")
    random.shuffle(all_data)
    
    # 保存合并后的数据集
    output_file = os.path.join(output_dir, "cs_mixed.jsonl")
    print(f"💾 保存到: {output_file}")
    save_jsonl(all_data, output_file)
    
    # 统计信息
    total_samples = len(all_data)
    stats['total'] = total_samples
    
    print("\n📊 数据集合并完成!")
    print("=" * 50)
    for dataset_name, count in stats.items():
        if dataset_name != 'total':
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            print(f"{dataset_name:15}: {count:6,} 样本 ({percentage:5.1f}%)")
    print("-" * 50)
    print(f"{'总计':15}: {total_samples:6,} 样本 (100.0%)")
    
    return stats

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="合并commonsense数据集")
    parser.add_argument("--datasets_dir", type=str, default="raw_datasets",
                       help="数据集根目录")
    parser.add_argument("--output_dir", type=str, default="raw_datasets/commonsense",
                       help="输出目录")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="每个数据集最大样本数")
    
    args = parser.parse_args()
    
    # 合并数据集
    stats = mix_commonsense_datasets(
        datasets_dir=args.datasets_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        max_samples_per_dataset=args.max_samples
    )

if __name__ == "__main__":
    main()
