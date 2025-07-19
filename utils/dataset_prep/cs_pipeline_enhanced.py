"""
扩展的Commonsense数据集处理Pipeline
处理所有splits: train, validation, test

这是一个数据预处理脚本，用于处理7个commonsense数据集的所有splits。
支持individual数据集处理和mixed数据集生成。

参数使用指南:
===============

基础使用:
# 默认处理所有数据集的所有splits (train/validation/test)，生成individual和mixed数据
python utils/dataset_prep/cs_pipeline_all_splits.py

# 只处理特定数据集
python utils/dataset_prep/cs_pipeline_all_splits.py --datasets arc-challenge arc-easy boolq

# 只处理特定splits  
python utils/dataset_prep/cs_pipeline_all_splits.py --splits train validation

# 只生成mixed数据，不生成individual数据集
python utils/dataset_prep/cs_pipeline_all_splits.py --mixed-only

# 只生成individual数据集，不生成mixed数据
python utils/dataset_prep/cs_pipeline_all_splits.py --individual-only

# 使用自定义目录
python utils/dataset_prep/cs_pipeline_all_splits.py --datasets_dir ./my_raw_data --output_dir ./my_output

# 验证模式 - 只检查数据格式，不生成文件
python utils/dataset_prep/cs_pipeline_all_splits.py --validate_only

# 设置随机种子
python utils/dataset_prep/cs_pipeline_all_splits.py --seed 123

可用数据集:
- arc-challenge: AI2 Reasoning Challenge (挑战版)
- arc-easy: AI2 Reasoning Challenge (简单版) 
- boolq: Boolean Questions
- hellaswag: HellaSwag常识推理
- openbookqa: Open Book Question Answering
- piqa: Physical Interaction QA
- winogrande: Winogrande代词解析

输出文件结构:
data_to_lora/cs/
├── mixed/
│   ├── cs_mixed_train.jsonl                    # 原始混合训练数据
│   ├── cs_mixed_formatted_train.jsonl         # 格式化混合训练数据 
│   ├── cs_mixed_validation.jsonl              # 原始混合验证数据
│   ├── cs_mixed_formatted_validation.jsonl    # 格式化混合验证数据
│   ├── cs_mixed_test.jsonl                    # 原始混合测试数据
│   └── cs_mixed_formatted_test.jsonl          # 格式化混合测试数据
├── arc-challenge/
│   ├── arc-challenge_train_formatted.jsonl
│   ├── arc-challenge_validation_formatted.jsonl
│   └── arc-challenge_test_formatted.jsonl
├── arc-easy/
│   └── ...
└── [其他数据集]/
    └── ...

使用场景:
- 首次设置: 运行默认命令处理所有数据
- 增量更新: 指定特定数据集重新处理
- 调试验证: 使用 --validate_only 检查数据质量
- 自定义训练: 选择特定数据集和splits组合
"""

import json
import random
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import argparse

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
    """标准化不同数据集的格式"""
    standardized = []
    
    for i, item in enumerate(data):
        standard_item = {
            "dataset_source": dataset_name,
            "original_data": item
        }
        
        if dataset_name in ['arc-challenge', 'arc-easy']:
            # ARC格式: question, choices, answerKey
            choices_dict = item.get("choices", {})
            choices_list = []
            if isinstance(choices_dict, dict):
                labels = choices_dict.get("label", [])
                texts = choices_dict.get("text", [])
                choices_list = [f"{label}: {text}" for label, text in zip(labels, texts)]
            
            standard_item.update({
                "question": item.get("question", ""),
                "choices": choices_list,
                "answer": item.get("answerKey", ""),
                "answer_index": ord(item.get("answerKey", "A")) - ord('A') if item.get("answerKey") else -1
            })
            
        elif dataset_name == 'boolq':
            # BoolQ格式: question, passage, answer
            standard_item.update({
                "question": f"Question: {item.get('question', '')}\nPassage: {item.get('passage', '')}",
                "choices": ["False", "True"],
                "answer": str(item.get("answer", False)),
                "answer_index": 1 if item.get("answer") else 0
            })
            
        elif dataset_name == 'hellaswag':
            # HellaSwag格式: ctx, endings, label
            ctx = item.get("ctx", "")
            endings = item.get("endings", [])
            label = item.get("label")
            
            # 处理label可能是字符串的情况
            if isinstance(label, str) and label.isdigit():
                label_idx = int(label)
            elif isinstance(label, int):
                label_idx = label
            else:
                label_idx = -1
            
            standard_item.update({
                "question": ctx,
                "choices": endings,
                "answer": endings[label_idx] if endings and label_idx >= 0 and label_idx < len(endings) else "",
                "answer_index": label_idx
            })
            
        elif dataset_name == 'openbookqa':
            # OpenBookQA格式: question_stem, choices, answerKey
            choices_dict = item.get("choices", {})
            choices_list = []
            if isinstance(choices_dict, dict):
                labels = choices_dict.get("label", [])
                texts = choices_dict.get("text", [])
                choices_list = [f"{label}: {text}" for label, text in zip(labels, texts)]
            
            standard_item.update({
                "question": item.get("question_stem", ""),
                "choices": choices_list,
                "answer": item.get("answerKey", ""),
                "answer_index": ord(item.get("answerKey", "A")) - ord('A') if item.get("answerKey") else -1
            })
            
        elif dataset_name == 'piqa':
            # PIQA格式: goal, sol1, sol2 (没有label)
            standard_item.update({
                "question": item.get("goal", ""),
                "choices": [item.get("sol1", ""), item.get("sol2", "")],
                "answer_index": -1,  # PIQA没有标准答案标签
                "answer": ""  # 没有预定义答案
            })
            
        elif dataset_name == 'winogrande':
            # Winogrande格式: sentence, option1, option2, answer
            standard_item.update({
                "question": item.get("sentence", ""),
                "choices": [item.get("option1", ""), item.get("option2", "")],
                "answer": item.get("answer", ""),
                "answer_index": 0 if item.get("answer") == "1" else 1 if item.get("answer") == "2" else -1
            })
        
        standardized.append(standard_item)
    
    return standardized

def format_to_final(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """将数据格式化为最终的标准格式"""
    formatted_data = []
    
    for i, sample in enumerate(data):
        dataset = sample.get('dataset_source', '')
        original_data = sample.get('original_data', sample)
        
        # 生成统一ID
        if dataset == 'hellaswag':
            original_id = original_data.get('ind', f'h_{i}')
            unified_id = f"hellaswag_{original_id}"
            task_type = "sentence_completion"
        elif dataset == 'winogrande':
            original_id = hash(str(original_data.get('sentence', ''))) % 100000
            unified_id = f"winogrande_{original_id}"
            task_type = "pronoun_resolution"
        elif dataset == 'piqa':
            original_id = hash(str(original_data.get('goal', ''))) % 100000
            unified_id = f"piqa_{original_id}"
            task_type = "physical_reasoning"
        elif dataset == 'boolq':
            original_id = hash(str(original_data.get('question', ''))) % 100000
            unified_id = f"boolq_{original_id}"
            task_type = "yes_no_question"
        elif dataset in ['arc-challenge', 'arc-easy']:
            original_id = original_data.get('id', f'arc_{i}')
            unified_id = f"{dataset}_{original_id}"
            task_type = "multiple_choice"
        elif dataset == 'openbookqa':
            original_id = original_data.get('id', f'obqa_{i}')
            unified_id = f"openbookqa_{original_id}"
            task_type = "multiple_choice"
        else:
            unified_id = f"{dataset}_{i}"
            task_type = "unknown"
        
        # 构建最终格式
        choices = sample.get('choices', [])
        answer_idx = sample.get('answer_index', -1)
        answer_text = sample.get('answer', '')
        
        # 如果answer_text为空但有answer_index，从choices中获取
        if not answer_text and answer_idx >= 0 and answer_idx < len(choices):
            answer_text = choices[answer_idx]
        
        formatted_sample = {
            "id": unified_id,
            "dataset": dataset,
            "task_type": task_type,
            "input": sample.get('question', ''),
            "options": choices,
            "target": answer_text,
            "target_idx": answer_idx
        }
        
        formatted_data.append(formatted_sample)
    
    return formatted_data

def process_single_split(datasets_dir: str, split_name: str, output_dir: str, datasets_filter: List[str] = None, seed: int = 42):
    """处理单个split（train/validation/test）"""
    
    # 使用过滤器或默认所有数据集
    if datasets_filter:
        datasets = datasets_filter
    else:
        datasets = [
            'arc-challenge', 'arc-easy', 'boolq', 'hellaswag', 
            'openbookqa', 'piqa', 'winogrande'
        ]
    
    all_data = []
    stats = {}
    
    print(f"🔄 开始处理 {split_name} split...")
    
    for dataset_name in datasets:
        # 构建文件路径
        file_name = f"{dataset_name}_{split_name}.jsonl"
        if split_name == "validation":
            # 有些数据集可能用val而不是validation
            val_file = os.path.join(datasets_dir, dataset_name, f"{dataset_name}_val.jsonl")
            validation_file = os.path.join(datasets_dir, dataset_name, f"{dataset_name}_validation.jsonl")
            
            if os.path.exists(validation_file):
                file_path = validation_file
            elif os.path.exists(val_file):
                file_path = val_file
            else:
                print(f"⚠️  警告: {dataset_name} 没有找到validation文件")
                stats[dataset_name] = 0
                continue
        else:
            file_path = os.path.join(datasets_dir, dataset_name, file_name)
        
        # 加载数据
        data = load_jsonl(file_path)
        
        if not data:
            print(f"⚠️  警告: {dataset_name} {split_name}集为空或文件不存在")
            stats[dataset_name] = 0
            continue
        
        # 标准化格式
        standardized_data = standardize_format(data, dataset_name)
        
        all_data.extend(standardized_data)
        stats[dataset_name] = len(standardized_data)
        print(f"   {dataset_name}: {len(standardized_data)} 样本")
    
    if not all_data:
        print(f"❌ {split_name} split没有找到任何数据")
        return stats
    
    # 打乱数据（可选，对于test通常不打乱）
    if split_name == "train":
        print("🔀 打乱数据顺序...")
        random.seed(seed)
        random.shuffle(all_data)
    
    # 保存混合数据（先保存到mixed子目录）
    mixed_output_dir = os.path.join(output_dir, "mixed")
    mixed_output_file = os.path.join(mixed_output_dir, f"cs_mixed_{split_name}.jsonl")
    save_jsonl(all_data, mixed_output_file)
    print(f"✅ 混合数据已保存到: {mixed_output_file}")
    
    # 格式化为最终格式
    print("🔄 格式化为最终格式...")
    formatted_data = format_to_final(all_data)
    
    # 保存格式化数据
    formatted_output_file = os.path.join(mixed_output_dir, f"cs_mixed_formatted_{split_name}.jsonl")
    save_jsonl(formatted_data, formatted_output_file)
    print(f"✅ 格式化数据已保存到: {formatted_output_file}")
    
    total_samples = sum(stats.values())
    stats['total'] = total_samples
    
    print(f"\n📊 {split_name.upper()} 数据集统计:")
    print("=" * 50)
    for dataset_name, count in stats.items():
        if dataset_name != 'total':
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            print(f"{dataset_name:15}: {count:6,} 样本 ({percentage:5.1f}%)")
    print("-" * 50)
    print(f"{'总计':15}: {total_samples:6,} 样本 (100.0%)")
    print()
    
    return stats

def process_individual_datasets(datasets_dir: str, output_base_dir: str, datasets_filter: List[str] = None, splits_filter: List[str] = None):
    """处理各个数据集的individual splits"""
    
    # 使用过滤器或默认值
    if datasets_filter:
        datasets = datasets_filter
    else:
        datasets = [
            'arc-challenge', 'arc-easy', 'boolq', 'hellaswag', 
            'openbookqa', 'piqa', 'winogrande'
        ]
    
    if splits_filter:
        splits = splits_filter
    else:
        splits = ['train', 'validation', 'test']
    
    print("🔄 开始处理各数据集的individual splits...")
    
    for dataset_name in datasets:
        print(f"\n📂 处理数据集: {dataset_name}")
        
        for split_name in splits:
            # 构建输入文件路径
            file_name = f"{dataset_name}_{split_name}.jsonl"
            if split_name == "validation":
                # 检查validation或val文件
                val_file = os.path.join(datasets_dir, dataset_name, f"{dataset_name}_val.jsonl")
                validation_file = os.path.join(datasets_dir, dataset_name, f"{dataset_name}_validation.jsonl")
                
                if os.path.exists(validation_file):
                    input_file = validation_file
                elif os.path.exists(val_file):
                    input_file = val_file
                else:
                    print(f"   ⚠️  跳过: 没有找到validation文件")
                    continue
            else:
                input_file = os.path.join(datasets_dir, dataset_name, file_name)
            
            if not os.path.exists(input_file):
                print(f"   ⚠️  跳过: {split_name} 文件不存在")
                continue
            
            # 加载数据
            data = load_jsonl(input_file)
            if not data:
                print(f"   ⚠️  跳过: {split_name} 文件为空")
                continue
            
            # 标准化格式
            standardized_data = standardize_format(data, dataset_name)
            
            # 格式化为最终格式
            formatted_data = format_to_final(standardized_data)
            
            # 构建输出路径
            output_dir = os.path.join(output_base_dir, dataset_name)
            output_file = os.path.join(output_dir, f"{dataset_name}_{split_name}_formatted.jsonl")
            
            # 保存格式化数据
            save_jsonl(formatted_data, output_file)
            print(f"   ✅ {split_name}: {len(formatted_data)} 样本 -> {output_file}")

def validate_datasets(datasets_dir: str, datasets: List[str], splits: List[str]) -> bool:
    """验证数据集文件是否存在和格式是否正确"""
    print("🔍 验证数据集文件...")
    
    all_valid = True
    for dataset_name in datasets:
        print(f"\n📂 验证 {dataset_name}:")
        
        for split_name in splits:
            # 构建文件路径
            file_name = f"{dataset_name}_{split_name}.jsonl"
            if split_name == "validation":
                val_file = os.path.join(datasets_dir, dataset_name, f"{dataset_name}_val.jsonl")
                validation_file = os.path.join(datasets_dir, dataset_name, f"{dataset_name}_validation.jsonl")
                
                if os.path.exists(validation_file):
                    file_path = validation_file
                elif os.path.exists(val_file):
                    file_path = val_file
                else:
                    print(f"   ❌ {split_name}: 文件不存在")
                    all_valid = False
                    continue
            else:
                file_path = os.path.join(datasets_dir, dataset_name, file_name)
            
            if not os.path.exists(file_path):
                print(f"   ❌ {split_name}: 文件不存在 ({file_path})")
                all_valid = False
                continue
            
            # 尝试加载并验证格式
            try:
                data = load_jsonl(file_path)
                if not data:
                    print(f"   ⚠️ {split_name}: 文件为空")
                else:
                    print(f"   ✅ {split_name}: {len(data)} 样本")
            except Exception as e:
                print(f"   ❌ {split_name}: 格式错误 - {e}")
                all_valid = False
    
    if all_valid:
        print("\n✅ 所有数据集验证通过")
    else:
        print("\n❌ 数据集验证失败")
    
    return all_valid

def show_processing_plan(datasets: List[str], splits: List[str], process_mixed: bool, process_individual: bool, output_dir: str):
    """显示处理计划"""
    print("📋 处理计划:")
    print(f"  数据集: {datasets}")
    print(f"  Splits: {splits}")
    print(f"  输出目录: {output_dir}")
    print()
    
    if process_mixed:
        print("🔄 将生成混合数据集:")
        for split in splits:
            print(f"  - {output_dir}/mixed/cs_mixed_{split}.jsonl")
            print(f"  - {output_dir}/mixed/cs_mixed_formatted_{split}.jsonl")
        print()
    
    if process_individual:
        print("🔄 将生成个别数据集:")
        for dataset in datasets:
            for split in splits:
                print(f"  - {output_dir}/{dataset}/{dataset}_{split}_formatted.jsonl")
    
    print("🏃 这是dry run模式，不会实际处理文件")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="扩展的Commonsense数据集处理Pipeline")
    parser.add_argument("--datasets", nargs='*', 
                       help="要处理的数据集列表。默认处理所有7个数据集。可选: arc-challenge, arc-easy, boolq, hellaswag, openbookqa, piqa, winogrande")
    parser.add_argument("--splits", nargs='*', default=['train', 'validation', 'test'],
                       help="要处理的splits列表。默认: train validation test")
    parser.add_argument("--datasets_dir", type=str, default="raw_datasets/cs",
                       help="原始数据集根目录")
    parser.add_argument("--output_dir", type=str, default="data_to_lora/cs",
                       help="输出基目录")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子用于打乱数据")
    parser.add_argument("--mixed_only", action="store_true",
                       help="只生成mixed数据，不处理individual数据集")
    parser.add_argument("--individual_only", action="store_true",
                       help="只处理individual数据集，不生成mixed数据")
    parser.add_argument("--validate_only", action="store_true",
                       help="仅验证数据格式，不生成输出文件")
    parser.add_argument("--dry_run", action="store_true",
                       help="干运行，显示处理计划但不实际处理")
    
    args = parser.parse_args()
    
    # 所有可用的数据集
    all_datasets = [
        'arc-challenge', 'arc-easy', 'boolq', 'hellaswag', 
        'openbookqa', 'piqa', 'winogrande'
    ]
    
    # 确定要处理的数据集
    if args.datasets is None or len(args.datasets) == 0:
        # 默认：处理所有数据集
        datasets_to_process = all_datasets
        print("🚀 Commonsense数据集处理Pipeline")
        print("=" * 70)
        print(f"原始数据目录: {args.datasets_dir}")
        print(f"输出目录: {args.output_dir}")
        print(f"处理模式: 所有数据集 (默认)")
        print(f"处理splits: {args.splits}")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
    else:
        # 验证用户指定的数据集
        datasets_to_process = []
        for dataset in args.datasets:
            if dataset in all_datasets:
                datasets_to_process.append(dataset)
            else:
                print(f"⚠️ 警告: 未知数据集 '{dataset}'，将被跳过")
        
        if not datasets_to_process:
            print("❌ 错误: 没有有效的数据集要处理")
            return False
        
        print("🚀 Commonsense数据集处理Pipeline")
        print("=" * 70)
        print(f"原始数据目录: {args.datasets_dir}")
        print(f"输出目录: {args.output_dir}")
        print(f"处理数据集: {datasets_to_process}")
        print(f"处理splits: {args.splits}")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
    
    # 检查互斥参数
    if args.mixed_only and args.individual_only:
        print("❌ 错误: --mixed_only 和 --individual_only 不能同时使用")
        return False
    
    # 确定处理模式
    process_mixed = not args.individual_only
    process_individual = not args.mixed_only
    
    if args.validate_only:
        print("\n🔍 验证模式: 检查数据格式和完整性...")
        return validate_datasets(args.datasets_dir, datasets_to_process, args.splits)
    
    if args.dry_run:
        print("\n🏃 Dry Run模式: 显示处理计划...")
        show_processing_plan(datasets_to_process, args.splits, process_mixed, process_individual, args.output_dir)
        return True
    
    try:
        success = True
        all_stats = {}
        
        # 1. 处理混合数据集
        if process_mixed:
            print("\n" + "=" * 70)
            print("🔄 步骤1: 处理混合数据集...")
            print("=" * 70)
            
            for split_name in args.splits:
                print(f"\n📋 处理 {split_name.upper()} split...")
                stats = process_single_split(
                    datasets_dir=args.datasets_dir,
                    split_name=split_name,
                    output_dir=args.output_dir,
                    datasets_filter=datasets_to_process,
                    seed=args.seed
                )
                all_stats[split_name] = stats
        
        # 2. 处理个别数据集
        if process_individual:
            print("\n" + "=" * 70)
            print("🔄 步骤2: 处理个别数据集...")
            print("=" * 70)
            process_individual_datasets(
                datasets_dir=args.datasets_dir,
                output_base_dir=args.output_dir,
                datasets_filter=datasets_to_process,
                splits_filter=args.splits
            )
        
        # 3. 生成总体统计
        print("\n" + "=" * 70)
        print("🎉 处理完成!")
        print("=" * 70)
        
        if process_mixed and all_stats:
            print("\n📈 混合数据集统计:")
            for split_name, stats in all_stats.items():
                total = stats.get('total', 0)
                print(f"  {split_name:12}: {total:6,} 样本")
        
        print(f"\n✅ 所有数据已保存到: {args.output_dir}")
        
        return success
        
    except Exception as e:
        print(f"\n❌ 处理过程中发生错误: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
