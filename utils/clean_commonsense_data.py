"""
clean_commonsense_data.py
清洗和统一commonsense数据集格式，使其适合训练使用
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载JSONL文件"""
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """保存数据到JSONL文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def clean_text(text: str) -> str:
    """清理文本，去除多余的空格和特殊字符"""
    if not text:
        return ""
    
    # 去除多余的空格
    text = ' '.join(text.split())
    
    # 去除一些特殊的标记（如HellaSwag中的[header]等）
    # 但保留有意义的内容
    text = text.replace('\n', ' ').replace('\t', ' ')
    
    return text.strip()

def extract_clean_format(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    从原始数据中提取并清洗为统一格式
    
    统一格式包含以下字段：
    - id: 样本ID（如果有）
    - dataset: 数据集名称
    - task_type: 任务类型
    - input: 输入文本（问题 + 上下文）
    - options: 选项列表
    - target: 正确答案
    - target_idx: 正确答案索引
    """
    
    dataset_source = item.get('dataset_source', 'unknown')
    original_data = item.get('original_data', {})
    
    # 基础信息
    cleaned_item = {
        'id': original_data.get('id', original_data.get('ind', f"{dataset_source}_{hash(str(original_data)) % 10000}")),
        'dataset': dataset_source,
        'task_type': get_task_type(dataset_source),
    }
    
    if dataset_source in ['arc-challenge', 'arc-easy']:
        # ARC 格式处理
        question = clean_text(item.get('question', ''))
        choices = [clean_text(choice) for choice in item.get('choices', [])]
        answer = item.get('answer', '')
        answer_idx = item.get('answer_index', -1)
        
        cleaned_item.update({
            'input': question,
            'options': choices,
            'target': answer,
            'target_idx': answer_idx
        })
        
    elif dataset_source == 'boolq':
        # BoolQ 格式处理
        question = clean_text(item.get('question', ''))
        passage = clean_text(item.get('passage', ''))
        answer = item.get('answer', '')
        answer_idx = item.get('answer_index', -1)
        
        # 组合问题和上下文
        input_text = f"{passage} Question: {question}" if passage else question
        
        cleaned_item.update({
            'input': input_text,
            'options': ['False', 'True'],
            'target': answer,
            'target_idx': answer_idx
        })
        
    elif dataset_source == 'hellaswag':
        # HellaSwag 格式处理
        context = clean_text(item.get('question', ''))  # 在我们的格式中，question字段存储的是ctx
        choices = [clean_text(choice) for choice in item.get('choices', [])]
        answer = clean_text(item.get('answer', ''))
        answer_idx = item.get('answer_index', -1)
        
        cleaned_item.update({
            'input': context,
            'options': choices,
            'target': answer,
            'target_idx': answer_idx
        })
        
    elif dataset_source == 'openbookqa':
        # OpenBookQA 格式处理
        question = clean_text(item.get('question', ''))
        choices = [clean_text(choice) for choice in item.get('choices', [])]
        answer = item.get('answer', '')
        answer_idx = item.get('answer_index', -1)
        
        cleaned_item.update({
            'input': question,
            'options': choices,
            'target': answer,
            'target_idx': answer_idx
        })
        
    elif dataset_source == 'piqa':
        # PIQA 格式处理
        goal = clean_text(item.get('question', ''))  # 在我们的格式中，question字段存储的是goal
        choices = [clean_text(choice) for choice in item.get('choices', [])]
        
        # PIQA没有标准答案，我们可以设置为-1或者基于某种启发式方法
        answer_idx = item.get('answer_index', -1)
        answer = choices[answer_idx] if answer_idx >= 0 and answer_idx < len(choices) else ""
        
        cleaned_item.update({
            'input': goal,
            'options': choices,
            'target': answer,
            'target_idx': answer_idx
        })
        
    elif dataset_source == 'winogrande':
        # WinoGrande 格式处理
        sentence = clean_text(item.get('question', ''))  # 在我们的格式中，question字段存储的是sentence
        choices = [clean_text(choice) for choice in item.get('choices', [])]
        answer = item.get('answer', '')
        
        # 处理答案索引
        if answer == '1':
            answer_idx = 0
        elif answer == '2':
            answer_idx = 1
        else:
            answer_idx = item.get('answer_index', -1)
        
        target_answer = choices[answer_idx] if answer_idx >= 0 and answer_idx < len(choices) else ""
        
        cleaned_item.update({
            'input': sentence,
            'options': choices,
            'target': target_answer,
            'target_idx': answer_idx
        })
    
    else:
        # 未知格式，尽量保持原有信息
        cleaned_item.update({
            'input': clean_text(str(item.get('question', ''))),
            'options': item.get('choices', []),
            'target': str(item.get('answer', '')),
            'target_idx': item.get('answer_index', -1)
        })
    
    return cleaned_item

def get_task_type(dataset_source: str) -> str:
    """根据数据集名称返回任务类型"""
    task_types = {
        'arc-challenge': 'multiple_choice_science',
        'arc-easy': 'multiple_choice_science',
        'boolq': 'yes_no_question',
        'hellaswag': 'sentence_completion',
        'openbookqa': 'multiple_choice_science',
        'piqa': 'physical_reasoning',
        'winogrande': 'pronoun_resolution'
    }
    return task_types.get(dataset_source, 'unknown')

def validate_cleaned_item(item: Dict[str, Any]) -> bool:
    """验证清洗后的数据项是否有效"""
    required_fields = ['id', 'dataset', 'task_type', 'input', 'options', 'target', 'target_idx']
    
    # 检查必需字段
    for field in required_fields:
        if field not in item:
            return False
    
    # 检查输入是否为空
    if not item['input'].strip():
        return False
    
    # 检查选项是否为列表且不为空
    if not isinstance(item['options'], list) or len(item['options']) == 0:
        return False
    
    # 检查答案索引是否在有效范围内（-1表示无答案也是有效的）
    target_idx = item['target_idx']
    if target_idx != -1 and (target_idx < 0 or target_idx >= len(item['options'])):
        return False
    
    return True

def clean_commonsense_dataset(
    input_file: str,
    output_file: str,
    validation_split: float = 0.0
) -> Dict[str, Any]:
    """
    清洗commonsense数据集
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        validation_split: 验证集比例（0.0-1.0）
    
    Returns:
        统计信息字典
    """
    
    print(f"🔄 开始清洗数据集: {input_file}")
    
    # 加载原始数据
    raw_data = load_jsonl(input_file)
    print(f"📥 加载了 {len(raw_data)} 个原始样本")
    
    # 清洗数据
    cleaned_data = []
    invalid_count = 0
    dataset_stats = {}
    
    for i, item in enumerate(raw_data):
        try:
            cleaned_item = extract_clean_format(item)
            
            # 验证清洗后的数据
            if validate_cleaned_item(cleaned_item):
                cleaned_data.append(cleaned_item)
                
                # 统计各数据集的数量
                dataset = cleaned_item['dataset']
                dataset_stats[dataset] = dataset_stats.get(dataset, 0) + 1
            else:
                invalid_count += 1
                print(f"⚠️  样本 {i} 验证失败，跳过")
                
        except Exception as e:
            invalid_count += 1
            print(f"⚠️  样本 {i} 处理失败: {str(e)[:100]}...")
    
    print(f"✅ 清洗完成: {len(cleaned_data)} 个有效样本, {invalid_count} 个无效样本")
    
    # 分割训练集和验证集
    if validation_split > 0.0:
        import random
        random.shuffle(cleaned_data)
        split_idx = int(len(cleaned_data) * (1 - validation_split))
        
        train_data = cleaned_data[:split_idx]
        val_data = cleaned_data[split_idx:]
        
        # 保存训练集
        train_file = output_file.replace('.jsonl', '_train.jsonl')
        save_jsonl(train_data, train_file)
        print(f"💾 训练集保存到: {train_file} ({len(train_data)} 样本)")
        
        # 保存验证集
        val_file = output_file.replace('.jsonl', '_val.jsonl')
        save_jsonl(val_data, val_file)
        print(f"💾 验证集保存到: {val_file} ({len(val_data)} 样本)")
        
        final_data = train_data
    else:
        # 保存全部数据
        save_jsonl(cleaned_data, output_file)
        print(f"💾 清洗后数据保存到: {output_file}")
        final_data = cleaned_data
    
    # 统计信息
    stats = {
        'total_samples': len(final_data),
        'invalid_samples': invalid_count,
        'dataset_distribution': dataset_stats,
        'task_type_distribution': {},
        'validation_split': validation_split
    }
    
    # 统计任务类型分布
    for item in final_data:
        task_type = item['task_type']
        stats['task_type_distribution'][task_type] = stats['task_type_distribution'].get(task_type, 0) + 1
    
    # 打印统计信息
    print("\n📊 清洗后数据统计:")
    print("=" * 50)
    print("按数据集分布:")
    for dataset, count in sorted(dataset_stats.items()):
        percentage = (count / len(final_data) * 100) if len(final_data) > 0 else 0
        print(f"  {dataset:15}: {count:6,} 样本 ({percentage:5.1f}%)")
    
    print("\n按任务类型分布:")
    for task_type, count in sorted(stats['task_type_distribution'].items()):
        percentage = (count / len(final_data) * 100) if len(final_data) > 0 else 0
        print(f"  {task_type:20}: {count:6,} 样本 ({percentage:5.1f}%)")
    
    return stats

def create_sample_file(input_file: str, output_file: str, sample_size: int = 100):
    """创建样本文件用于检查清洗效果"""
    data = load_jsonl(input_file)
    if data:
        sample_data = data[:sample_size]
        save_jsonl(sample_data, output_file)
        print(f"📋 已创建样本文件: {output_file} ({len(sample_data)} 样本)")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="清洗commonsense数据集")
    parser.add_argument("--input_file", type=str, default="raw_datasets/commonsense/cs_all.jsonl",
                       help="输入文件路径")
    parser.add_argument("--output_file", type=str, default="raw_datasets/commonsense/cs_all_cleaned.jsonl",
                       help="输出文件路径")
    parser.add_argument("--validation_split", type=float, default=0.0,
                       help="验证集比例 (0.0-1.0)")
    parser.add_argument("--create_sample", action="store_true",
                       help="创建样本文件")
    parser.add_argument("--sample_size", type=int, default=100,
                       help="样本文件大小")
    
    args = parser.parse_args()
    
    # 清洗数据集
    stats = clean_commonsense_dataset(
        input_file=args.input_file,
        output_file=args.output_file,
        validation_split=args.validation_split
    )
    
    # 保存统计信息
    stats_file = args.output_file.replace('.jsonl', '_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"📈 统计信息已保存到: {stats_file}")
    
    # 创建样本文件（如果需要）
    if args.create_sample:
        sample_file = args.output_file.replace('.jsonl', '_sample.jsonl')
        create_sample_file(args.output_file, sample_file, args.sample_size)

if __name__ == "__main__":
    main()
