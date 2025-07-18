"""
format_for_training.py
将清洗后的数据进一步格式化为适合模型训练的格式
"""

import json
import os
from typing import List, Dict, Any

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载JSONL文件"""
    data = []
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

def format_for_multiple_choice(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    格式化为多选题训练格式
    
    格式：
    {
        "instruction": "Choose the best answer for the following question.",
        "input": "Question: {question}\nOptions:\nA. {option1}\nB. {option2}\n...",
        "output": "A" / "B" / "C" / "D" / "无答案"
    }
    """
    
    question = item['input']
    options = item['options']
    target_idx = item['target_idx']
    
    # 构建选项文本
    option_labels = ['A', 'B', 'C', 'D', 'E', 'F']  # 支持最多6个选项
    options_text = []
    for i, option in enumerate(options[:6]):  # 最多6个选项
        if i < len(option_labels):
            options_text.append(f"{option_labels[i]}. {option}")
    
    options_str = '\n'.join(options_text)
    
    # 构建输入文本
    input_text = f"Question: {question}\nOptions:\n{options_str}"
    
    # 确定答案
    if target_idx >= 0 and target_idx < len(option_labels) and target_idx < len(options):
        answer = option_labels[target_idx]
    else:
        answer = "无答案"
    
    return {
        "instruction": "Choose the best answer for the following question.",
        "input": input_text,
        "output": answer,
        "dataset": item['dataset'],
        "task_type": item['task_type'],
        "id": item['id']
    }

def format_for_yes_no(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    格式化为是非题训练格式
    """
    
    input_text = item['input']
    target_idx = item['target_idx']
    
    # 对于BoolQ，答案是True/False
    if target_idx == 1:
        answer = "True"
    elif target_idx == 0:
        answer = "False"
    else:
        answer = "Unknown"
    
    return {
        "instruction": "Answer the following yes/no question with True or False.",
        "input": input_text,
        "output": answer,
        "dataset": item['dataset'],
        "task_type": item['task_type'],
        "id": item['id']
    }

def format_for_completion(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    格式化为句子补全任务
    """
    
    context = item['input']
    options = item['options']
    target_idx = item['target_idx']
    
    # 构建选项文本
    option_labels = ['A', 'B', 'C', 'D', 'E', 'F']
    options_text = []
    for i, option in enumerate(options[:6]):
        if i < len(option_labels):
            options_text.append(f"{option_labels[i]}. {option}")
    
    options_str = '\n'.join(options_text)
    
    input_text = f"Context: {context}\nChoose the best continuation:\n{options_str}"
    
    # 确定答案
    if target_idx >= 0 and target_idx < len(option_labels) and target_idx < len(options):
        answer = option_labels[target_idx]
    else:
        answer = "无答案"
    
    return {
        "instruction": "Choose the best continuation for the given context.",
        "input": input_text,
        "output": answer,
        "dataset": item['dataset'],
        "task_type": item['task_type'],
        "id": item['id']
    }

def format_item_for_training(item: Dict[str, Any]) -> Dict[str, Any]:
    """根据任务类型格式化单个数据项"""
    
    task_type = item['task_type']
    
    if task_type == 'yes_no_question':
        return format_for_yes_no(item)
    elif task_type in ['sentence_completion']:
        return format_for_completion(item)
    else:
        # 其他任务类型都使用多选题格式
        return format_for_multiple_choice(item)

def create_training_format(
    input_file: str,
    output_file: str,
    max_samples: int = None
) -> Dict[str, Any]:
    """
    将清洗后的数据转换为训练格式
    
    Args:
        input_file: 输入文件（清洗后的数据）
        output_file: 输出文件
        max_samples: 最大样本数（用于创建子集）
    
    Returns:
        统计信息
    """
    
    print(f"🔄 开始格式化训练数据: {input_file}")
    
    # 加载清洗后的数据
    cleaned_data = load_jsonl(input_file)
    print(f"📥 加载了 {len(cleaned_data)} 个清洗样本")
    
    # 限制样本数量（如果指定）
    if max_samples and len(cleaned_data) > max_samples:
        import random
        random.seed(42)
        cleaned_data = random.sample(cleaned_data, max_samples)
        print(f"🎯 随机采样 {max_samples} 个样本")
    
    # 格式化数据
    training_data = []
    failed_count = 0
    
    for i, item in enumerate(cleaned_data):
        try:
            formatted_item = format_item_for_training(item)
            training_data.append(formatted_item)
        except Exception as e:
            failed_count += 1
            print(f"⚠️  样本 {i} 格式化失败: {str(e)[:100]}...")
    
    print(f"✅ 格式化完成: {len(training_data)} 个训练样本, {failed_count} 个失败")
    
    # 保存训练数据
    save_jsonl(training_data, output_file)
    print(f"💾 训练数据保存到: {output_file}")
    
    # 统计信息
    stats = {
        'total_samples': len(training_data),
        'failed_samples': failed_count,
        'dataset_distribution': {},
        'task_type_distribution': {},
        'instruction_types': {}
    }
    
    # 统计分布
    for item in training_data:
        dataset = item['dataset']
        task_type = item['task_type']
        instruction = item['instruction']
        
        stats['dataset_distribution'][dataset] = stats['dataset_distribution'].get(dataset, 0) + 1
        stats['task_type_distribution'][task_type] = stats['task_type_distribution'].get(task_type, 0) + 1
        stats['instruction_types'][instruction] = stats['instruction_types'].get(instruction, 0) + 1
    
    # 打印统计信息
    print("\n📊 训练数据统计:")
    print("=" * 60)
    print("按数据集分布:")
    for dataset, count in sorted(stats['dataset_distribution'].items()):
        percentage = (count / len(training_data) * 100) if len(training_data) > 0 else 0
        print(f"  {dataset:15}: {count:6,} 样本 ({percentage:5.1f}%)")
    
    print("\n按任务类型分布:")
    for task_type, count in sorted(stats['task_type_distribution'].items()):
        percentage = (count / len(training_data) * 100) if len(training_data) > 0 else 0
        print(f"  {task_type:20}: {count:6,} 样本 ({percentage:5.1f}%)")
    
    print("\n按指令类型分布:")
    for instruction, count in sorted(stats['instruction_types'].items()):
        percentage = (count / len(training_data) * 100) if len(training_data) > 0 else 0
        print(f"  {count:6,} 样本 ({percentage:5.1f}%): {instruction[:50]}...")
    
    return stats

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="格式化数据为训练格式")
    parser.add_argument("--input_file", type=str, default="raw_datasets/commonsense/cs_all_cleaned.jsonl",
                       help="输入文件路径（清洗后的数据）")
    parser.add_argument("--output_file", type=str, default="raw_datasets/commonsense/cs_all_training.jsonl",
                       help="输出文件路径")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="最大样本数")
    
    args = parser.parse_args()
    
    # 创建训练格式
    stats = create_training_format(
        input_file=args.input_file,
        output_file=args.output_file,
        max_samples=args.max_samples
    )
    
    # 保存统计信息
    stats_file = args.output_file.replace('.jsonl', '_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"📈 统计信息已保存到: {stats_file}")

if __name__ == "__main__":
    main()
