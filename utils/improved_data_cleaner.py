"""
improved_data_cleaner.py
改进版数据清洗工具，进一步统一格式
"""

import json
import random
import os
from pathlib import Path
from typing import List, Dict, Any

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

def clean_and_standardize_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    进一步清理和标准化单个样本
    统一格式：
    - id: 统一为字符串格式 dataset_originalid
    - dataset: 数据集名称
    - task_type: 任务类型
    - input: 清理后的输入文本
    - options: 选项列表
    - target: 答案文本
    - target_idx: 答案索引
    """
    
    dataset = sample.get('dataset_source', sample.get('dataset', ''))
    original_data = sample.get('original_data', sample)
    
    # 如果已经是清洗后的格式，直接使用
    if 'input' in sample and 'options' in sample:
        # 统一ID格式
        original_id = sample.get('id', hash(str(sample)) % 100000)
        unified_id = f"{dataset}_{original_id}"
        
        return {
            "id": unified_id,
            "dataset": dataset,
            "task_type": sample.get('task_type', 'unknown'),
            "input": sample.get('input', '').strip(),
            "options": sample.get('options', []),
            "target": sample.get('target', ''),
            "target_idx": sample.get('target_idx', -1)
        }
    
    # 否则从original_data解析
    # 统一ID格式
    if dataset == 'hellaswag':
        original_id = original_data.get('ind', sample.get('id', ''))
        unified_id = f"hellaswag_{original_id}"
    elif dataset == 'winogrande':
        original_id = hash(original_data.get('sentence', '')) % 100000
        unified_id = f"winogrande_{original_id}"
    elif dataset == 'piqa':
        original_id = hash(original_data.get('goal', '')) % 100000
        unified_id = f"piqa_{original_id}"
    elif dataset == 'boolq':
        original_id = hash(original_data.get('question', '')) % 100000
        unified_id = f"boolq_{original_id}"
    elif dataset in ['arc-challenge', 'arc-easy']:
        original_id = original_data.get('id', sample.get('id', ''))
        unified_id = f"{dataset}_{original_id}".replace('/', '_')
    elif dataset == 'openbookqa':
        original_id = original_data.get('id', sample.get('id', ''))
        unified_id = f"openbookqa_{original_id}"
    else:
        unified_id = f"{dataset}_{hash(str(sample)) % 100000}"
    
    # 获取基本信息
    if dataset in ['arc-challenge', 'arc-easy']:
        input_text = original_data.get('question', '')
        options = original_data.get('choices', {}).get('text', [])
        answer_key = original_data.get('answerKey', '')
        target_idx = ord(answer_key) - ord('A') if answer_key else -1
        target = options[target_idx] if 0 <= target_idx < len(options) else ""
        task_type = "multiple_choice"
        
    elif dataset == 'boolq':
        input_text = original_data.get('question', '')
        options = ["False", "True"]
        answer = original_data.get('answer', False)
        target_idx = 1 if answer else 0
        target = "True" if answer else "False"
        task_type = "yes_no_question"
        
    elif dataset == 'hellaswag':
        input_text = original_data.get('ctx', '')
        options = original_data.get('endings', [])
        label = original_data.get('label', '')
        target_idx = int(label) if label.isdigit() else -1
        target = options[target_idx] if 0 <= target_idx < len(options) else ""
        task_type = "sentence_completion"
        
    elif dataset == 'openbookqa':
        input_text = original_data.get('question_stem', '')
        options = original_data.get('choices', {}).get('text', [])
        answer_key = original_data.get('answerKey', '')
        target_idx = ord(answer_key) - ord('A') if answer_key else -1
        target = options[target_idx] if 0 <= target_idx < len(options) else ""
        task_type = "multiple_choice"
        
    elif dataset == 'piqa':
        input_text = original_data.get('goal', '')
        options = [original_data.get('sol1', ''), original_data.get('sol2', '')]
        target_idx = -1  # PIQA通常没有标准答案标签
        target = ""  # 没有标准答案
        task_type = "physical_reasoning"
        
    elif dataset == 'winogrande':
        input_text = original_data.get('sentence', '')
        options = [original_data.get('option1', ''), original_data.get('option2', '')]
        answer = original_data.get('answer', '')
        target_idx = 0 if answer == "1" else 1 if answer == "2" else -1
        target = options[target_idx] if 0 <= target_idx < len(options) else ""
        task_type = "pronoun_resolution"
        
    else:
        # 默认处理
        input_text = sample.get('question', sample.get('input', ''))
        options = sample.get('choices', sample.get('options', []))
        target_idx = sample.get('answer_index', sample.get('target_idx', -1))
        target = sample.get('answer', sample.get('target', ''))
        task_type = "unknown"
    
    # 清理输入文本
    input_text = input_text.strip()
    if len(input_text) > 1000:  # 限制输入长度
        input_text = input_text[:1000] + "..."
    
    # 清理选项
    options = [opt.strip() for opt in options if opt and opt.strip()]
    
    return {
        "id": unified_id,
        "dataset": dataset,
        "task_type": task_type,
        "input": input_text,
        "options": options,
        "target": target,
        "target_idx": target_idx
    }

def improve_cleaned_data(input_file: str, output_file: str):
    """进一步改进已清洗的数据"""
    
    print(f"🔄 加载数据: {input_file}")
    data = load_jsonl(input_file)
    
    print(f"📊 处理 {len(data)} 个样本...")
    
    improved_data = []
    stats = {}
    
    for i, sample in enumerate(data):
        try:
            improved_sample = clean_and_standardize_sample(sample)
            improved_data.append(improved_sample)
            
            # 统计
            dataset = improved_sample['dataset']
            task_type = improved_sample['task_type']
            
            if dataset not in stats:
                stats[dataset] = {'total': 0, 'task_types': {}}
            stats[dataset]['total'] += 1
            stats[dataset]['task_types'][task_type] = stats[dataset]['task_types'].get(task_type, 0) + 1
            
        except Exception as e:
            print(f"⚠️  处理第{i+1}个样本时出错: {e}")
            continue
    
    print(f"💾 保存到: {output_file}")
    save_jsonl(improved_data, output_file)
    
    # 打印统计信息
    print("\n📊 改进后数据统计:")
    print("=" * 60)
    for dataset, info in stats.items():
        print(f"{dataset}: {info['total']} 样本")
        for task_type, count in info['task_types'].items():
            print(f"  - {task_type}: {count} 样本")
    print("-" * 60)
    print(f"总计: {len(improved_data)} 样本")
    
    return improved_data, stats

def create_sample_file(data: List[Dict[str, Any]], output_file: str, sample_size: int = 20):
    """创建样本文件"""
    sample_data = data[:sample_size]
    save_jsonl(sample_data, output_file)
    print(f"📋 样本文件: {output_file} ({len(sample_data)} 样本)")

def main():
    """主函数"""
    
    # 输入和输出文件
    input_file = "raw_datasets/commonsense/cs_all_cleaned.jsonl"
    output_file = "raw_datasets/commonsense/cs_all_improved.jsonl"
    sample_file = "raw_datasets/commonsense/cs_all_improved_sample.jsonl"
    
    print("🚀 开始数据改进...")
    
    # 进一步改进清洗
    improved_data, stats = improve_cleaned_data(input_file, output_file)
    
    # 创建样本文件
    create_sample_file(improved_data, sample_file)
    
    print("\n✅ 数据改进完成!")

if __name__ == "__main__":
    main()
