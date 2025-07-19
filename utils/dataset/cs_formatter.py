#!/usr/bin/env python3
"""
最终数据清理工具
从原始的cs_mixed.jsonl重新处理，确保数据格式正确
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List
import traceback

def load_and_clean_sample(line: str, line_num: int) -> Dict[str, Any]:
    """
    加载并清理单个样本
    """
    try:
        # 尝试JSON解析
        try:
            sample = json.loads(line)
        except json.JSONDecodeError:
            # 尝试eval处理单引号格式
            sample = eval(line.strip())
        
        if not isinstance(sample, dict):
            return None
        
        # 获取数据集名称
        dataset = sample.get('dataset_source', sample.get('dataset', ''))
        if not dataset:
            return None
        
        # 获取原始数据
        original_data = sample.get('original_data', sample)
        
        # 生成统一ID
        if dataset == 'hellaswag':
            original_id = original_data.get('ind', f'h_{line_num}')
            unified_id = f"hellaswag_{original_id}"
        elif dataset == 'winogrande':
            original_id = hash(str(original_data.get('sentence', ''))) % 100000
            unified_id = f"winogrande_{original_id}"
        elif dataset == 'piqa':
            original_id = hash(str(original_data.get('goal', ''))) % 100000
            unified_id = f"piqa_{original_id}"
        elif dataset == 'boolq':
            original_id = hash(str(original_data.get('question', ''))) % 100000
            unified_id = f"boolq_{original_id}"
        elif dataset in ['arc-challenge', 'arc-easy']:
            original_id = original_data.get('id', f'arc_{line_num}')
            unified_id = f"{dataset}_{original_id}".replace('/', '_')
        elif dataset == 'openbookqa':
            original_id = original_data.get('id', f'obqa_{line_num}')
            unified_id = f"openbookqa_{original_id}"
        else:
            unified_id = f"{dataset}_{line_num}"
        
        # 根据数据集类型提取信息
        if dataset in ['arc-challenge', 'arc-easy']:
            input_text = original_data.get('question', '').strip()
            choices = original_data.get('choices', {})
            options = choices.get('text', []) if isinstance(choices, dict) else []
            answer_key = original_data.get('answerKey', '')
            target_idx = ord(answer_key) - ord('A') if answer_key and len(answer_key) == 1 else -1
            target = options[target_idx] if 0 <= target_idx < len(options) else ""
            task_type = "multiple_choice"
            
        elif dataset == 'boolq':
            question = original_data.get('question', '').strip()
            passage = original_data.get('passage', '').strip()
            # 组合问题和段落
            input_text = f"Question: {question}\nPassage: {passage[:500]}..." if len(passage) > 500 else f"Question: {question}\nPassage: {passage}"
            options = ["False", "True"]
            answer = original_data.get('answer', False)
            target_idx = 1 if answer else 0
            target = "True" if answer else "False"
            task_type = "yes_no_question"
            
        elif dataset == 'hellaswag':
            ctx = original_data.get('ctx', '').strip()
            ctx_a = original_data.get('ctx_a', '')
            ctx_b = original_data.get('ctx_b', '')
            # 组合上下文
            if ctx_a and ctx_b:
                input_text = f"{ctx_a} {ctx_b}".strip()
            else:
                input_text = ctx
            options = original_data.get('endings', [])
            label = original_data.get('label', '')
            target_idx = int(label) if label.isdigit() else -1
            target = options[target_idx] if 0 <= target_idx < len(options) else ""
            task_type = "sentence_completion"
            
        elif dataset == 'openbookqa':
            input_text = original_data.get('question_stem', '').strip()
            choices = original_data.get('choices', {})
            options = choices.get('text', []) if isinstance(choices, dict) else []
            answer_key = original_data.get('answerKey', '')
            target_idx = ord(answer_key) - ord('A') if answer_key and len(answer_key) == 1 else -1
            target = options[target_idx] if 0 <= target_idx < len(options) else ""
            task_type = "multiple_choice"
            
        elif dataset == 'piqa':
            input_text = original_data.get('goal', '').strip()
            sol1 = original_data.get('sol1', '').strip()
            sol2 = original_data.get('sol2', '').strip()
            options = [sol1, sol2]
            # PIQA数据集通常没有标准答案
            target_idx = -1
            target = ""
            task_type = "physical_reasoning"
            
        elif dataset == 'winogrande':
            sentence = original_data.get('sentence', '').strip()
            option1 = original_data.get('option1', '').strip()
            option2 = original_data.get('option2', '').strip()
            input_text = sentence
            options = [option1, option2]
            answer = original_data.get('answer', '')
            target_idx = 0 if answer == "1" else 1 if answer == "2" else -1
            target = options[target_idx] if 0 <= target_idx < len(options) else ""
            task_type = "pronoun_resolution"
            
        else:
            return None
        
        # 验证必要字段
        if not input_text:
            return None
        
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
        
    except Exception as e:
        print(f"第{line_num}行处理失败: {e}")
        return None

def main():
    """
    主函数
    """
    try:
        # 文件路径
        input_file = Path("raw_datasets/commonsense/cs_mixed.jsonl")
        output_file = Path("raw_datasets/commonsense/cs_mixed_formatted.jsonl")
        
        # 确保目录存在
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"从 {input_file} 读取数据并进行最终清理...")
        
        if not input_file.exists():
            print(f"输入文件不存在: {input_file}")
            return
        
        # 处理数据
        processed_count = 0
        skipped_count = 0
        dataset_counts = {}
        
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            for line_num, line in enumerate(f_in, 1):
                if not line.strip():
                    continue
                
                # 清理样本
                cleaned_sample = load_and_clean_sample(line, line_num)
                
                if cleaned_sample is None:
                    skipped_count += 1
                    if skipped_count <= 5:  # 只显示前5个错误
                        print(f"跳过第{line_num}行")
                    continue
                
                # 统计数据集
                dataset = cleaned_sample.get('dataset', 'unknown')
                dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
                
                # 写入输出文件
                f_out.write(json.dumps(cleaned_sample, ensure_ascii=False) + '\n')
                
                processed_count += 1
                
                if processed_count % 10000 == 0:
                    print(f"已处理 {processed_count} 个样本...")
        
        print(f"\n处理完成！")
        print(f"- 输入文件: {input_file}")
        print(f"- 输出文件: {output_file}")
        print(f"- 成功处理: {processed_count} 个样本")
        print(f"- 跳过: {skipped_count} 个样本")
        
        print("\n数据集统计:")
        total = 0
        for dataset, count in sorted(dataset_counts.items()):
            print(f"  {dataset}: {count}")
            total += count
        print(f"  总计: {total}")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
