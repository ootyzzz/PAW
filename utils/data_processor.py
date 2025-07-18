"""
数据处理模块 - 处理commonsense数据集
支持LoRA微调所需的数据格式转换和预处理
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class CommonsenseDataset(Dataset):
    """Commonsense数据集类"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        """
        初始化数据集
        
        Args:
            data_path: jsonl数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        self._load_data(data_path)
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
    
    def _load_data(self, data_path: str):
        """加载jsonl数据文件"""
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        # 验证数据格式
                        if self._validate_item(item):
                            self.data.append(item)
                        else:
                            logger.warning(f"Invalid data format: {item}")
        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {e}")
            raise
    
    def _validate_item(self, item: Dict[str, Any]) -> bool:
        """验证数据项格式"""
        # 支持两种数据格式：
        # 1. 标准格式: instruction, input, output
        # 2. Commonsense格式: id, dataset, task_type, input, options, target, target_idx
        
        standard_fields = ['instruction', 'input', 'output']
        commonsense_fields = ['id', 'dataset', 'task_type', 'input', 'options', 'target']
        
        # 检查是否为标准格式
        if all(field in item for field in standard_fields):
            return True
        
        # 检查是否为commonsense格式
        if all(field in item for field in commonsense_fields):
            return True
        
        return False
    
    def _format_prompt(self, item: Dict[str, Any]) -> str:
        """格式化输入提示，支持多种数据格式"""
        
        # 标准格式
        if 'instruction' in item and 'input' in item and 'output' in item:
            instruction = item['instruction']
            input_text = item['input']
            if input_text.strip():
                return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
            else:
                return f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        # Commonsense格式
        elif 'task_type' in item and 'input' in item and 'options' in item and 'target' in item:
            task_type = item['task_type']
            input_text = item['input']
            options = item['options']
            
            # 根据任务类型生成指令
            if task_type == "sentence_completion":
                instruction = "Complete the following sentence by choosing the best option:"
            elif task_type == "pronoun_resolution":
                instruction = "Choose the correct word that the pronoun refers to:"
            elif task_type == "multiple_choice":
                instruction = "Answer the following question by choosing the correct option:"
            elif task_type == "physical_reasoning":
                instruction = "Solve this physical reasoning problem by choosing the best option:"
            elif task_type == "yes_no_question":
                instruction = "Answer this question with True or False:"
            else:
                instruction = f"Complete this {task_type} task by choosing the best option:"
            
            # 格式化选项
            if len(options) == 2 and task_type == "yes_no_question":
                # 布尔问题，简化选项显示
                options_text = "A. True\nB. False"
            else:
                # 其他多选题
                options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
            
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Options:\n{options_text}\n\n### Response:\n"
        
        else:
            # 回退到原始格式
            return f"### Input:\n{item.get('input', '')}\n\n### Response:\n"
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """获取单个数据项"""
        item = self.data[idx]
        
        # 格式化输入
        prompt = self._format_prompt(item)
        
        # 获取目标输出
        if 'output' in item:
            # 标准格式
            target_output = item['output']
        elif 'target' in item and 'options' in item:
            # Commonsense格式 - 转换为选项字母
            target = item['target']
            options = item['options']
            target_idx = item.get('target_idx', -1)
            
            if target_idx >= 0 and target_idx < len(options):
                # 使用选项索引生成字母
                target_output = chr(65 + target_idx)  # A, B, C, D...
            else:
                # 尝试从选项中查找目标
                try:
                    target_idx = options.index(target)
                    target_output = chr(65 + target_idx)
                except ValueError:
                    # 如果找不到，直接使用目标文本
                    target_output = target
        else:
            # 回退
            target_output = ""
        
        full_text = prompt + target_output
        
        # Tokenization
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # 计算labels (only for response part)
        prompt_tokenized = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors='pt'
        )
        
        labels = tokenized['input_ids'].clone()
        # Mask prompt部分，只计算response的loss
        prompt_length = len(prompt_tokenized['input_ids'][0])
        labels[0, :prompt_length] = -100
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

class DataProcessor:
    """数据处理器"""
    
    def __init__(self, model_path: str):
        """
        初始化数据处理器
        
        Args:
            model_path: 模型路径，用于加载tokenizer
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 设置pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def create_dataloader(
        self, 
        data_path: str, 
        batch_size: int = 4, 
        max_length: int = 512,
        shuffle: bool = True
    ) -> DataLoader:
        """
        创建数据加载器
        
        Args:
            data_path: 数据文件路径
            batch_size: 批次大小
            max_length: 最大序列长度
            shuffle: 是否随机打乱
            
        Returns:
            DataLoader实例
        """
        dataset = CommonsenseDataset(
            data_path=data_path,
            tokenizer=self.tokenizer,
            max_length=max_length
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Windows上建议设为0
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    def validate_data(self, data_path: str) -> Dict[str, Any]:
        """
        验证数据文件
        
        Args:
            data_path: 数据文件路径
            
        Returns:
            验证结果字典
        """
        try:
            dataset = CommonsenseDataset(
                data_path=data_path,
                tokenizer=self.tokenizer,
                max_length=512
            )
            
            # 统计信息
            total_samples = len(dataset)
            
            # 抽样检查前几个样本
            sample_data = []
            for i in range(min(3, total_samples)):
                item = dataset[i]
                sample_data.append({
                    'input_ids_shape': item['input_ids'].shape,
                    'attention_mask_shape': item['attention_mask'].shape,
                    'labels_shape': item['labels'].shape,
                })
            
            return {
                'valid': True,
                'total_samples': total_samples,
                'sample_data': sample_data,
                'message': f"Successfully validated {total_samples} samples"
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'message': f"Validation failed: {e}"
            }

if __name__ == "__main__":
    # 测试代码
    import sys
    import os
    
    # 设置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    model_path = os.path.join(project_root, "models", "Qwen-Qwen2.5-0.5B")
    data_path = os.path.join(project_root, "raw_datasets", "commonsense", "cs_all_unbalanced.jsonl")
    
    # 测试数据处理器
    processor = DataProcessor(model_path)
    result = processor.validate_data(data_path)
    
    print("Data validation result:")
    print(json.dumps(result, indent=2))
