#!/usr/bin/env python3
"""
Lightning风格的快速评估脚本 - 基于PyTorch Lightning优化的评估方法
支持同时评估多个模型，包括基础模型和LoRA模型
使用Lightning的数据加载和并行处理机制，显著提高评估速度

使用示例:
python eval/lightning_eval.py --models_list \
    /root/autodl-tmp/models/Qwen_Qwen2.5-1.5B \
    /root/autodl-tmp/models/gemma-2-2b-it \
    /root/PAW/runs/Qwen_Qwen2.5-1.5B/arc-challenge_lora_20250723_191421/final_model \
    --dataset arc-challenge
"""

import os
import sys
import json
import argparse
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import random

# 设置环境变量以解决tokenizers并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 禁用CUDA图优化，解决Gemma模型的deterministic index put问题
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"
# 设置transformers日志级别，减少警告信息
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
torch.set_float32_matmul_precision('medium')  # 使用更保守的设置
# 禁用CUDA图以避免deterministic问题
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.enable_flash_sdp(False)  # 禁用Flash Attention
torch.backends.cuda.enable_mem_efficient_sdp(False)  # 禁用内存高效Attention

import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

import pandas as pd
from tqdm import tqdm


def get_test_file_path(dataset_name: str) -> str:
    """获取测试文件路径"""
    # 尝试多个可能的数据目录路径
    possible_paths = [
        f"data_to_lora/cs/{dataset_name}",  # 从PAW根目录运行
        f"../data_to_lora/cs/{dataset_name}",  # 从pipeline目录运行
        f"../../data_to_lora/cs/{dataset_name}",  # 从子目录运行
    ]
    
    for data_dir in possible_paths:
        test_file = f"{data_dir}/{dataset_name}_test_formatted.jsonl"
        validation_file = f"{data_dir}/{dataset_name}_validation_formatted.jsonl"
        
        if os.path.exists(test_file):
            return test_file
        elif os.path.exists(validation_file):
            print(f"📝 使用validation文件作为test: {validation_file}")
            return validation_file
    
    # 如果都找不到，给出详细的错误信息
    raise FileNotFoundError(f"数据集 {dataset_name} 找不到test或validation文件。尝试过的路径: {possible_paths}")


class SimpleDataset(Dataset):
    """简单的数据集类，适用于评估"""
    def __init__(self, data_file: str, sample_ratio: float = 1.0):
        self.data = self._load_data(data_file)
        
        # 如果需要采样加速评估
        if sample_ratio < 1.0:
            original_size = len(self.data)
            sample_size = max(1, int(original_size * sample_ratio))
            # 使用固定种子保证采样可重复
            random.seed(42)
            self.data = random.sample(self.data, sample_size)
            print(f"  📊 采样数据: {sample_size}/{original_size} ({sample_ratio*100:.1f}%)")
        else:
            print(f"  📊 使用完整数据: {len(self.data)}样本")
    
    def _load_data(self, data_file: str) -> List[Dict[str, Any]]:
        """从JSONL文件加载数据"""
        data = []
        if os.path.exists(data_file):
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx].copy()


class LightningModelEvaluator(pl.LightningModule):
    """Lightning模型评估模块"""
    
    def __init__(self, model_path: str, base_model_path: str = None, max_length: int = 512):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_path = model_path
        self.base_model_path = base_model_path
        self.max_length = max_length
        
        # 创建模型名称用于报告
        self.model_name = Path(model_path).name
        
        # 加载模型和tokenizer
        self._load_model()
        
    def _load_model(self):
        """加载模型和tokenizer"""
        print(f"📦 加载模型: {self.model_path}")
        
        # 检查是否是本地路径还是Hugging Face模型ID
        is_local_path = os.path.exists(self.model_path)
        
        print(f"🔍 模型路径检查: {self.model_path}")
        print(f"🔍 是否为本地路径: {is_local_path}")
        
        # 检查模型路径是否存在
        if not is_local_path:
            print(f"❌ 模型路径不存在: {self.model_path}")
            raise FileNotFoundError(f"模型路径不存在: {self.model_path}")
        
        try:
            # 模型加载参数
            load_kwargs = {
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "trust_remote_code": True,
                "use_cache": True,
                "device_map": "auto" if torch.cuda.is_available() else None,
            }
            
            print(f"🔍 模型加载参数: {load_kwargs}")
            
            # 检查是否是LoRA模型
            config_path = Path(self.model_path) / "adapter_config.json"
            print(f"🔍 检查LoRA配置文件: {config_path} (存在: {config_path.exists()})")
            
            if config_path.exists():
            # LoRA模型加载流程
            print("🔧 检测到LoRA模型，使用PEFT加载...")
            try:
                # 加载PEFT配置获取基础模型信息
                peft_config = PeftConfig.from_pretrained(self.model_path)
                detected_base_model = peft_config.base_model_name_or_path
                
                # 使用提供的基础模型路径或检测到的路径
                actual_base_model = self.base_model_path or detected_base_model
                
                # 确认基础模型路径
                if not os.path.exists(actual_base_model) and "/" not in actual_base_model:
                    # 可能是相对路径，尝试autodl-tmp中的常见位置
                    for prefix in ["/root/autodl-tmp/models/", "/root/autodl-tmp/"]:
                        test_path = f"{prefix}{actual_base_model}"
                        if os.path.exists(test_path):
                            actual_base_model = test_path
                            break
                
                print(f"📦 加载基础模型: {actual_base_model}")
                
                # 加载基础模型的tokenizer (移除local_files_only限制)
                tokenizer_kwargs = {"trust_remote_code": True}
                
                self.tokenizer = AutoTokenizer.from_pretrained(actual_base_model, **tokenizer_kwargs)
                
                # 特殊处理Gemma模型
                if "gemma" in actual_base_model.lower():
                    print("🦙 检测到Gemma模型，应用特殊配置...")
                    load_kwargs.update({
                        "attn_implementation": "eager",  # 避免使用flash attention
                        "use_cache": False,  # 禁用缓存机制
                        "_attn_implementation_internal": "eager"
                    })
                
                # 加载基础模型
                base_model = AutoModelForCausalLM.from_pretrained(
                    actual_base_model,
                    **load_kwargs
                )
                
                print(f"🔧 加载LoRA适配器: {self.model_path}")
                # 加载PEFT模型
                self.model = PeftModel.from_pretrained(base_model, self.model_path)
                
            except Exception as e:
                print(f"❌ 作为PEFT模型加载失败: {e}")
                raise RuntimeError(f"无法加载LoRA模型: {self.model_path}，LoRA模型必须与正确的基础模型匹配")
        else:
            # 常规模型加载流程
            print("📦 加载为常规模型...")
            
            # 处理tokenizer (移除严格的local_files_only限制)
            tokenizer_kwargs = {"trust_remote_code": True}
                
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, **tokenizer_kwargs)
            except Exception as e:
                print(f"⚠️ 标准tokenizer加载失败: {e}")
                print("尝试使用备用tokenizer选项...")
                tokenizer_kwargs["use_fast"] = False
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, **tokenizer_kwargs)
            
            # 针对Gemma模型的特殊处理
            model_name_lower = self.model_path.lower()
            special_kwargs = load_kwargs.copy()
            
            if "gemma" in model_name_lower:
                print("🔍 检测到Gemma模型，应用特殊配置...")
                special_kwargs.update({
                    "attn_implementation": "eager",  # 避免使用flash attention
                    "use_cache": False,  # 禁用缓存机制
                    "_attn_implementation_internal": "eager"
                })
                
            # 加载模型，移除严格的local_files_only限制
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **special_kwargs
            )
        
        # 确保模型处于评估模式
        self.model.eval()
        
        # 设置pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print(f"✅ 模型加载成功: {self.model_path}")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {self.model_path}")
        print(f"❌ 错误类型: {type(e).__name__}")
        print(f"❌ 错误信息: {str(e)}")
        print(f"❌ 详细错误:")
        traceback.print_exc()
        raise RuntimeError(f"无法加载模型 {self.model_path}: {str(e)}")

    def test_step(self, batch, batch_idx):
        """单个测试步骤"""
        try:
            # 计算损失
            loss = self._compute_loss(batch)
            # 计算准确率
            accuracy = self._compute_accuracy(batch)
            perplexity = torch.exp(loss)
            
            batch_size = len(batch)
            
            # 记录指标
            self.log('test/loss', loss, batch_size=batch_size)
            self.log('test/accuracy', accuracy, batch_size=batch_size)
            self.log('test/perplexity', perplexity, batch_size=batch_size)
            
            return {
                'loss': loss,
                'accuracy': accuracy,
                'perplexity': perplexity,
                'batch_size': batch_size
            }
        except Exception as e:
            print(f"❌ test_step失败 (batch_idx={batch_idx}): {e}")
            print(f"❌ batch内容: {batch}")
            traceback.print_exc()
            # 返回默认值避免训练中断
            return {
                'loss': torch.tensor(float('inf')),
                'accuracy': torch.tensor(0.0),
                'perplexity': torch.tensor(float('inf')),
                'batch_size': len(batch) if batch else 1
            }
        
    def _compute_loss(self, batch):
        """计算损失"""
        try:
            inputs = []
            labels = []
            
            for item in batch:
                # 处理多选题格式
                if 'input' in item and 'options' in item:
                    question = item['input']
                    options = item['options']
                    target = item.get('target', 'A')
                    
                    # 格式化问题、选项和答案
                    text = f"Question: {question}\n"
                    for option in options:
                        text += f"{option}\n"
                    text += f"Answer: {target}"
                else:
                    # 备选：使用任何文本字段
                    text = item.get('text', str(item))
                
                # Tokenize
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                inputs.append(encoding['input_ids'].squeeze())
                labels.append(encoding['input_ids'].squeeze())

            if inputs:
                input_ids = torch.stack(inputs).to(self.device)
                attention_mask = torch.ones_like(input_ids).to(self.device)
                labels = torch.stack(labels).to(self.device)
            else:
                return torch.tensor(0.0)
            
            # 计算损失
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            return outputs.loss
            
        except Exception as e:
            print(f"❌ _compute_loss失败: {e}")
            print(f"❌ batch大小: {len(batch) if batch else 'None'}")
            if batch:
                print(f"❌ 第一个样本: {batch[0] if len(batch) > 0 else 'Empty'}")
            traceback.print_exc()
            return torch.tensor(float('inf'))

    def _compute_accuracy(self, batch):
        """计算准确率"""
        if not isinstance(batch, list):
            return torch.tensor(0.25)  # 4选1题的随机基线
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for item in batch:
                try:
                    # 解析数据项
                    question = item.get('input', '')
                    options = item.get('options', [])
                    correct_answer = item.get('target', 'A')
                    
                    if not options:
                        total += 1
                        continue
                    
                    # 格式化带选项的问题
                    prompt = f"Question: {question}\n"
                    for option in options:
                        prompt += f"{option}\n"
                    prompt += "Answer:"
                    
                    # Tokenize
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors='pt',
                        truncation=True,
                        max_length=self.max_length,
                        padding=True
                    ).to(self.device)
                    
                    # Gemma模型特殊处理
                    model_name_lower = self.model_path.lower()
                    generation_kwargs = {
                        "max_new_tokens": 3,  # 减少生成长度
                        "do_sample": False,
                        "pad_token_id": self.tokenizer.eos_token_id,
                        "use_cache": False,  # 禁用缓存
                        "output_attentions": False,
                        "output_hidden_states": False,
                    }
                    
                    if "gemma" in model_name_lower:
                        # Gemma模型特殊适配
                        generation_kwargs.update({
                            "temperature": 1.0,
                            "top_p": 1.0,
                            "repetition_penalty": 1.0,
                        })
                    
                    # 生成答案
                    outputs = self.model.generate(
                        **inputs,
                        **generation_kwargs
                    )
                    
                    # 解码生成的答案
                    generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                    generated_answer = generated_text.strip().upper()
                    
                    # 提取第一个字母 (A, B, C, 或 D)
                    predicted_answer = None
                    for char in generated_answer:
                        if char in ['A', 'B', 'C', 'D']:
                            predicted_answer = char
                            break
                    
                    # 如果没有找到明确答案，尝试匹配选项前缀
                    if predicted_answer is None:
                        for option in options:
                            if option.startswith('A:') and 'A' in generated_answer:
                                predicted_answer = 'A'
                            elif option.startswith('B:') and 'B' in generated_answer:
                                predicted_answer = 'B'
                            elif option.startswith('C:') and 'C' in generated_answer:
                                predicted_answer = 'C'
                            elif option.startswith('D:') and 'D' in generated_answer:
                                predicted_answer = 'D'
                            if predicted_answer:
                                break
                    
                    # 与正确答案比较
                    if predicted_answer == correct_answer:
                        correct += 1
                    
                    total += 1
                    
                except Exception as e:
                    print(f"⚠️ 处理样本错误: {e}")
                    total += 1
                    continue
        
        if total == 0:
            return torch.tensor(0.0)
        
        accuracy = correct / total
        return torch.tensor(accuracy)

    def configure_optimizers(self):
        """配置优化器 - 评估模式不需要，但Lightning需要这个方法"""
        return None


def load_model_for_eval(model_path, device="auto", **kwargs):
    """加载模型并准备好评估"""
    print(f"⏳ 加载模型 {model_path}...")
    
    # 确定设备
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 检查是否是本地路径
    is_local_path = os.path.exists(model_path)
    
    try:
        # 环境变量设置，避免tokenizer警告和优化
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        # 加载tokenizer (特别处理本地模型路径)
        tokenizer_kwargs = {"trust_remote_code": True}
        if is_local_path:
            tokenizer_kwargs["local_files_only"] = True
            
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
        except Exception as e:
            print(f"警告: 标准tokenizer加载失败，尝试使用预训练tokenizer: {e}")
            tokenizer_kwargs["use_fast"] = False
            tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
        
        # 模型名称转小写用于判断
        model_name_lower = model_path.lower() if isinstance(model_path, str) else ""
        
        # 根据模型类型设置不同的加载参数
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            "device_map": device,
            "trust_remote_code": True,
        }
        
        # 添加本地文件参数
        if is_local_path:
            model_kwargs["local_files_only"] = True
        
        # Gemma模型需要特殊处理
        if "gemma" in model_name_lower:
            print("🦙 检测到Gemma模型，使用特殊加载设置")
            model_kwargs.update({
                "attn_implementation": "eager",  # 避免使用flash attention
            })
        
        # 检查是否为LoRA微调模型
        adapter_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_path):
            print("🔍 检测到LoRA适配器配置")
            # 加载基础模型
            base_model_path = kwargs.get('base_model_path')
            if not base_model_path:
                # 尝试从adapter_config.json中找到基础模型
                try:
                    with open(adapter_path, 'r') as f:
                        adapter_config = json.load(f)
                    base_model_path = adapter_config.get('base_model_name_or_path')
                    print(f"📄 从adapter_config.json获取到基础模型: {base_model_path}")
                except Exception as e:
                    raise ValueError(f"LoRA模型需要指定base_model_path参数: {e}")
            
            # 检查基础模型路径
            if not os.path.exists(base_model_path) and "/" not in base_model_path:
                # 可能是相对路径，尝试autodl-tmp中的常见位置
                for prefix in ["/root/autodl-tmp/models/", "/root/autodl-tmp/"]:
                    test_path = f"{prefix}{base_model_path}"
                    if os.path.exists(test_path):
                        base_model_path = test_path
                        print(f"🔍 定位到基础模型: {base_model_path}")
                        break
            
            # 加载基础模型
            print(f"🔄 加载基础模型: {base_model_path}")
            
            # 特殊处理本地基础模型
            base_kwargs = model_kwargs.copy()
            if os.path.exists(base_model_path):
                base_kwargs["local_files_only"] = True
                
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                **base_kwargs
            )
            
            # 加载LoRA权重
            print(f"🔄 加载LoRA权重: {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
        else:
            # 直接加载模型
            print(f"🔄 加载标准模型: {model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
        
        # 将模型置于评估模式
        model.eval()
        
        print(f"✅ 模型 {model_path} 加载完成")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        traceback.print_exc()
        raise


def evaluate_models(
    models_list: List[str],
    dataset_name: str,
    output_dir: str = "eval/results",
    base_model_path: str = None,
    sample_ratio: float = 1.0,
    batch_size: int = 8
):
    """评估多个模型并保存结果"""
    print("\n" + "=" * 70)
    print(f"🚀 Lightning 批量模型评估")
    print("=" * 70)
    
    # 准备输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 加载测试数据
    test_file = get_test_file_path(dataset_name)
    test_dataset = SimpleDataset(test_file, sample_ratio=sample_ratio)
    
    print(f"📝 数据集: {dataset_name}")
    print(f"📝 测试文件: {test_file}")
    print(f"📊 样本数量: {len(test_dataset)}")
    print(f"📊 批处理大小: {batch_size}")
    print(f"📊 采样比例: {sample_ratio*100:.1f}%")
    
    results = {}
    start_time = time.time()
    
    # 准备共享数据加载器 - 使用固定的随机种子以确保可比性
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # Lightning测试推荐打乱顺序
        num_workers=2,  # 减少worker数量，降低fork带来的警告
        pin_memory=True,
        collate_fn=lambda batch: batch,  # 保持批次格式不变
        generator=torch.Generator().manual_seed(42),  # 固定随机种子
        persistent_workers=True  # 保持worker持续运行，避免频繁fork
    )
    
    # 评估每个模型
    for i, model_path in enumerate(models_list):
        print(f"\n{'='*70}")
        print(f"📊 [{i+1}/{len(models_list)}] 评估模型: {model_path}")
        
        model_name = Path(model_path).name
        if not model_name:  # 处理路径末尾的斜杠
            model_name = Path(model_path).parent.name
            
        try:
            # 初始化Lightning评估模块
            evaluator = LightningModelEvaluator(model_path, base_model_path)
            
            # 创建Trainer (无需checkpoint) - 针对Gemma模型优化
            trainer_kwargs = {
                "accelerator": 'auto',
                "devices": 'auto',
                "precision": '16-mixed' if torch.cuda.is_available() else 32,
                "logger": False,
                "enable_checkpointing": False,  # 评估不需要检查点
                "enable_model_summary": False,  # 关闭模型摘要
                "enable_progress_bar": True,
                "deterministic": False,  # 对Gemma模型禁用deterministic
                "num_sanity_val_steps": 0,  # 避免sanity检查
                "inference_mode": True,  # 使用推理模式
                "benchmark": False,  # 关闭基准测试
            }
            
            # 如果是Gemma模型，使用更保守的设置
            if "gemma" in model_path.lower():
                trainer_kwargs.update({
                    "precision": 32,  # 使用32位精度避免数值问题
                    "deterministic": False,  # 完全禁用deterministic
                })
            
            trainer = Trainer(**trainer_kwargs)
            
            # 执行测试
            eval_start = time.time()
            test_results = trainer.test(evaluator, dataloaders=test_loader)
            eval_time = time.time() - eval_start
            
            # 整理结果 - 确保所有值都是Python标量而不是Tensor
            model_results = {}
            if test_results and len(test_results) > 0:
                raw_results = test_results[0]
                # 转换所有的tensor值为Python标量
                for key, value in raw_results.items():
                    if hasattr(value, 'item'):
                        model_results[key] = value.item()
                    else:
                        model_results[key] = value
            
            # 添加时间指标
            model_results['eval_time_seconds'] = eval_time
            model_results['samples_per_second'] = len(test_dataset) / eval_time
            
            # 添加到结果集
            results[model_name] = {
                dataset_name: model_results
            }
            
            # 保存单个模型结果
            result_file = output_path / f"{model_name}_{dataset_name}_evaluation_results.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(model_results, f, indent=4, ensure_ascii=False)
                
            print(f"✅ 评估完成 (用时: {eval_time:.1f}秒, {model_results['samples_per_second']:.1f} 样本/秒)")
            print(f"📊 结果:")
            print(f"  - Loss: {model_results.get('test/loss', 0):.4f}")
            print(f"  - Accuracy: {model_results.get('test/accuracy', 0):.4f}") 
            print(f"  - Perplexity: {model_results.get('test/perplexity', 0):.4f}")
            
            # 清理内存
            del evaluator
            del trainer
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ 评估失败: {e}")
            print(f"❌ 模型路径: {model_path}")
            print(f"❌ 模型名称: {model_name}")
            print(f"❌ 数据集: {dataset_name}")
            print(f"❌ 详细错误信息:")
            traceback.print_exc()
            results[model_name] = {
                dataset_name: {"error": str(e)}
            }
    
    # 计算总用时
    total_time = time.time() - start_time
    
    # 保存汇总结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = output_path / f"lightning_evaluation_summary_{timestamp}.json"
    
    summary_data = {
        "evaluation_summary": {
            "dataset": dataset_name,
            "total_models": len(models_list),
            "sample_ratio": sample_ratio,
            "batch_size": batch_size,
            "total_samples": len(test_dataset),
            "total_time_seconds": total_time,
            "timestamp": datetime.now().isoformat()
        },
        "results": results
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)
    
    # 保存CSV格式结果
    rows = []
    for model_name, model_results in results.items():
        for dataset_name, dataset_results in model_results.items():
            if 'error' not in dataset_results:
                # 确保所有值都是Python标量
                loss_val = dataset_results.get('test/loss', 0)
                acc_val = dataset_results.get('test/accuracy', 0)
                ppl_val = dataset_results.get('test/perplexity', 0)
                time_val = dataset_results.get('eval_time_seconds', 0)
                samples_val = dataset_results.get('samples_per_second', 0)
                
                # 转换tensor/numpy值为Python标量
                if hasattr(loss_val, 'item'):
                    loss_val = float(loss_val.item())
                if hasattr(acc_val, 'item'):
                    acc_val = float(acc_val.item())
                if hasattr(ppl_val, 'item'):
                    ppl_val = float(ppl_val.item())
                if hasattr(time_val, 'item'):
                    time_val = float(time_val.item())
                if hasattr(samples_val, 'item'):
                    samples_val = float(samples_val.item())
                
                rows.append({
                    'Model': str(model_name),
                    'Dataset': str(dataset_name),
                    'Loss': round(float(loss_val), 4),
                    'Accuracy': round(float(acc_val), 4),
                    'Perplexity': round(float(ppl_val), 4),
                    'Eval_Time(s)': round(float(time_val), 1),
                    'Samples/Sec': round(float(samples_val), 1),
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
    
    csv_file = None  # 初始化变量
    if rows:
        try:
            # 手动写CSV文件，避免pandas问题
            csv_file = output_path / f"lightning_evaluation_results_{timestamp}.csv"
            
            # 获取列名
            headers = ['Model', 'Dataset', 'Loss', 'Accuracy', 'Perplexity', 'Eval_Time(s)', 'Samples/Sec', 'Timestamp']
            
            with open(csv_file, 'w', encoding='utf-8', newline='') as f:
                f.write(','.join(headers) + '\n')
                for row in rows:
                    values = []
                    for header in headers:
                        val = row.get(header, '')
                        # 确保值是字符串，并处理可能的逗号
                        if isinstance(val, str) and ',' in val:
                            val = f'"{val}"'
                        values.append(str(val))
                    f.write(','.join(values) + '\n')
            
            print(f"� 结果已保存到: {csv_file}")
            
            # 尝试pandas方式作为备选
            try:
                # 只有在手动方式成功后才尝试pandas
                import pandas as pd
                df = pd.DataFrame(rows)
                cumulative_csv = output_path / "all_evaluation_results.csv"
                if cumulative_csv.exists():
                    existing_df = pd.read_csv(cumulative_csv, encoding='utf-8-sig')
                    # 移除重复项
                    for _, row_data in df.iterrows():
                        mask = (existing_df['Model'] == row_data['Model']) & (existing_df['Dataset'] == row_data['Dataset'])
                        existing_df = existing_df[~mask]
                    combined_df = pd.concat([existing_df, df], ignore_index=True)
                    combined_df.to_csv(cumulative_csv, index=False, encoding='utf-8-sig')
                else:
                    df.to_csv(cumulative_csv, index=False, encoding='utf-8-sig')
                print(f"📁 累积结果: {cumulative_csv}")
            except Exception as pandas_error:
                print(f"⚠️ pandas累积CSV更新失败: {pandas_error}")
                # 不影响主要CSV文件的保存
        except Exception as e:
            print(f"⚠️ 保存CSV结果失败: {e}")
            traceback.print_exc()
    
    # 打印汇总表格
    print("\n" + "=" * 80)
    print("📊 评估结果汇总")
    print("=" * 80)
    print(f"{'Model':<40} {'Dataset':<15} {'Loss':<8} {'Accuracy':<10} {'Perplexity':<12} {'Time(s)':<8} {'Samples/s':<10}")
    print("-" * 110)
    
    for model_name, model_results in results.items():
        for dataset_name, dataset_results in model_results.items():
            if 'error' not in dataset_results:
                print(f"{model_name:<40} {dataset_name:<15} "
                      f"{dataset_results.get('test/loss', 0):<8.4f} "
                      f"{dataset_results.get('test/accuracy', 0):<10.4f} "
                      f"{dataset_results.get('test/perplexity', 0):<12.4f} "
                      f"{dataset_results.get('eval_time_seconds', 0):<8.1f} "
                      f"{dataset_results.get('samples_per_second', 0):<10.1f}")
            else:
                print(f"{model_name:<40} {dataset_name:<15} {'ERROR':<8} {'ERROR':<10} {'ERROR':<12} {'ERROR':<8} {'ERROR':<10}")
    
    print("\n" + "=" * 80)
    print(f"⏱️  总评估时间: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    print(f"📁 汇总结果: {summary_file}")
    if csv_file:
        print(f"📊 CSV结果: {csv_file}")
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Lightning风格的快速模型评估工具")
    parser.add_argument("--models_list", type=str, nargs="+", required=True,
                       help="要评估的模型路径列表")
    parser.add_argument("--dataset", type=str, default="arc-challenge",
                       help="数据集名称 (默认: arc-challenge)")
    parser.add_argument("--output_dir", type=str, default="eval/results",
                       help="评估结果输出目录 (默认: eval/results)")
    parser.add_argument("--base_model", type=str, default=None,
                       help="指定基础模型路径，用于加载LoRA模型 (可选)")
    parser.add_argument("--sample_ratio", type=float, default=1.0,
                       help="数据采样比例，加速评估 (默认: 1.0 = 100%)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="批处理大小 (默认: 8)")
    
    args = parser.parse_args()
    
    print("🔬 Lightning模型评估工具")
    print("=" * 50)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 验证模型路径
    valid_models = []
    for model_path in args.models_list:
        if os.path.exists(model_path):
            valid_models.append(model_path)
        else:
            print(f"⚠️ 警告: 模型路径不存在: {model_path}")
            print(f"将尝试作为HuggingFace模型名称加载")
            valid_models.append(model_path)  # 仍然添加，让下游处理
    
    if len(valid_models) == 0:
        print("❌ 错误: 没有有效的模型路径")
        return False
    
    # 检测LoRA模型和基础模型
    lora_models = []
    for model_path in valid_models:
        if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "adapter_config.json")):
            lora_models.append(model_path)
    
    if lora_models and not args.base_model:
        print(f"ℹ️ 检测到{len(lora_models)}个LoRA模型:")
        for lora in lora_models:
            print(f"  - {lora}")
        print(f"💡 如果加载失败，请使用 --base_model 参数指定基础模型")
    
    try:
        # 运行评估
        results = evaluate_models(
            models_list=valid_models,
            dataset_name=args.dataset,
            output_dir=args.output_dir,
            base_model_path=args.base_model,
            sample_ratio=args.sample_ratio,
            batch_size=args.batch_size
        )
        
        print("✅ 评估完成")
        return True
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
