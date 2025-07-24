#Centralized evaluation logic for models and datasets.


import os
import sys
import argparse
import torch
torch.set_float32_matmul_precision('high')  # 提升 float32 matmul 性能
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from typing import Dict, Any, List
import json
from pathlib import Path
from datetime import datetime
import time
import pandas as pd
import random

def custom_collate_fn(batch):
    """Custom collate function to preserve list of dictionaries format"""
    return batch

def compute_loss(model, tokenizer, batch, max_length):
    """Compute loss for a given batch."""
    # DataLoader returns a list of items, each item is what __getitem__ returns
    if isinstance(batch, list) and len(batch) > 0:
        inputs, labels = [], []
        for item in batch:
            if isinstance(item, dict):
                # For multiple choice questions, create text from the original format
                if 'input' in item and 'options' in item:
                    question = item['input']
                    options = item['options']
                    target = item.get('target', 'A')
                    
                    # Format the question with options and answer
                    text = f"Question: {question}\n"
                    for option in options:
                        text += f"{option}\n"
                    text += f"Answer: {target}"
                else:
                    # Fallback to any text field
                    text = item.get('text', str(item))
            else:
                # Handle string format
                text = str(item)
            
            encoding = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            inputs.append(encoding['input_ids'].squeeze())
            labels.append(encoding['input_ids'].squeeze())

        if inputs:
            input_ids = torch.stack(inputs).to(model.device)
            attention_mask = torch.ones_like(input_ids).to(model.device)
            labels = torch.stack(labels).to(model.device)
        else:
            # Empty batch fallback
            return torch.tensor(0.0)
        
    else:
        # Handle pre-tokenized format (shouldn't happen with our DataLoader)
        print("⚠️  Warning: Unexpected batch format in compute_loss")
        return torch.tensor(0.0)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return outputs.loss

def compute_accuracy(model, tokenizer, batch, max_length):
    """Compute accuracy for a given batch of multiple choice questions."""
    # DataLoader returns a list of items, each item is what __getitem__ returns
    if not isinstance(batch, list):
        print("⚠️  Warning: Cannot compute proper accuracy for non-list batch")
        return torch.tensor(0.25)  # Random baseline for 4-choice questions
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for item in batch:
            try:
                # Parse the data item
                if isinstance(item, dict):
                    question = item.get('input', '')
                    options = item.get('options', [])
                    correct_answer = item.get('target', 'A')
                else:
                    # If item is a string, try to parse it as the full text
                    # This is a fallback for text-only format
                    print(f"⚠️  Warning: Unexpected data format: {type(item)}")
                    total += 1
                    continue
                
                if not options:
                    total += 1
                    continue
                
                # Format the question with options
                prompt = f"Question: {question}\n"
                for option in options:
                    prompt += f"{option}\n"
                prompt += "Answer:"
                
                # Tokenize the prompt
                inputs = tokenizer(
                    prompt,
                    return_tensors='pt',
                    truncation=True,
                    max_length=max_length,
                    padding=True
                ).to(model.device)
                
                # Generate answer
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=5,  # Only need 1-2 tokens for A/B/C/D
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # Decode the generated answer
                generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                generated_answer = generated_text.strip().upper()
                
                # Extract the first letter (A, B, C, or D)
                predicted_answer = None
                for char in generated_answer:
                    if char in ['A', 'B', 'C', 'D']:
                        predicted_answer = char
                        break
                
                # If no clear answer found, try to match with option prefixes
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
                
                # Compare with correct answer
                if predicted_answer == correct_answer:
                    correct += 1
                
                total += 1
                
            except Exception as e:
                print(f"⚠️  Error processing item: {e}")
                total += 1
                continue
    
    if total == 0:
        return torch.tensor(0.0)
    
    accuracy = correct / total
    return torch.tensor(accuracy)

def save_evaluation_results(results: Dict[str, Any], output_dir: str, model_name: str):
    """Save evaluation results to a file."""
    output_path = Path(output_dir) / f"{model_name}_evaluation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"📁 Results saved to {output_path}")

def evaluate_model(model_path: str, test_loader, max_length: int, output_dir: str, base_model_path: str = None):
    """Evaluate a model on a test dataset and save results."""
    print(f"📦 Loading model from {model_path}")
    
    # Check if this is a LoRA model or base model
    config_path = Path(model_path) / "adapter_config.json"
    peft_config_path = Path(model_path) / "adapter_config.json"
    
    if config_path.exists() or peft_config_path.exists():
        # This is a LoRA/PEFT model
        print("🔧 Detected LoRA/PEFT model, loading with PEFT...")
        try:
            # Load PEFT config to get base model info
            peft_config = PeftConfig.from_pretrained(model_path)
            detected_base_model = peft_config.base_model_name_or_path
            
            # Use provided base model path if available, otherwise use detected one
            actual_base_model = base_model_path or detected_base_model
            
            print(f"📦 Loading base model: {actual_base_model}")
            # Load base model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(actual_base_model, trust_remote_code=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                actual_base_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            print(f"🔧 Loading LoRA adapters from: {model_path}")
            # Load the PEFT model
            model = PeftModel.from_pretrained(base_model, model_path)
            
        except Exception as e:
            print(f"❌ Failed to load as PEFT model: {e}")
            # If we have a fallback base model, try that
            if not base_model_path:
                # Try common base model paths
                fallback_paths = [
                    "models/Qwen_Qwen2.5-1.5B",
                    "models/Qwen-Qwen2.5-0.5B", 
                    "Qwen/Qwen2.5-1.5B",
                    "Qwen/Qwen2.5-0.5B"
                ]
                
                for fallback_path in fallback_paths:
                    if os.path.exists(fallback_path):
                        print(f"🔄 Trying fallback base model: {fallback_path}")
                        try:
                            tokenizer = AutoTokenizer.from_pretrained(fallback_path, trust_remote_code=True)
                            base_model = AutoModelForCausalLM.from_pretrained(
                                fallback_path,
                                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                device_map="auto" if torch.cuda.is_available() else None,
                                trust_remote_code=True
                            )
                            model = PeftModel.from_pretrained(base_model, model_path)
                            print(f"✅ Successfully loaded with fallback: {fallback_path}")
                            break
                        except Exception as fe:
                            print(f"❌ Fallback failed: {fe}")
                            continue
                else:
                    print("🔄 All fallbacks failed, trying to load as regular model...")
                    # Final fallback to regular loading
                    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto" if torch.cuda.is_available() else None,
                        trust_remote_code=True
                    )
            else:
                raise e  # Re-raise if we had a specific base model path but it failed
    else:
        # Regular model loading
        print("📦 Loading as regular model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
    
    model.eval()
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_loss, total_accuracy, total_samples = 0.0, 0.0, 0
    total_batches = len(test_loader)
    
    print(f"🔄 开始评估 {total_batches} 个批次...")
    start_time = time.time()

    for batch_idx, batch in enumerate(test_loader):
        # Show progress every 10 batches or on last batch
        if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
            progress = (batch_idx + 1) / total_batches * 100
            print(f"  📊 进度: {batch_idx + 1}/{total_batches} ({progress:.1f}%)")
        
        loss = compute_loss(model, tokenizer, batch, max_length)
        accuracy = compute_accuracy(model, tokenizer, batch, max_length)

        batch_size = len(batch)
        total_loss += loss.item() * batch_size
        total_accuracy += accuracy.item() * batch_size
        total_samples += batch_size

    end_time = time.time()
    eval_time = end_time - start_time

    avg_loss = total_loss / total_samples
    avg_accuracy = total_accuracy / total_samples
    perplexity = torch.exp(torch.tensor(avg_loss))

    results = {
        'loss': avg_loss,
        'accuracy': avg_accuracy,
        'perplexity': perplexity.item(),
        'eval_time_seconds': eval_time,
        'total_samples': total_samples,
        'total_batches': total_batches
    }

    print(f"✅ 评估完成! 用时: {eval_time:.1f}秒")
    print(f"📊 Evaluation Results:")
    print(f"  - Loss: {avg_loss:.4f}")
    print(f"  - Accuracy: {avg_accuracy:.4f}")
    print(f"  - Perplexity: {perplexity:.4f}")
    print(f"  - 样本数: {total_samples}, 批次数: {total_batches}")
    print(f"  - 评估时间: {eval_time:.1f}秒 ({total_samples/eval_time:.1f} samples/sec)")

    # Save results to a file
    model_name = Path(model_path).name  # Use .name instead of .stem to preserve full name
    save_evaluation_results(results, output_dir, model_name)

    return results


class SimpleDataset:
    """Simple dataset class for evaluation"""
    def __init__(self, data_file: str, sample_ratio: float = 1.0):
        self.data = self._load_data(data_file)
        
        # Sample data if sample_ratio < 1.0
        if sample_ratio < 1.0:
            original_size = len(self.data)
            sample_size = max(1, int(original_size * sample_ratio))
            # Use fixed seed for reproducible sampling
            random.seed(42)
            self.data = random.sample(self.data, sample_size)
            print(f"  📊 采样数据: {sample_size}/{original_size} ({sample_ratio*100:.1f}%)")
        
    def _load_data(self, data_file: str) -> List[Dict[str, Any]]:
        """Load data from JSONL file"""
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
        # Return the original dictionary data for proper accuracy calculation
        return self.data[idx]


def get_test_file_path(dataset_name: str) -> str:
    """Get the test file path for a dataset"""
    data_dir = f"data_to_lora/cs/{dataset_name}"
    test_file = f"{data_dir}/{dataset_name}_test_formatted.jsonl"
    validation_file = f"{data_dir}/{dataset_name}_validation_formatted.jsonl"
    
    if os.path.exists(test_file):
        return test_file
    elif os.path.exists(validation_file):
        print(f"📝 Using validation file as test: {validation_file}")
        return validation_file
    else:
        raise FileNotFoundError(f"Neither test nor validation file found for {dataset_name}")


def load_existing_results(output_dir: str) -> Dict[str, Any]:
    """Load existing evaluation results from previous runs"""
    results = {}
    output_path = Path(output_dir)
    
    if not output_path.exists():
        return results
    
    # Load individual result files
    for json_file in output_path.glob("*_evaluation_results.json"):
        try:
            model_name = json_file.stem.replace("_evaluation_results", "")
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if model_name not in results:
                    results[model_name] = {}
                # Try to extract dataset name from filename or path
                # This is a simple heuristic - you might need to adjust based on your naming
                results[model_name]["unknown"] = data
        except Exception as e:
            print(f"⚠️  Failed to load {json_file}: {e}")
    
    # Load batch summary files
    for summary_file in output_path.glob("batch_evaluation_summary_*.json"):
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'results' in data:
                    results.update(data['results'])
        except Exception as e:
            print(f"⚠️  Failed to load {summary_file}: {e}")
    
    return results


def save_results_table(results: Dict[str, Any], output_dir: str):
    """Save results to cumulative CSV table only"""
    if not results:
        print("⚠️  No results to save as table")
        return
    
    # Prepare data for table
    rows = []
    for model_name, model_results in results.items():
        for dataset_name, dataset_results in model_results.items():
            if isinstance(dataset_results, dict) and 'error' not in dataset_results:
                row = {
                    'Model': model_name,
                    'Dataset': dataset_name,
                    'Loss': round(dataset_results.get('loss', 0), 4),
                    'Accuracy': round(dataset_results.get('accuracy', 0), 4),
                    'Perplexity': round(dataset_results.get('perplexity', 0), 4),
                    'Eval_Time(s)': round(dataset_results.get('eval_time_seconds', 0), 1),
                    'Samples': dataset_results.get('total_samples', 0),
                    'Batches': dataset_results.get('total_batches', 0),
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            else:
                row = {
                    'Model': model_name,
                    'Dataset': dataset_name,
                    'Loss': 'ERROR',
                    'Accuracy': 'ERROR',
                    'Perplexity': 'ERROR',
                    'Eval_Time(s)': 0,
                    'Samples': 0,
                    'Batches': 0,
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            rows.append(row)
    
    if not rows:
        print("⚠️  No valid results to save as table")
        return
    
    df = pd.DataFrame(rows)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Only save to cumulative CSV file
    cumulative_csv = output_path / "all_evaluation_results.csv"
    if cumulative_csv.exists():
        # Read existing data
        existing_df = pd.read_csv(cumulative_csv, encoding='utf-8-sig')
        
        # Remove duplicate entries (same model + dataset)
        for _, row in df.iterrows():
            mask = (existing_df['Model'] == row['Model']) & (existing_df['Dataset'] == row['Dataset'])
            existing_df = existing_df[~mask]
        
        # Append new results
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df.to_csv(cumulative_csv, index=False, encoding='utf-8-sig')
        print(f"📊 累积结果已更新: {cumulative_csv} (新增 {len(df)} 行)")
    else:
        # Create new cumulative file
        df.to_csv(cumulative_csv, index=False, encoding='utf-8-sig')
        print(f"📊 创建累积结果文件: {cumulative_csv} ({len(df)} 行)")
    
    return df


def estimate_completion_time(completed_tasks: int, total_tasks: int, elapsed_time: float) -> str:
    """Estimate remaining time"""
    if completed_tasks == 0:
        return "未知"
    
    avg_time_per_task = elapsed_time / completed_tasks
    remaining_tasks = total_tasks - completed_tasks
    remaining_time = remaining_tasks * avg_time_per_task
    
    if remaining_time < 60:
        return f"{remaining_time:.0f}秒"
    elif remaining_time < 3600:
        return f"{remaining_time/60:.1f}分钟"
    else:
        return f"{remaining_time/3600:.1f}小时"


def batch_evaluate_models(models_dir: str, datasets: List[str], output_dir: str = "eval/results", base_model_path: str = None, sample_ratio: float = 0.15, strategy: str = "dataset_first", specific_models: List[str] = None, batch_size: int = 8):
    """Batch evaluate multiple models on multiple datasets with resume capability"""
    print("🚀 开始批量评估模型...")
    print(f"📁 模型目录: {models_dir}")
    print(f"📊 数据集: {', '.join(datasets)}")
    print(f"📁 输出目录: {output_dir}")
    print(f"📊 采样比例: {sample_ratio*100:.1f}% (加速评估)")
    print(f"🔧 批处理大小: {batch_size}")
    print(f"🎯 评估策略: {'优先完成数据集' if strategy == 'dataset_first' else '优先完成模型'}")
    if base_model_path:
        print(f"📦 指定基础模型: {base_model_path}")
    
    # Find all model directories
    models_path = Path(models_dir)
    if not models_path.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    if specific_models:
        # Use specific models list
        model_dirs = []
        for model_name in specific_models:
            model_path = models_path / model_name
            if model_path.exists() and model_path.is_dir():
                model_dirs.append(model_path)
            else:
                print(f"⚠️  指定的模型不存在: {model_name}")
        if not model_dirs:
            raise FileNotFoundError(f"None of the specified models found in: {models_dir}")
        print(f"📦 使用指定的 {len(model_dirs)} 个模型: {', '.join([d.name for d in model_dirs])}")
    else:
        # Use all models in directory
        model_dirs = [d for d in models_path.iterdir() if d.is_dir()]
        if not model_dirs:
            raise FileNotFoundError(f"No model directories found in: {models_dir}")
        print(f"📦 发现 {len(model_dirs)} 个模型目录")
    
    total_tasks = len(model_dirs) * len(datasets)
    print(f"🎯 总计评估任务: {total_tasks} 个 ({len(model_dirs)} 模型 × {len(datasets)} 数据集)")
    
    # Load existing results
    existing_results = load_existing_results(output_dir)
    print(f"📂 加载已有结果: {len(existing_results)} 个模型的历史结果")
    
    all_results = existing_results.copy()
    completed_tasks = sum(len(model_results) for model_results in existing_results.values())
    start_time = time.time()
    
    print(f"⚡ 已完成任务: {completed_tasks}/{total_tasks}")
    
    # Choose evaluation order based on strategy
    if strategy == "dataset_first":
        # Complete one dataset at a time across all models
        order_desc = "按数据集优先顺序 (更快看到完整数据集结果)"
        evaluation_pairs = [(model_dir, dataset) for dataset in datasets for model_dir in model_dirs]
    else:  # model_first
        # Complete one model at a time across all datasets  
        order_desc = "按模型优先顺序 (更快看到完整模型结果)"
        evaluation_pairs = [(model_dir, dataset) for model_dir in model_dirs for dataset in datasets]
    
    print(f"📋 评估顺序: {order_desc}")
    
    try:
        for pair_idx, (model_dir, dataset) in enumerate(evaluation_pairs):
            model_name = model_dir.name
            
            # Check if this combination already exists
            if model_name in all_results and dataset in all_results[model_name]:
                print(f"⏭️  跳过已完成: {model_name} × {dataset}")
                continue
            
            print(f"\n{'='*70}")
            print(f"🔄 [{pair_idx+1}/{len(evaluation_pairs)}] 评估: {model_name} × {dataset}")
            elapsed_time = time.time() - start_time
            remaining_time = estimate_completion_time(completed_tasks, total_tasks, elapsed_time)
            print(f"⏱️  预计剩余时间: {remaining_time}")
            print(f"{'='*70}")
            
            # For base models, use the directory directly
            model_path = str(model_dir)
            
            if model_name not in all_results:
                all_results[model_name] = {}
            
            task_start_time = time.time()
            
            try:
                # Get test file
                test_file = get_test_file_path(dataset)
                
                # Create dataset with sampling
                test_dataset = SimpleDataset(test_file, sample_ratio=sample_ratio)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
                
                print(f"  📝 数据集大小: {len(test_dataset)} 样本 (采样后)")
                
                # Evaluate
                results = evaluate_model(model_path, test_loader, max_length=512, output_dir=output_dir, base_model_path=base_model_path)
                all_results[model_name][dataset] = results
                
                task_time = time.time() - task_start_time
                completed_tasks += 1
                total_progress = completed_tasks / total_tasks * 100
                
                print(f"✅ 评估完成 (用时: {task_time:.1f}秒)")
                print(f"🎯 总体进度: {completed_tasks}/{total_tasks} ({total_progress:.1f}%)")
                
                # Save intermediate results and table every few tasks
                if completed_tasks % 5 == 0 or completed_tasks == total_tasks:
                    print("💾 保存中间结果...")
                    save_results_table(all_results, output_dir)
                
            except KeyboardInterrupt:
                print(f"\n⚠️  用户中断评估")
                break
            except Exception as e:
                print(f"❌ 评估失败: {e}")
                all_results[model_name][dataset] = {"error": str(e)}
                completed_tasks += 1
    
    except KeyboardInterrupt:
        print(f"\n⚠️  评估被用户中断")
    
    total_time = time.time() - start_time
    
    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = Path(output_dir) / f"batch_evaluation_summary_{timestamp}.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Add timing info to results
    summary_data = {
        "evaluation_summary": {
            "total_models": len(model_dirs),
            "total_datasets": len(datasets),
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "sample_ratio": sample_ratio,
            "strategy": strategy,
            "specific_models": specific_models,
            "total_time_seconds": total_time,
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.now().isoformat()
        },
        "results": all_results
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)
    
    print(f"\n🎉 批量评估完成!")
    print(f"⏱️  总用时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    if completed_tasks > 0:
        print(f"📈 平均每个任务: {total_time/completed_tasks:.1f}秒")
    print(f"📁 综合结果保存至: {summary_file}")
    
    # Save final results table
    df = save_results_table(all_results, output_dir)
    
    # Print summary table
    print_evaluation_table(all_results)
    
    return all_results


def print_evaluation_table(results: Dict[str, Any]):
    """Print evaluation results as a formatted table"""
    print("\n" + "="*80)
    print("📊 评估结果汇总表")
    print("="*80)
    
    # Get all datasets
    all_datasets = set()
    for model_results in results.values():
        all_datasets.update(model_results.keys())
    all_datasets = sorted(list(all_datasets))
    
    # Print header
    print(f"{'Model':<30} {'Dataset':<15} {'Loss':<8} {'Accuracy':<10} {'Perplexity':<12}")
    print("-" * 80)
    
    # Print results
    for model_name, model_results in results.items():
        for i, dataset in enumerate(all_datasets):
            if dataset in model_results and 'error' not in model_results[dataset]:
                res = model_results[dataset]
                model_display = model_name if i == 0 else ""
                print(f"{model_display:<30} {dataset:<15} {res['loss']:<8.4f} {res['accuracy']:<10.4f} {res['perplexity']:<12.4f}")
            elif dataset in model_results:
                model_display = model_name if i == 0 else ""
                print(f"{model_display:<30} {dataset:<15} {'ERROR':<8} {'ERROR':<10} {'ERROR':<12}")
        if model_results:  # Add empty line between models
            print()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="模型评估工具")
    parser.add_argument("--models_dir", type=str, default="runs",
                       help="模型目录路径 (默认: runs)")
    parser.add_argument("--datasets", type=str, nargs="+", 
                       default=["arc-challenge", "arc-easy", "boolq", "hellaswag", "openbookqa", "piqa", "winogrande"],
                       help="要评估的数据集列表")
    parser.add_argument("--output_dir", type=str, default="eval/results",
                       help="评估结果输出目录 (默认: eval/results)")
    parser.add_argument("--single_model", type=str, default=None,
                       help="评估单个模型路径 (可选)")
    parser.add_argument("--single_dataset", type=str, default=None,
                       help="评估单个数据集 (可选)")
    parser.add_argument("--base_model", type=str, default=None,
                       help="指定基础模型路径，用于加载LoRA模型 (可选)")
    parser.add_argument("--sample_ratio", type=float, default=0.05,
                       help="数据采样比例，加速评估 (默认: 0.05 = 5%)")
    parser.add_argument("--strategy", type=str, choices=["dataset_first", "model_first"], default="dataset_first",
                       help="评估策略: dataset_first(优先完成数据集) 或 model_first(优先完成模型)")
    parser.add_argument("--specific_models", type=str, nargs="+", default=None,
                       help="指定要评估的模型名称列表 (可选)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="批处理大小 (默认: 8)")
    
    args = parser.parse_args()
    
    print("🔬 模型评估工具")
    print("=" * 50)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        if args.single_model and args.single_dataset:
            # Single model, single dataset evaluation
            print(f"🎯 单模型单数据集评估")
            print(f"📦 模型: {args.single_model}")
            print(f"📊 数据集: {args.single_dataset}")
            print(f"📊 采样比例: {args.sample_ratio*100:.1f}%")
            print(f"🔧 批处理大小: {args.batch_size}")
            if args.base_model:
                print(f"📦 基础模型: {args.base_model}")
            
            test_file = get_test_file_path(args.single_dataset)
            test_dataset = SimpleDataset(test_file, sample_ratio=args.sample_ratio)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
            
            results = evaluate_model(args.single_model, test_loader, max_length=512, output_dir=args.output_dir, base_model_path=args.base_model)
            
            # Save single result as table
            single_results = {Path(args.single_model).name: {args.single_dataset: results}}
            save_results_table(single_results, args.output_dir)
            
            print("✅ 评估完成")
            
        else:
            # Batch evaluation
            results = batch_evaluate_models(args.models_dir, args.datasets, args.output_dir, args.base_model, args.sample_ratio, args.strategy, args.specific_models, args.batch_size)
            print("✅ 批量评估完成")
            
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
