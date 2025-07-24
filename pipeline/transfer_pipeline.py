#!/usr/bin/env python3
"""
transfer_pipeline.py
自动化LoRA训练和迁移管道

主要功能:
1. 训练 source model + LoRA
2. 训练 target model + LoRA  
3. 迁移 source LoRA → target model
4. 评估所有5个模型
5. 更新结果表格

使用方法:
python transfer_pipeline.py \
  --source_model Llama-3.2-3B-Instruct \
  --target_model Qwen_Qwen2.5-1.5B \
  --dataset arc-challenge
"""

import os
import sys
import yaml
import argparse
import subprocess
import pandas as pd
import warnings
from datetime import datetime
from pathlib import Path
import json
import shutil
from tqdm import tqdm
import time

# 修复MKL线程层冲突 - 必须在导入numpy/pandas之前设置
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'

# 屏蔽 Transformers 警告，但保留重要信息
warnings.filterwarnings("ignore", message=".*cache_implementation.*")
warnings.filterwarnings("ignore", message=".*generation flags are not valid.*")
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

class TransferPipeline:
    def __init__(self, config_path="config/pipeline_config.yaml"):
        """初始化管道"""
        self.config = self._load_config(config_path)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = None
        
    def _load_config(self, config_path):
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _get_model_short_name(self, model_path):
        """获取模型简称"""
        model_name = os.path.basename(model_path)
        # 移除常见前缀
        model_name = model_name.replace("models/", "")
        return model_name
    
    def _create_experiment_id(self, source_model, target_model, dataset):
        """创建实验ID"""
        source_short = self._get_model_short_name(source_model)
        target_short = self._get_model_short_name(target_model)
        return f"{source_short}_to_{target_short}_{dataset}_{self.timestamp}"
    
    def _check_existing_experiment(self, source_model, target_model, dataset):
        """检查是否已存在相同实验"""
        csv_path = os.path.join(self.config['paths']['results_dir'], 
                               self.config['results']['csv_file'])
        
        if not os.path.exists(csv_path):
            return None
            
        try:
            df = pd.read_csv(csv_path)
            # 检查是否是空文件或只有表头
            if df.empty or len(df) == 0:
                return None
                
            # 确保所有列都是字符串类型进行比较
            df['source_model'] = df['source_model'].astype(str)
            df['target_model'] = df['target_model'].astype(str)
            df['dataset'] = df['dataset'].astype(str)
            
            existing = df[
                (df['source_model'] == str(source_model)) & 
                (df['target_model'] == str(target_model)) & 
                (df['dataset'] == str(dataset))
            ]
            if not existing.empty:
                return existing.iloc[-1]  # 返回最新的记录
        except (pd.errors.EmptyDataError, KeyError) as e:
            print(f"💡 将重新创建结果文件")
            # 重新创建CSV文件
            header = "experiment_id,source_model,target_model,dataset,source_acc,source_lora_acc,target_acc,target_lora_acc,transferred_acc,source_lora_path,target_lora_path,transferred_lora_path,timestamp,notes,training_config"
            with open(csv_path, 'w') as f:
                f.write(header + '\n')
        except Exception as e:
            print(f"⚠️ 读取历史记录时出错: {e}")
            print(f"💡 将重新创建结果文件")
        
        return None
    
    def _run_command(self, cmd, description, cwd=None):
        """运行命令并返回结果"""
        print(f"\n🚀 {description}")
        print(f"📝 命令: {cmd}")
        
        try:
            import re
            
            # 设置环境变量，保留训练相关输出
            env = os.environ.copy()
            env.update({
                'PYTHONUNBUFFERED': '1',  # 确保Python输出不被缓冲
                'PYTHONIOENCODING': 'utf-8'
            })
            
            # 启动进程，使用实时输出
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # 合并stderr到stdout
                text=True,
                cwd=cwd,
                bufsize=0,  # 无缓冲
                universal_newlines=True,
                env=env
            )
            
            stdout_lines = []
            current_progress_line = None
            
            # 实时处理输出
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    line = output.strip()
                    stdout_lines.append(line)
                    
                    # 显示重要的SwanLab信息（包括链接和基本状态）
                    if any(keyword in line for keyword in [
                        'swanlab.cn', 'View run at', 'View project at',
                        'Tracking run with swanlab', 'Syncing run'
                    ]):
                        print(f"🔗 {line}")
                        continue
                    
                    # 显示训练开始信息
                    if any(keyword in line for keyword in [
                        '🚀 Lightning LoRA 训练', '开始Lightning训练',
                        'Lightning训练完成', '实验目录:', '最终模型:'
                    ]):
                        print(f"   {line}")
                        continue
                    
                    # 处理进度条 - 只显示最新的进度
                    if re.search(r'\d+%\|[█▉▊▋▌▍▎▏ ]*\|', line) or ('it/s' in line and ('Epoch' in line or 'step' in line)):
                        if current_progress_line:
                            # 清除之前的进度行
                            print('\r' + ' ' * len(current_progress_line) + '\r', end='', flush=True)
                        print(f"\r📊 {line}", end='', flush=True)
                        current_progress_line = line
                    else:
                        # 如果有之前的进度行，先换行
                        if current_progress_line:
                            print()  # 换行
                            current_progress_line = None
                        
                        # 显示重要的状态信息，但过滤噪音
                        if not any(noise in line for noise in [
                            'LOCAL_RANK', 'CUDA_VISIBLE_DEVICES', 'Sanity Checking DataLoader',
                            'generation flags are not valid', 'cache_implementation',
                            'Using 16bit Automatic Mixed Precision', 'GPU available:', 'TPU available:'
                        ]) and line.strip():
                            # 显示有用的信息
                            if any(useful in line for useful in [
                                '✅', '📊', '🎯', '📁', '⚠️', '❌', 
                                'accuracy', 'loss', 'test_result', 'final_model',
                                'Lightning训练', '模型加载', '可训练参数'
                            ]):
                                print(f"   {line}")
            
            # 确保最后换行
            if current_progress_line:
                print()
            
            # 等待进程结束
            process.wait()
            
            if process.returncode != 0:
                print(f"❌ 命令失败 (退出码: {process.returncode})")
                # 显示最后几行输出作为错误信息
                if stdout_lines:
                    print("最后的输出:")
                    for line in stdout_lines[-5:]:
                        if line.strip():
                            print(f"   {line}")
                return None
            
            print(f"✅ {description} 完成")
            return '\n'.join(stdout_lines)
            
        except Exception as e:
            print(f"❌ 执行命令时出错: {e}")
            return None
    
    def _parse_training_output(self, output):
        """解析训练输出，提取准确率"""
        try:
            lines = output.split('\n')
            for line in lines:
                if 'test/accuracy' in line and 'test_result' in line:
                    # 提取类似 'test/accuracy': 0.7465870380401611 的信息
                    import re
                    match = re.search(r"'test/accuracy':\s*([\d.]+)", line)
                    if match:
                        return float(match.group(1))
            return None
        except Exception as e:
            print(f"⚠️ 解析训练输出时出错: {e}")
            return None
    
    def train_model(self, model_path, dataset):
        """训练模型+LoRA"""
        model_name = self._get_model_short_name(model_path)
        print(f"\n📚 开始训练 {model_name} + LoRA (数据集: {dataset})")
        
        # 检查是否已有训练结果
        runs_dir = os.path.join(self.config['paths']['runs_dir'], dataset, model_name)
        if os.path.exists(runs_dir):
            # 查找最新的训练结果
            existing_runs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
            if existing_runs:
                latest_run = sorted(existing_runs)[-1]
                final_model_path = os.path.join(runs_dir, latest_run, "final_model")
                if os.path.exists(final_model_path):
                    print(f"✅ 发现已有训练结果: {final_model_path}")
                    return final_model_path, None  # 返回路径和空的准确率(需要评估)
        
        # 构建训练命令
        cmd = f"cd {os.path.dirname(self.config['paths']['train_script'])} && " \
              f"python {os.path.basename(self.config['paths']['train_script'])} " \
              f"--dataset {dataset} " \
              f"--base_model {model_path} " \
              f"--bs {self.config['training']['default_batch_size']} " \
              f"--max_steps {self.config['training']['default_max_steps']}"
        
        output = self._run_command(cmd, f"训练 {model_name} LoRA")
        if output is None:
            return None, None
        
        # 解析输出获取模型路径和准确率
        accuracy = self._parse_training_output(output)
        
        # 查找生成的模型路径
        runs_dir = os.path.join(self.config['paths']['runs_dir'], dataset, model_name)
        if os.path.exists(runs_dir):
            runs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
            if runs:
                latest_run = sorted(runs)[-1]
                final_model_path = os.path.join(runs_dir, latest_run, "final_model")
                return final_model_path, accuracy
        
        return None, None
    
    def transfer_lora(self, source_lora_path, source_model, target_model, dataset):
        """迁移LoRA"""
        source_name = self._get_model_short_name(source_model)
        target_name = self._get_model_short_name(target_model)
        
        print(f"\n🔄 开始迁移 LoRA: {source_name} → {target_name}")
        
        # 创建输出目录
        output_dir = os.path.join(
            self.config['paths']['transferred_lora_dir'],
            dataset,
            f"{source_name}_to_{target_name}",
            self.timestamp
        )
        os.makedirs(output_dir, exist_ok=True)
        
        # 构建迁移命令
        cmd = f"cd {os.path.dirname(self.config['paths']['transfer_script'])} && " \
              f"python {os.path.basename(self.config['paths']['transfer_script'])} " \
              f"--source_lora {source_lora_path} " \
              f"--source_model {source_model} " \
              f"--target_model {target_model} " \
              f"--output {output_dir} " \
              f"--similarity_threshold {self.config['transfer']['similarity_threshold']}"
        
        output = self._run_command(cmd, f"迁移 LoRA ({source_name} → {target_name})")
        if output is None:
            return None
        
        return output_dir
    
    def evaluate_model(self, model_path, base_model, dataset, is_lora=True):
        """评估模型性能"""
        model_type = "LoRA" if is_lora else "基础模型"
        model_name = self._get_model_short_name(base_model)
        
        print(f"\n📊 开始评估 {model_name} {model_type}")
        
        # 构建评估命令，保持在项目根目录
        if is_lora:
            cmd = f"python {self.config['paths']['eval_script']} " \
                  f"--models_list {model_path} " \
                  f"--dataset {dataset} " \
                  f"--sample_ratio {self.config['evaluation']['sample_ratio']} " \
                  f"--base_model {base_model}"
        else:
            # 评估基础模型 - 也需要使用models_list参数
            cmd = f"python {self.config['paths']['eval_script']} " \
                  f"--models_list {base_model} " \
                  f"--dataset {dataset} " \
                  f"--sample_ratio {self.config['evaluation']['sample_ratio']}"
        
        output = self._run_command(cmd, f"评估 {model_name} {model_type}")
        if output is None:
            return None
        
        # 解析评估输出获取准确率
        try:
            import re
            
            # 方法1: 查找表格格式的accuracy
            table_pattern = r"\|\s*test/accuracy\s*\|\s*([\d.]+)\s*\|"
            match = re.search(table_pattern, output, re.IGNORECASE)
            if match:
                accuracy = float(match.group(1))
                print(f"   📊 从表格提取准确率: {accuracy:.4f}")
                return accuracy
            
            # 方法2: 查找字典格式的accuracy
            dict_pattern = r"['\"]?test/accuracy['\"]?\s*[:\|]\s*([\d.]+)"
            match = re.search(dict_pattern, output, re.IGNORECASE)
            if match:
                accuracy = float(match.group(1))
                print(f"   📊 从字典提取准确率: {accuracy:.4f}")
                return accuracy
            
            # 方法3: 查找一般accuracy信息
            general_pattern = r"accuracy['\"]?\s*[:\|=]\s*([\d.]+)"
            match = re.search(general_pattern, output, re.IGNORECASE)
            if match:
                accuracy = float(match.group(1))
                print(f"   📊 从一般格式提取准确率: {accuracy:.4f}")
                return accuracy
            
            # 方法4: 逐行分析
            lines = output.split('\n')
            for line in lines:
                line = line.strip()
                if 'accuracy' in line.lower() and any(char.isdigit() for char in line):
                    # 提取行中的所有数字
                    numbers = re.findall(r'\d+\.?\d*', line)
                    for num_str in numbers:
                        try:
                            num = float(num_str)
                            # 准确率通常在0-1之间
                            if 0 <= num <= 1:
                                print(f"   📊 从行提取准确率: {num:.4f} (行: {line[:50]}...)")
                                return num
                        except:
                            continue
            
            print(f"   ⚠️ 未能提取准确率，返回None")
            # 调试：显示评估输出的关键部分
            print("   🔍 评估输出关键行:")
            for line in output.split('\n')[-20:]:  # 显示最后20行
                if any(keyword in line.lower() for keyword in ['accuracy', 'test', 'loss']):
                    print(f"     {line.strip()}")
            
        except Exception as e:
            print(f"⚠️ 解析评估输出时出错: {e}")
        
        return None
    
    def save_results(self, experiment_data):
        """保存实验结果"""
        print(f"\n💾 保存实验结果...")
        
        # 深度清理数据，确保所有值都是Python原生类型
        def deep_clean_value(value):
            if value is None:
                return None
            elif hasattr(value, 'item'):  # numpy scalar
                try:
                    return float(value.item())
                except:
                    return str(value)
            elif hasattr(value, 'tolist'):  # numpy array
                try:
                    return value.tolist()
                except:
                    return str(value)
            elif isinstance(value, (list, tuple)):
                return [deep_clean_value(v) for v in value]
            elif isinstance(value, dict):
                return {k: deep_clean_value(v) for k, v in value.items()}
            elif isinstance(value, float) and (value != value):  # NaN
                return None
            elif isinstance(value, (int, float, str, bool)):
                return value
            else:
                return str(value)
        
        clean_data = {}
        for key, value in experiment_data.items():
            clean_data[key] = deep_clean_value(value)
        
        # 保存到CSV
        csv_path = os.path.join(self.config['paths']['results_dir'], 
                               self.config['results']['csv_file'])
        
        try:
            # 创建DataFrame时指定列
            expected_columns = [
                'experiment_id', 'source_model', 'target_model', 'dataset',
                'source_acc', 'source_lora_acc', 'target_acc', 'target_lora_acc', 'transferred_acc',
                'source_lora_path', 'target_lora_path', 'transferred_lora_path',
                'timestamp', 'notes', 'training_config'
            ]
            
            # 确保所有期望的列都存在
            for col in expected_columns:
                if col not in clean_data:
                    clean_data[col] = None
            
            # 创建新行，确保所有值都是基本类型
            new_row_data = {}
            for col in expected_columns:
                value = clean_data[col]
                if value is None:
                    new_row_data[col] = None
                else:
                    new_row_data[col] = str(value) if not isinstance(value, (int, float, bool)) else value
            
            df_new = pd.DataFrame([new_row_data])
            
            if os.path.exists(csv_path):
                try:
                    # 安全读取CSV，处理可能的numpy数组
                    df_existing = pd.read_csv(csv_path, converters={
                        col: lambda x: x if not str(x).startswith('[') else str(x) 
                        for col in expected_columns
                    })
                    # 确保列顺序一致
                    for col in expected_columns:
                        if col not in df_existing.columns:
                            df_existing[col] = None
                    df_existing = df_existing[expected_columns]
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                except Exception as e:
                    print(f"⚠️ 读取现有结果时出错: {e}")
                    print("💡 将创建新的结果文件")
                    df_combined = df_new
            else:
                df_combined = df_new
            
            df_combined.to_csv(csv_path, index=False)
            
        except Exception as e:
            print(f"❌ 保存CSV时出错: {e}")
            print("💡 将尝试简化数据格式保存")
            
            # 备用保存方法：直接写入文本格式
            backup_path = csv_path.replace('.csv', '_backup.txt')
            try:
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(f"实验结果备份 - {datetime.now()}\n")
                    f.write("=" * 50 + "\n")
                    for key, value in clean_data.items():
                        f.write(f"{key}: {value}\n")
                print(f"📝 结果已保存到备用文件: {backup_path}")
                return  # 提前返回，避免调用_update_markdown_summary
            except Exception as backup_error:
                print(f"❌ 备用保存也失败: {backup_error}")
                return
        
        # 更新Markdown总结
        try:
            self._update_markdown_summary(df_combined)
        except Exception as md_error:
            print(f"⚠️ 更新Markdown失败: {md_error}")
        
        print(f"✅ 结果已保存到:")
        print(f"   📊 CSV: {csv_path}")
        print(f"   📝 Markdown: {os.path.join(self.config['paths']['results_dir'], self.config['results']['markdown_file'])}")
    
    def _update_markdown_summary(self, df):
        """更新Markdown总结文件"""
        md_path = os.path.join(self.config['paths']['results_dir'], 
                              self.config['results']['markdown_file'])
        
        content = f"""# 📊 LoRA迁移实验结果汇总

> 自动生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
> 管理脚本: transfer_pipeline.py  
> 总实验数: {len(df)}

## 实验概述

本文档记录所有LoRA训练和迁移实验的结果，包括：
- 基础模型性能
- LoRA微调后性能  
- 跨模型LoRA迁移性能
- 详细的配置信息

---

## 最新实验结果

"""
        
        # 按数据集分组显示结果
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            content += f"\n### 数据集: {dataset}\n\n"
            
            for _, row in dataset_df.iterrows():
                content += f"#### 实验: {row['source_model']} → {row['target_model']}\n\n"
                content += "| 模型配置 | 准确率 | 提升 | 备注 |\n"
                content += "|---------|--------|------|------|\n"
                
                # 基础模型行
                content += f"| {self._get_model_short_name(row['source_model'])} (source) | {row['source_acc']:.4f} | - | 基础模型 |\n"
                if pd.notna(row['source_lora_acc']):
                    improvement = (row['source_lora_acc'] - row['source_acc']) * 100
                    content += f"| {self._get_model_short_name(row['source_model'])} + LoRA | {row['source_lora_acc']:.4f} | +{improvement:.2f}% | 源模型微调 |\n"
                
                content += f"| {self._get_model_short_name(row['target_model'])} (target) | {row['target_acc']:.4f} | - | 基础模型 |\n"
                if pd.notna(row['target_lora_acc']):
                    improvement = (row['target_lora_acc'] - row['target_acc']) * 100
                    content += f"| {self._get_model_short_name(row['target_model'])} + LoRA | {row['target_lora_acc']:.4f} | +{improvement:.2f}% | 目标模型微调 |\n"
                
                if pd.notna(row['transferred_acc']):
                    improvement = (row['transferred_acc'] - row['target_acc']) * 100
                    content += f"| {self._get_model_short_name(row['target_model'])} + 迁移LoRA | {row['transferred_acc']:.4f} | +{improvement:.2f}% | 迁移LoRA |\n"
                
                content += f"\n**实验时间:** {row['timestamp']}  \n"
                content += f"**配置:** {row['training_config']}\n\n"
                content += "---\n\n"
        
        content += f"""
## 使用说明

### 运行新实验
```bash
python transfer_pipeline.py \\
  --source_model Llama-3.2-3B-Instruct \\
  --target_model Qwen_Qwen2.5-1.5B \\
  --dataset arc-challenge
```

### 查看详细数据
```bash
# 查看CSV格式数据 (可用Excel打开)
cat results/experiment_results.csv

# 只运行评估 (跳过训练)
python transfer_pipeline.py --source_model ... --target_model ... --dataset ... --eval_only
```

---

*最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _save_partial_results(self, results, status_message):
        """立即保存部分结果"""
        try:
            print(f"💾 {status_message} - 立即更新结果...")
            
            # 深度清理数据类型，确保pandas兼容
            def deep_clean_value(value):
                if value is None:
                    return None
                elif hasattr(value, 'item'):  # numpy scalar
                    try:
                        return float(value.item())
                    except:
                        return str(value)
                elif hasattr(value, 'tolist'):  # numpy array
                    try:
                        return value.tolist()
                    except:
                        return str(value)
                elif isinstance(value, (list, tuple)):
                    return [deep_clean_value(v) for v in value]
                elif isinstance(value, dict):
                    return {k: deep_clean_value(v) for k, v in value.items()}
                elif isinstance(value, float) and (value != value):  # NaN
                    return None
                elif isinstance(value, (int, float, str, bool)):
                    return value
                else:
                    return str(value)
            
            clean_results = {}
            for key, value in results.items():
                clean_results[key] = deep_clean_value(value)
            
            # 立即写入CSV (追加模式)
            csv_path = os.path.join(self.config['paths']['results_dir'], 
                                   self.config['results']['csv_file'])
            
            # 确保目录存在
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            
            # 读取现有数据
            if os.path.exists(csv_path):
                try:
                    # 安全读取CSV，处理可能的类型问题
                    df_existing = pd.read_csv(csv_path, dtype=str)  # 先都读成字符串
                    # 然后处理数值列
                    numeric_cols = ['source_acc', 'target_acc', 'transferred_acc', 'source_lora_acc', 'target_lora_acc']
                    for col in numeric_cols:
                        if col in df_existing.columns:
                            df_existing[col] = pd.to_numeric(df_existing[col], errors='coerce')
                    
                    # 确保列类型一致
                    for col in df_existing.columns:
                        if col in clean_results:
                            # 将所有值转换为字符串再转回适当类型，避免类型冲突
                            if df_existing[col].dtype == 'object':
                                continue
                            try:
                                clean_results[col] = str(clean_results[col]) if clean_results[col] is not None else None
                            except:
                                clean_results[col] = str(clean_results[col])
                except Exception as read_error:
                    print(f"⚠️ 读取现有CSV失败: {read_error}")
                    df_existing = pd.DataFrame()
            else:
                df_existing = pd.DataFrame()
            
            # 检查是否已存在相同实验
            experiment_id = clean_results.get('experiment_id')
            if not df_existing.empty and experiment_id:
                # 更新现有记录
                mask = df_existing['experiment_id'] == experiment_id
                if mask.any():
                    for key, value in clean_results.items():
                        if key in df_existing.columns:
                            df_existing.loc[mask, key] = value
                        else:
                            df_existing[key] = None
                            df_existing.loc[mask, key] = value
                else:
                    # 添加新记录
                    new_row = pd.DataFrame([clean_results])
                    df_existing = pd.concat([df_existing, new_row], ignore_index=True)
            else:
                # 添加新记录
                new_row = pd.DataFrame([clean_results])
                df_existing = pd.concat([df_existing, new_row], ignore_index=True)
            
            # 保存更新后的CSV
            df_existing.to_csv(csv_path, index=False)
            
            # 立即更新Markdown摘要
            self._update_markdown_summary(df_existing)
            
            print(f"✅ 结果已保存: {csv_path}")
            
        except Exception as e:
            print(f"⚠️ 部分结果保存失败: {e}")
            # 备用保存方法
            backup_path = os.path.join(self.config['paths']['results_dir'], 
                                     f"partial_backup_{self.timestamp}.json")
            try:
                with open(backup_path, 'w', encoding='utf-8') as f:
                    json.dump(clean_results, f, indent=2, ensure_ascii=False, default=str)
                print(f"📝 备用保存: {backup_path}")
            except Exception as backup_error:
                print(f"⚠️ 备用保存也失败: {backup_error}")
                # 最后的文本备用
                text_backup = os.path.join(self.config['paths']['results_dir'], 
                                         f"text_backup_{self.timestamp}.txt")
                with open(text_backup, 'w', encoding='utf-8') as f:
                    f.write(f"实验结果 - {datetime.now()}\n")
                    f.write("=" * 50 + "\n")
                    for key, value in results.items():
                        f.write(f"{key}: {value}\n")
                print(f"📝 文本备用保存: {text_backup}")
    
    def _check_step_completed(self, step_name, model_path, dataset):
        """检查步骤是否已完成，返回(已完成, 结果)"""
        if step_name == "source_lora_training":
            model_name = self._get_model_short_name(model_path)
            runs_dir = os.path.join(self.config['paths']['runs_dir'], dataset, model_name)
            if os.path.exists(runs_dir):
                existing_runs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
                if existing_runs:
                    latest_run = sorted(existing_runs)[-1]
                    final_model_path = os.path.join(runs_dir, latest_run, "final_model")
                    if os.path.exists(final_model_path):
                        return True, final_model_path
        
        elif step_name == "target_lora_training":
            model_name = self._get_model_short_name(model_path)
            runs_dir = os.path.join(self.config['paths']['runs_dir'], dataset, model_name)
            if os.path.exists(runs_dir):
                existing_runs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
                if existing_runs:
                    latest_run = sorted(existing_runs)[-1]
                    final_model_path = os.path.join(runs_dir, latest_run, "final_model")
                    if os.path.exists(final_model_path):
                        return True, final_model_path
        
        elif step_name == "lora_transfer":
            # 检查迁移结果目录
            source_name = self._get_model_short_name(model_path)  # 这里model_path是source_model
            target_name = self._get_model_short_name(dataset)     # 这里dataset是target_model (参数重用)
            transfer_base_dir = os.path.join(
                self.config['paths']['transferred_lora_dir'],
                dataset,  # 实际dataset
                f"{source_name}_to_{target_name}"
            )
            if os.path.exists(transfer_base_dir):
                existing_transfers = [d for d in os.listdir(transfer_base_dir) if os.path.isdir(os.path.join(transfer_base_dir, d))]
                if existing_transfers:
                    latest_transfer = sorted(existing_transfers)[-1]
                    transfer_path = os.path.join(transfer_base_dir, latest_transfer)
                    return True, transfer_path
        
        return False, None
    
    def run_pipeline(self, source_model, target_model, dataset, eval_only=False):
        """运行完整管道 - 新流程：训练源LoRA → 迁移 → 评估目标基础 → 评估迁移LoRA → 训练目标LoRA → 评估源基础"""
        self.experiment_id = self._create_experiment_id(source_model, target_model, dataset)
        
        print(f"\n🎯 开始LoRA迁移实验")
        print(f"📋 实验ID: {self.experiment_id}")
        print(f"🎲 源模型: {source_model}")
        print(f"🎯 目标模型: {target_model}")
        print(f"📚 数据集: {dataset}")
        print("=" * 80)
        
        # 检查是否有历史记录
        existing = self._check_existing_experiment(source_model, target_model, dataset)
        if existing is not None and not eval_only:
            print(f"⚠️ 发现相同实验记录 (时间: {existing['timestamp']})")
            response = input("是否继续? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print("🚫 实验取消")
                return False
        
        # 初始化结果字典
        results = {
            'experiment_id': self.experiment_id,
            'source_model': source_model,
            'target_model': target_model,
            'dataset': dataset,
            'timestamp': self.timestamp,
            'training_config': f"batch_size={self.config['training']['default_batch_size']}, "
                              f"max_steps={self.config['training']['default_max_steps']}, "
                              f"lr={self.config['training']['default_lr']}",
            'notes': '自动化管道生成'
        }
        
        # 🎯 新流程：6个步骤
        total_steps = 6 if not eval_only else 4
        progress_bar = tqdm(total=total_steps, desc="🚀 LoRA迁移管道", position=0, leave=True)
        
        try:
            if not eval_only:
                # 步骤1: 训练源LoRA (自动包含测试)
                progress_bar.set_description("🎯 步骤1: 训练源LoRA")
                source_lora_path, source_lora_acc = self.train_model(source_model, dataset)
                if source_lora_path is None:
                    raise Exception("源模型训练失败")
                
                results.update({
                    'source_lora_path': source_lora_path,
                    'source_lora_acc': source_lora_acc,
                })
                self._save_partial_results(results, "源LoRA训练完成")
                progress_bar.update(1)
                
                # 步骤2: 迁移LoRA  
                progress_bar.set_description("🔄 步骤2: 迁移LoRA")
                transferred_lora_path = self.transfer_lora(
                    source_lora_path, source_model, target_model, dataset
                )
                if transferred_lora_path is None:
                    raise Exception("LoRA迁移失败")
                
                results['transferred_lora_path'] = transferred_lora_path
                self._save_partial_results(results, "LoRA迁移完成")
                progress_bar.update(1)
            
            # 步骤3: 评估目标基础模型
            progress_bar.set_description("📊 步骤3: 评估目标基础模型")
            target_acc = self.evaluate_model(target_model, target_model, dataset, is_lora=False)
            results['target_acc'] = target_acc
            self._save_partial_results(results, "目标基础模型评估完成")
            progress_bar.update(1)
            
            if not eval_only:
                # 步骤4: 评估迁移LoRA
                progress_bar.set_description("📊 步骤4: 评估迁移LoRA")
                transferred_acc = self.evaluate_model(
                    results['transferred_lora_path'], target_model, dataset, is_lora=True
                )
                results['transferred_acc'] = transferred_acc
                self._save_partial_results(results, "迁移LoRA评估完成")
                progress_bar.update(1)
                
                # 步骤5: 训练目标LoRA (自动包含测试)
                progress_bar.set_description("🎯 步骤5: 训练目标LoRA")
                target_lora_path, target_lora_acc = self.train_model(target_model, dataset)
                if target_lora_path is None:
                    print("⚠️ 目标模型训练失败，跳过")
                    target_lora_acc = None
                
                results.update({
                    'target_lora_path': target_lora_path,
                    'target_lora_acc': target_lora_acc,
                })
                self._save_partial_results(results, "目标LoRA训练完成")
                progress_bar.update(1)
            
            # 步骤6: 评估源基础模型 (补齐)
            progress_bar.set_description("📊 步骤6: 评估源基础模型")
            source_acc = self.evaluate_model(source_model, source_model, dataset, is_lora=False)
            results['source_acc'] = source_acc
            self._save_partial_results(results, "源基础模型评估完成")
            progress_bar.update(1)
            
            # 最终保存完整结果
            progress_bar.set_description("💾 保存最终结果")
            self.save_results(results)
            progress_bar.close()
            
            # 打印总结
            self._print_summary(results)
            
            # 提示可选命令
            if not eval_only:
                self._print_optional_commands(source_model, target_model, dataset)
            
            print("\n🎉 管道执行成功!")
            return True
            
        except Exception as e:
            progress_bar.close()
            print(f"\n❌ 管道执行失败: {e}")
            # 保存部分结果
            self._save_partial_results(results, f"失败: {e}")
            return False
    
    def _print_summary(self, results):
        """打印实验总结"""
        print(f"\n🎉 实验完成! 总结如下:")
        print("=" * 60)
        
        source_name = self._get_model_short_name(results['source_model'])
        target_name = self._get_model_short_name(results['target_model'])
        
        # 处理可能为None的值
        source_acc = results.get('source_acc', 0)
        target_acc = results.get('target_acc', 0)
        source_lora_acc = results.get('source_lora_acc')
        target_lora_acc = results.get('target_lora_acc')
        transferred_acc = results.get('transferred_acc')
        
        print(f"📊 {source_name} (源模型): {source_acc:.4f}")
        if source_lora_acc is not None:
            improvement = (source_lora_acc - source_acc) * 100
            print(f"📊 {source_name} + LoRA: {source_lora_acc:.4f} (+{improvement:.2f}%)")
        
        print(f"📊 {target_name} (目标模型): {target_acc:.4f}")
        
        if transferred_acc is not None:
            improvement = (transferred_acc - target_acc) * 100
            print(f"📊 {target_name} + 迁移LoRA: {transferred_acc:.4f} (+{improvement:.2f}%)")
        
        if target_lora_acc is not None:
            improvement = (target_lora_acc - target_acc) * 100
            print(f"📊 {target_name} + 直训LoRA: {target_lora_acc:.4f} (+{improvement:.2f}%)")
        
        print("=" * 60)
        print(f"📁 详细结果: results/experiment_summary.md")
    
    def _print_optional_commands(self, source_model, target_model, dataset):
        """打印可选的目标模型LoRA训练命令"""
        target_name = self._get_model_short_name(target_model)
        
        print(f"\n💡 可选：训练目标模型 {target_name} 的LoRA进行对比")
        print("=" * 60)
        
        # 训练命令
        train_cmd = f"python {self.config['paths']['train_script']} " \
                   f"--dataset {dataset} " \
                   f"--base_model {target_model} " \
                   f"--bs {self.config['training']['default_batch_size']} " \
                   f"--max_steps {self.config['training']['default_max_steps']}"
        
        print(f"� 训练 {target_name} LoRA:")
        print(f"   {train_cmd}")
        
        # 评估命令 
        eval_cmd = f"python {self.config['paths']['eval_script']} " \
                  f"--models_list [训练后的模型路径] " \
                  f"--dataset {dataset} " \
                  f"--sample_ratio {self.config['evaluation']['sample_ratio']} " \
                  f"--base_model {target_model}"
        
        print(f"\n📊 评估 {target_name} LoRA:")
        print(f"   {eval_cmd}")
        print()
        print("💡 训练完成后可以对比 '目标模型+LoRA' vs '目标模型+迁移LoRA' 的性能差异")


def main():
    parser = argparse.ArgumentParser(
        description="LoRA训练和迁移自动化管道",
        epilog="""
使用示例:
  # 快速测试 (0.5B→1.5B, 20步训练, 5%评估)
  python transfer_pipeline.py --quick_test
  
  # 自定义模型
  python transfer_pipeline.py --source_model gemma-2-2b-it --target_model Qwen_Qwen2.5-1.5B --dataset arc-challenge
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source_model", type=str, 
                       help="源模型路径或名称")
    parser.add_argument("--target_model", type=str, 
                       help="目标模型路径或名称")
    parser.add_argument("--dataset", type=str, 
                       help="数据集名称")
    parser.add_argument("--config", type=str, default="config/pipeline_config.yaml",
                       help="配置文件路径")
    parser.add_argument("--eval_only", action="store_true",
                       help="仅运行评估，跳过训练和迁移")
    parser.add_argument("--quick_test", action="store_true",
                       help="快速测试模式：自动使用0.5B→1.5B配置")
    
    args = parser.parse_args()
    
    # 快速测试模式：使用预设配置
    if args.quick_test:
        print("🚀 快速测试模式：0.5B → 1.5B")
        args.config = "config/quick_test_config.yaml"
        quick_config = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
        
        # 使用推荐配置
        if not args.source_model:
            args.source_model = quick_config['recommended_models']['source']
        if not args.target_model:
            args.target_model = quick_config['recommended_models']['target'] 
        if not args.dataset:
            args.dataset = quick_config['recommended_models']['dataset']
            
        print(f"📦 源模型: {args.source_model}")
        print(f"🎯 目标模型: {args.target_model}")
        print(f"📚 数据集: {args.dataset}")
        print(f"⚡ 训练步数: 20, 评估比例: 5%")
        print("")
    
    # 验证必需参数
    if not all([args.source_model, args.target_model, args.dataset]):
        print("❌ 错误: 需要指定 --source_model, --target_model, --dataset")
        print("💡 或者使用 --quick_test 自动配置")
        parser.print_help()
        return
    
    # 验证模型路径
    config = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    models_dir = config['paths']['models_dir']
    
    # 处理模型路径
    if not args.source_model.startswith('/'):
        args.source_model = os.path.join(models_dir, args.source_model)
    if not args.target_model.startswith('/'):
        args.target_model = os.path.join(models_dir, args.target_model)
    
    # 验证模型存在
    if not os.path.exists(args.source_model):
        print(f"❌ 源模型不存在: {args.source_model}")
        return False
    if not os.path.exists(args.target_model):
        print(f"❌ 目标模型不存在: {args.target_model}")
        return False
    
    # 验证数据集
    if args.dataset not in config['training']['datasets']:
        print(f"❌ 不支持的数据集: {args.dataset}")
        print(f"✅ 支持的数据集: {', '.join(config['training']['datasets'])}")
        return False
    
    # 运行管道
    pipeline = TransferPipeline(args.config)
    success = pipeline.run_pipeline(
        args.source_model, 
        args.target_model, 
        args.dataset,
        eval_only=args.eval_only
    )
    
    if success:
        print(f"\n🎉 管道执行成功!")
        return True
    else:
        print(f"\n❌ 管道执行失败!")
        return False


if __name__ == "__main__":
    main()
