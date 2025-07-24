"""
结果管理模块
负责实验结果的保存、加载和管理
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from .config import PipelineConfig
from .utils import ModelUtils, DataCleaner


class ResultsManager:
    """结果管理器"""
    
    def __init__(self, config: PipelineConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self._ensure_results_dir()
    
    def _ensure_results_dir(self):
        """确保结果目录存在"""
        results_dir = self.config.get('paths.results_dir')
        os.makedirs(results_dir, exist_ok=True)
    
    def save_results(self, experiment_data: Dict[str, Any]):
        """保存实验结果"""
        if self.verbose:
            print(f"\n💾 保存实验结果...")
        
        # 清理数据
        clean_data = DataCleaner.clean_dict(experiment_data)
        
        # 保存到CSV
        success = self._save_to_csv(clean_data)
        
        if success:
            # 更新Markdown总结
            try:
                self._update_markdown_summary()
                if self.verbose:
                    csv_path = os.path.join(self.config.get('paths.results_dir'), 
                                          self.config.get('results.csv_file'))
                    md_path = os.path.join(self.config.get('paths.results_dir'), 
                                         self.config.get('results.markdown_file'))
                    print(f"Results saved to:")
                    print(f"  CSV: {csv_path}")
                    print(f"  Markdown: {md_path}")
            except Exception as md_error:
                if self.verbose:
                    print(f"WARNING: Failed to update Markdown: {md_error}")
        else:
            # Fallback save
            self._backup_save(clean_data)
    
    def _save_to_csv(self, clean_data: Dict[str, Any]) -> bool:
        """保存到CSV文件"""
        csv_path = os.path.join(self.config.get('paths.results_dir'), 
                               self.config.get('results.csv_file'))
        
        try:
            # 定义期望的列
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
            
            # 处理现有数据
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
                    if self.verbose:
                        print(f"WARNING: Error reading existing results: {e}")
                        print("NOTE: Will create new results file")
                    df_combined = df_new
            else:
                df_combined = df_new
            
            df_combined.to_csv(csv_path, index=False)
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"❌ 保存CSV时出错: {e}")
            return False
    
    def _backup_save(self, clean_data: Dict[str, Any]):
        """备用保存方法"""
        csv_path = os.path.join(self.config.get('paths.results_dir'), 
                               self.config.get('results.csv_file'))
        backup_path = csv_path.replace('.csv', '_backup.txt')
        
        try:
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(f"实验结果备份 - {datetime.now()}\n")
                f.write("=" * 50 + "\n")
                for key, value in clean_data.items():
                    f.write(f"{key}: {value}\n")
            
            if self.verbose:
                print(f"📝 结果已保存到备用文件: {backup_path}")
        except Exception as backup_error:
            if self.verbose:
                print(f"❌ 备用保存也失败: {backup_error}")
    
    def save_partial_results(self, results: Dict[str, Any], status_message: str):
        """立即保存部分结果"""
        try:
            if self.verbose:
                print(f"💾 {status_message} - 立即更新结果...")
            
            # 清理数据
            clean_results = DataCleaner.clean_dict(results)
            
            # 立即写入CSV (追加/更新模式)
            csv_path = os.path.join(self.config.get('paths.results_dir'), 
                                   self.config.get('results.csv_file'))
            
            # 读取现有数据
            df_existing = self._load_existing_csv(csv_path)
            
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
            self._update_markdown_summary()
            
            if self.verbose:
                print(f"✅ 结果已保存: {csv_path}")
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️ 部分结果保存失败: {e}")
            # 备用保存方法
            self._backup_partial_save(results, status_message)
    
    def _load_existing_csv(self, csv_path: str) -> pd.DataFrame:
        """安全加载现有CSV文件"""
        if not os.path.exists(csv_path):
            return pd.DataFrame()
        
        try:
            # 安全读取CSV，处理可能的类型问题
            df_existing = pd.read_csv(csv_path, dtype=str)  # 先都读成字符串
            # 然后处理数值列
            numeric_cols = ['source_acc', 'target_acc', 'transferred_acc', 'source_lora_acc', 'target_lora_acc']
            for col in numeric_cols:
                if col in df_existing.columns:
                    df_existing[col] = pd.to_numeric(df_existing[col], errors='coerce')
            
            return df_existing
        except Exception as read_error:
            if self.verbose:
                print(f"⚠️ 读取现有CSV失败: {read_error}")
            return pd.DataFrame()
    
    def _backup_partial_save(self, results: Dict[str, Any], status_message: str):
        """部分结果的备用保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON备用
        backup_path = os.path.join(self.config.get('paths.results_dir'), 
                                  f"partial_backup_{timestamp}.json")
        try:
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            if self.verbose:
                print(f"📝 备用保存: {backup_path}")
        except Exception:
            # 文本备用
            text_backup = os.path.join(self.config.get('paths.results_dir'), 
                                     f"text_backup_{timestamp}.txt")
            with open(text_backup, 'w', encoding='utf-8') as f:
                f.write(f"实验结果 - {status_message} - {datetime.now()}\n")
                f.write("=" * 50 + "\n")
                for key, value in results.items():
                    f.write(f"{key}: {value}\n")
            if self.verbose:
                print(f"📝 文本备用保存: {text_backup}")
    
    def check_existing_experiment(self, source_model: str, target_model: str, dataset: str) -> Optional[pd.Series]:
        """检查是否已存在相同实验"""
        csv_path = os.path.join(self.config.get('paths.results_dir'), 
                               self.config.get('results.csv_file'))
        
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
                return existing.iloc[-1]  # Return the latest record
        except Exception as e:
            if self.verbose:
                print(f"WARNING: Error reading history: {e}")
                print(f"NOTE: Will recreate results file")
        
        return None
    
    def _update_markdown_summary(self):
        """更新Markdown总结文件"""
        csv_path = os.path.join(self.config.get('paths.results_dir'), 
                               self.config.get('results.csv_file'))
        md_path = os.path.join(self.config.get('paths.results_dir'), 
                              self.config.get('results.markdown_file'))
        
        if not os.path.exists(csv_path):
            return
        
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                return
            
            content = self._generate_markdown_content(df)
            
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            if self.verbose:
                print(f"⚠️ 生成Markdown失败: {e}")
    
    def _generate_markdown_content(self, df: pd.DataFrame) -> str:
        """Generate Markdown content"""
        content = f"""# LoRA Transfer Experiment Results Summary

> Auto-generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
> Management script: transfer_pipeline.py  
> Total experiments: {len(df)}

## Experiment Overview

This document records all LoRA training and transfer experiment results, including:
- Base model performance
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
                source_acc = row.get('source_acc', 0)
                target_acc = row.get('target_acc', 0)
                
                content += f"| {ModelUtils.get_model_short_name(row['source_model'])} (source) | {source_acc:.4f} | - | 基础模型 |\n"
                
                if pd.notna(row.get('source_lora_acc')):
                    improvement = (row['source_lora_acc'] - source_acc) * 100
                    content += f"| {ModelUtils.get_model_short_name(row['source_model'])} + LoRA | {row['source_lora_acc']:.4f} | +{improvement:.2f}% | 源模型微调 |\n"
                
                content += f"| {ModelUtils.get_model_short_name(row['target_model'])} (target) | {target_acc:.4f} | - | 基础模型 |\n"
                
                if pd.notna(row.get('target_lora_acc')):
                    improvement = (row['target_lora_acc'] - target_acc) * 100
                    content += f"| {ModelUtils.get_model_short_name(row['target_model'])} + LoRA | {row['target_lora_acc']:.4f} | +{improvement:.2f}% | 目标模型微调 |\n"
                
                if pd.notna(row.get('transferred_acc')):
                    improvement = (row['transferred_acc'] - target_acc) * 100
                    content += f"| {ModelUtils.get_model_short_name(row['target_model'])} + 迁移LoRA | {row['transferred_acc']:.4f} | +{improvement:.2f}% | 迁移LoRA |\n"
                
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
        
        return content
