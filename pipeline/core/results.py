"""
结果管理模块
负责实验结果的保存、加载和管理
"""

import os
import csv
from datetime import datetime
from typing import Dict, Any, Optional, List
from .config import PipelineConfig
from .utils import ModelUtils


class ResultsManager:
    """结果管理器 - 基于CSV格式"""
    
    def __init__(self, config: PipelineConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self._ensure_results_dir()
        self.csv_headers = [
            'base_model', 'lora_source', 'dataset', 'accuracy', 'improvement_pct', 
            'config_details', 'run_file', 'timestamp', 'note'
        ]
    
    def _ensure_results_dir(self):
        """确保结果目录存在"""
        results_dir = self.config.get('paths.results_dir')
        os.makedirs(results_dir, exist_ok=True)
    
    def _get_csv_path(self) -> str:
        """获取CSV文件路径"""
        return os.path.join(
            self.config.get('paths.results_dir'), 
            self.config.get('results.csv_file')
        )
    
    def _ensure_csv_exists(self):
        """确保CSV文件存在，如果不存在则创建带头部的文件"""
        csv_path = self._get_csv_path()
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.csv_headers)
    
    def add_result(self, base_model: str, lora_source: str, dataset: str, 
                   accuracy: float, improvement_pct: float, config_details: str,
                   run_file: str = "", note: str = ""):
        """添加单个结果到CSV"""
        self._ensure_csv_exists()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        row_data = [
            base_model, lora_source, dataset, f"{accuracy:.4f}", 
            f"{improvement_pct:.2f}", config_details, run_file, timestamp, note
        ]
        
        csv_path = self._get_csv_path()
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)
        
        if self.verbose:
            print(f"💾 结果已添加: {base_model} - {lora_source} - {accuracy:.4f}")
    
    def save_pipeline_results(self, experiment_data: Dict[str, Any]):
        """保存完整管道实验结果"""
        if self.verbose:
            print(f"\n💾 保存管道实验结果...")
        
        # 提取基础信息
        source_model = ModelUtils.get_model_short_name(experiment_data.get('source_model', ''))
        target_model = ModelUtils.get_model_short_name(experiment_data.get('target_model', ''))
        dataset = experiment_data.get('dataset', '')
        experiment_id = experiment_data.get('experiment_id', '')
        
        # 获取配置信息
        training_config = experiment_data.get('training_config', '')
        
        # 添加各种模型结果
        results_added = 0
        
        # 1. 源基础模型
        if 'source_acc' in experiment_data and experiment_data['source_acc'] is not None:
            self.add_result(
                base_model=source_model,
                lora_source="base",
                dataset=dataset,
                accuracy=experiment_data['source_acc'],
                improvement_pct=0.0,
                config_details="-",
                run_file=experiment_id,
                note="源基础模型"
            )
            results_added += 1
        
        # 2. 源LoRA模型
        if 'source_lora_acc' in experiment_data and experiment_data['source_lora_acc'] is not None:
            source_base_acc = experiment_data.get('source_acc', 0)
            improvement = ((experiment_data['source_lora_acc'] - source_base_acc) / source_base_acc * 100) if source_base_acc > 0 else 0
            
            self.add_result(
                base_model=source_model,
                lora_source="lora",
                dataset=dataset,
                accuracy=experiment_data['source_lora_acc'],
                improvement_pct=improvement,
                config_details=f"LoRA: {source_model}, {training_config}",
                run_file=experiment_data.get('source_lora_path', ''),
                note="源LoRA模型"
            )
            results_added += 1
        
        # 3. 目标基础模型
        if 'target_acc' in experiment_data and experiment_data['target_acc'] is not None:
            self.add_result(
                base_model=target_model,
                lora_source="base",
                dataset=dataset,
                accuracy=experiment_data['target_acc'],
                improvement_pct=0.0,
                config_details="-",
                run_file=experiment_id,
                note="目标基础模型"
            )
            results_added += 1
        
        # 4. 迁移LoRA模型
        if 'transferred_acc' in experiment_data and experiment_data['transferred_acc'] is not None:
            target_base_acc = experiment_data.get('target_acc', 0)
            improvement = ((experiment_data['transferred_acc'] - target_base_acc) / target_base_acc * 100) if target_base_acc > 0 else 0
            
            # 获取迁移配置
            similarity_threshold = self.config.get('transfer.similarity_threshold', 0.0001)
            transfer_config = f"LoRA source: {source_model}, {training_config}; Adapter: 迁移, sim={similarity_threshold}"
            
            self.add_result(
                base_model=target_model,
                lora_source="adpt",
                dataset=dataset,
                accuracy=experiment_data['transferred_acc'],
                improvement_pct=improvement,
                config_details=transfer_config,
                run_file=experiment_data.get('transferred_lora_path', ''),
                note="迁移LoRA模型"
            )
            results_added += 1
        
        # 5. 目标LoRA模型
        if 'target_lora_acc' in experiment_data and experiment_data['target_lora_acc'] is not None:
            target_base_acc = experiment_data.get('target_acc', 0)
            improvement = ((experiment_data['target_lora_acc'] - target_base_acc) / target_base_acc * 100) if target_base_acc > 0 else 0
            
            self.add_result(
                base_model=target_model,
                lora_source="lora",
                dataset=dataset,
                accuracy=experiment_data['target_lora_acc'],
                improvement_pct=improvement,
                config_details=f"LoRA: {target_model}, {training_config}",
                run_file=experiment_data.get('target_lora_path', ''),
                note="目标LoRA模型"
            )
            results_added += 1
        
        if self.verbose:
            print(f"✅ 已添加 {results_added} 条结果到 {self._get_csv_path()}")
        
        return True
    
    def save_results(self, experiment_data: Dict[str, Any]):
        """保存实验结果 - 兼容原接口，但避免重复保存"""
        if self.verbose:
            print(f"\n💾 保存完整实验结果...")
        
        # 如果已经通过save_partial_results保存了结果，就不重复保存
        if hasattr(self, '_saved_keys') and self._saved_keys:
            if self.verbose:
                print(f"✅ 所有结果已通过增量保存完成，跳过重复保存")
            return True
        else:
            # 如果没有增量保存过，就使用完整保存
            return self.save_pipeline_results(experiment_data)
    
    def save_partial_results(self, results: Dict[str, Any], message: str):
        """保存部分结果 - 增量保存，避免重复"""
        if self.verbose:
            print(f"💾 {message} - 立即更新结果...")
        
        # 使用实例变量追踪已保存的结果，避免重复
        if not hasattr(self, '_saved_keys'):
            self._saved_keys = set()
        
        try:
            # 提取基础信息
            source_model = ModelUtils.get_model_short_name(results.get('source_model', ''))
            target_model = ModelUtils.get_model_short_name(results.get('target_model', ''))
            dataset = results.get('dataset', '')
            experiment_id = results.get('experiment_id', '')
            training_config = results.get('training_config', '')
            
            # 根据message类型决定保存哪个结果
            if "源LoRA训练完成" in message and 'source_lora_acc' in results:
                key = f"{source_model}_lora_{dataset}_{experiment_id}"
                if key not in self._saved_keys:
                    source_base_acc = results.get('source_acc', 0)
                    improvement = ((results['source_lora_acc'] - source_base_acc) / source_base_acc * 100) if source_base_acc > 0 else 0
                    self.add_result(
                        base_model=source_model,
                        lora_source="lora",
                        dataset=dataset,
                        accuracy=results['source_lora_acc'],
                        improvement_pct=improvement,
                        config_details=f"LoRA: {source_model}, {training_config}",
                        run_file=results.get('source_lora_path', ''),
                        note="源LoRA模型"
                    )
                    self._saved_keys.add(key)
            
            elif "目标基础模型评估完成" in message and 'target_acc' in results:
                key = f"{target_model}_base_{dataset}_{experiment_id}"
                if key not in self._saved_keys:
                    self.add_result(
                        base_model=target_model,
                        lora_source="base",
                        dataset=dataset,
                        accuracy=results['target_acc'],
                        improvement_pct=0.0,
                        config_details="-",
                        run_file=experiment_id,
                        note="目标基础模型"
                    )
                    self._saved_keys.add(key)
            
            elif "迁移LoRA评估完成" in message and 'transferred_acc' in results:
                key = f"{target_model}_adpt_{dataset}_{experiment_id}"
                if key not in self._saved_keys:
                    target_base_acc = results.get('target_acc', 0)
                    improvement = ((results['transferred_acc'] - target_base_acc) / target_base_acc * 100) if target_base_acc > 0 else 0
                    similarity_threshold = self.config.get('transfer.similarity_threshold', 0.0001)
                    transfer_config = f"LoRA source: {source_model}, {training_config}; Adapter: 迁移, sim={similarity_threshold}"
                    self.add_result(
                        base_model=target_model,
                        lora_source="adpt",
                        dataset=dataset,
                        accuracy=results['transferred_acc'],
                        improvement_pct=improvement,
                        config_details=transfer_config,
                        run_file=results.get('transferred_lora_path', ''),
                        note="迁移LoRA模型"
                    )
                    self._saved_keys.add(key)
            
            elif "目标LoRA训练完成" in message and 'target_lora_acc' in results:
                key = f"{target_model}_lora_{dataset}_{experiment_id}"
                if key not in self._saved_keys:
                    target_base_acc = results.get('target_acc', 0)
                    improvement = ((results['target_lora_acc'] - target_base_acc) / target_base_acc * 100) if target_base_acc > 0 else 0
                    self.add_result(
                        base_model=target_model,
                        lora_source="lora",
                        dataset=dataset,
                        accuracy=results['target_lora_acc'],
                        improvement_pct=improvement,
                        config_details=f"LoRA: {target_model}, {training_config}",
                        run_file=results.get('target_lora_path', ''),
                        note="目标LoRA模型"
                    )
                    self._saved_keys.add(key)
            
            elif "源基础模型评估完成" in message and 'source_acc' in results:
                key = f"{source_model}_base_{dataset}_{experiment_id}"
                if key not in self._saved_keys:
                    self.add_result(
                        base_model=source_model,
                        lora_source="base",
                        dataset=dataset,
                        accuracy=results['source_acc'],
                        improvement_pct=0.0,
                        config_details="-",
                        run_file=experiment_id,
                        note="源基础模型"
                    )
                    self._saved_keys.add(key)
            
            elif "LoRA迁移完成" in message:
                # 迁移完成但还没评估，不保存accuracy
                pass
                
        except Exception as e:
            if self.verbose:
                print(f"⚠️ 保存结果时出错: {e}")
                # 简单的备份到JSON
                backup_path = os.path.join(
                    self.config.get('paths.results_dir'),
                    f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                try:
                    import json
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
                    print(f"📝 备用保存: {backup_path}")
                except Exception as backup_e:
                    print(f"❌ 备用保存也失败: {backup_e}")
    
    def check_existing_experiment(self, source_model: str, target_model: str, dataset: str) -> Optional[Dict]:
        """检查是否存在相同的实验"""
        csv_path = self._get_csv_path()
        if not os.path.exists(csv_path):
            return None
        
        source_short = ModelUtils.get_model_short_name(source_model)
        target_short = ModelUtils.get_model_short_name(target_model)
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if (row['base_model'] in [source_short, target_short] and 
                        row['dataset'] == dataset and 
                        row['lora_source'] == 'adpt'):
                        return {'timestamp': row['timestamp']}
        except Exception:
            pass
        
        return None
