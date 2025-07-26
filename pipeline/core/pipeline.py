"""
LoRA训练和迁移主管道
整合所有模块，提供统一的管道接口
"""

import os
from typing import Dict, Any, Optional
from tqdm import tqdm

from .config import PipelineConfig, QuickTestConfig
from .trainer import ModelTrainer
from .evaluator import ModelEvaluator
from .transfer import LoRATransfer
from .results import ResultsManager
from .utils import ModelUtils, get_timestamp


class TransferPipeline:
    """LoRA迁移管道主类"""
    
    def __init__(self, config_path: str = None, quick_test: bool = False):
        """初始化管道
        
        Args:
            config_path: 配置文件路径
            quick_test: 是否使用快速测试配置
        """
        if quick_test:
            self.config = QuickTestConfig()
        else:
            self.config = PipelineConfig(config_path)
        
        self.timestamp = get_timestamp()
        self.experiment_id = None
        
        # 初始化模块
        self.trainer = ModelTrainer(self.config)
        self.evaluator = ModelEvaluator(self.config)
        self.transfer = LoRATransfer(self.config)
        self.results = ResultsManager(self.config)
    
    def run_pipeline(self, source_model: str, target_model: str, dataset: str, 
                    eval_only: bool = False) -> bool:
        """运行完整管道 - 新流程：评估源基础 → 训练源LoRA → 迁移 → 评估目标基础 → 评估迁移LoRA → 训练目标LoRA
        
        Args:
            source_model: 源模型路径
            target_model: 目标模型路径
            dataset: 数据集名称
            eval_only: 仅运行评估，跳过训练和迁移
            
        Returns:
            是否成功
        """
        # 验证输入
        if not self._validate_inputs(source_model, target_model, dataset):
            return False
        
        # 创建实验ID
        self.experiment_id = ModelUtils.create_experiment_id(
            source_model, target_model, dataset, self.timestamp
        )
        
        print("\n" + "="*60)
        print("LoRA Transfer Pipeline - Experiment Started")
        print("="*60)
        print(f"Experiment ID: {self.experiment_id}")
        print(f"Source Model:  {ModelUtils.get_model_short_name(source_model)}")
        print(f"Target Model:  {ModelUtils.get_model_short_name(target_model)}")
        print(f"Dataset:       {dataset}")
        
        # Display configuration info
        max_steps = self.config.get('training.default_max_steps', 600)
        sample_ratio = self.config.get('evaluation.sample_ratio', 1.0)
        batch_size = self.config.get('training.default_batch_size', 4)
        
        print(f"Training:      {max_steps} steps, batch_size={batch_size}")
        print(f"Evaluation:    {sample_ratio*100:.0f}% sample ratio")
        
        # Check if this is quick test mode
        if hasattr(self.config, 'is_quick_test') and self.config.is_quick_test:
            print(f"Mode:          Quick Test (Fast)")
        else:
            print(f"Mode:          Full Pipeline")
        
        print("="*60)
        
        # 检查是否有历史记录
        if not eval_only:
            existing = self.results.check_existing_experiment(source_model, target_model, dataset)
            if existing is not None:
                action = self._handle_existing_experiment(existing, source_model, target_model, dataset)
                if action == 'abort':
                    print("Experiment cancelled.")
                    return False
                elif action == 'delete':
                    print("🗑️ Cleaning up existing experiment outputs...")
                    self._cleanup_experiment_outputs(source_model, target_model, dataset)
                    print("✅ Cleanup completed. Starting fresh experiment.")
                elif action == 'continue':
                    print("⏩ Continuing with existing experiment outputs.")
        
        # 初始化结果字典
        results = self._init_results_dict(source_model, target_model, dataset)
        
        # 执行管道步骤
        return self._execute_pipeline_steps(results, source_model, target_model, dataset, eval_only)
    
    def _validate_inputs(self, source_model: str, target_model: str, dataset: str) -> bool:
        """验证输入参数"""
        # 处理模型路径
        if not source_model.startswith('/'):
            source_model = self.config.get_model_path(source_model)
        if not target_model.startswith('/'):
            target_model = self.config.get_model_path(target_model)
        
        # 验证模型存在
        if not ModelUtils.check_model_exists(source_model):
            print(f"❌ 源模型不存在: {source_model}")
            return False
        if not ModelUtils.check_model_exists(target_model):
            print(f"❌ 目标模型不存在: {target_model}")
            return False
        
        # 验证数据集
        supported_datasets = self.config.get('training.datasets', [])
        if dataset not in supported_datasets:
            print(f"❌ 不支持的数据集: {dataset}")
            print(f"✅ 支持的数据集: {', '.join(supported_datasets)}")
            return False
        
        return True
    
    def _handle_existing_experiment(self, existing: Dict, source_model: str, target_model: str, dataset: str) -> str:
        """处理已存在的实验，返回用户选择的操作"""
        import os
        import shutil
        import glob
        from pathlib import Path
        
        print(f"\nWarning: Found existing experiment (timestamp: {existing['timestamp']})")
        
        # 检查配置文件中的默认行为
        default_action = self.config.get('experiment_management.existing_experiment_action', 'prompt')
        
        if default_action == 'prompt':
            print("\nWhat would you like to do?")
            print("  [C]ontinue - Keep existing outputs and continue")
            print("  [D]elete   - Delete all existing outputs and start fresh")
            print("  [Y]es      - Same as Continue (for backward compatibility)")
            print("  [N]o       - Same as Abort (for backward compatibility)")
            
            while True:
                response = input("Choose an option (C/D/Y/N): ").strip().lower()
                if response in ['c', 'continue', 'y', 'yes']:
                    return 'continue'
                elif response in ['d', 'delete']:
                    return 'delete'
                elif response in ['n', 'no', '']:
                    return 'abort'
                else:
                    print("Invalid option. Please choose C, D, Y, or N.")
        elif default_action == 'continue':
            print("⏩ Auto-continuing (configured in YAML)")
            return 'continue'
        elif default_action == 'delete':
            print("🗑️ Auto-deleting (configured in YAML)")
            return 'delete'
        elif default_action == 'abort':
            print("❌ Auto-aborting (configured in YAML)")
            return 'abort'
        else:
            print(f"⚠️ Unknown default action: {default_action}, prompting user")
            return self._handle_existing_experiment(existing, source_model, target_model, dataset)
    
    def _cleanup_experiment_outputs(self, source_model: str, target_model: str, dataset: str):
        """清理实验输出文件"""
        import os
        import shutil
        import glob
        from pathlib import Path
        
        cleanup_targets = self.config.get('experiment_management.cleanup_targets', [])
        preserve_patterns = self.config.get('experiment_management.preserve_patterns', [])
        
        # 提取模型名称（去除路径）
        source_name = Path(source_model).name
        target_name = Path(target_model).name
        
        cleaned_count = 0
        
        for target in cleanup_targets:
            try:
                if target == 'training_outputs':
                    # 清理训练结果
                    pattern_paths = [
                        f"./train_lora/runs/{dataset}/{source_name}/*",
                        f"./train_lora/runs/{dataset}/{target_name}/*"
                    ]
                    for pattern in pattern_paths:
                        for path in glob.glob(pattern):
                            if self._should_preserve(path, preserve_patterns):
                                continue
                            if os.path.isdir(path):
                                shutil.rmtree(path)
                                print(f"  🗑️ Removed training output: {path}")
                                cleaned_count += 1
                
                elif target == 'transferred_lora':
                    # 清理迁移的LoRA
                    pattern = f"../autodl-tmp/transferred_lora/{dataset}/{source_name}_to_{target_name}/*"
                    for path in glob.glob(pattern):
                        if self._should_preserve(path, preserve_patterns):
                            continue
                        if os.path.isdir(path):
                            shutil.rmtree(path)
                            print(f"  🗑️ Removed transferred LoRA: {path}")
                            cleaned_count += 1
                
                elif target == 'evaluation_results':
                    # 清理评估结果
                    pattern_paths = [
                        f"./eval/results/*{source_name}*",
                        f"./eval/results/*{target_name}*",
                        f"./eval/results/*{dataset}*"
                    ]
                    for pattern in pattern_paths:
                        for path in glob.glob(pattern):
                            if self._should_preserve(path, preserve_patterns):
                                continue
                            if os.path.isfile(path):
                                os.remove(path)
                                print(f"  🗑️ Removed evaluation result: {path}")
                                cleaned_count += 1
                
                elif target == 'pipeline_results':
                    # 清理Pipeline结果，但保留总体结果文件的结构
                    results_dir = Path("./results")
                    if results_dir.exists():
                        # 删除特定实验的备份文件
                        backup_pattern = f"backup_*.json"
                        for backup_file in results_dir.glob(backup_pattern):
                            if self._should_preserve(str(backup_file), preserve_patterns):
                                continue
                            backup_file.unlink()
                            print(f"  🗑️ Removed backup: {backup_file}")
                            cleaned_count += 1
                        
                        # 清理CSV中的相关条目（这个比较复杂，暂时跳过自动清理）
                        print(f"  ℹ️ Note: CSV entries for this experiment may still exist in experiment_results.csv")
                        
            except Exception as e:
                print(f"  ⚠️ Error cleaning {target}: {e}")
        
        print(f"  ✅ Cleaned {cleaned_count} items")
    
    def _should_preserve(self, path: str, preserve_patterns: list) -> bool:
        """检查文件是否应该被保留"""
        import fnmatch
        path_name = os.path.basename(path)
        for pattern in preserve_patterns:
            if fnmatch.fnmatch(path_name, pattern) or fnmatch.fnmatch(path, pattern):
                return True
        return False
    
    def _init_results_dict(self, source_model: str, target_model: str, dataset: str) -> Dict[str, Any]:
        """初始化结果字典"""
        return {
            'experiment_id': self.experiment_id,
            'source_model': source_model,
            'target_model': target_model,
            'dataset': dataset,
            'timestamp': self.timestamp,
            'training_config': f"batch_size={self.config.get('training.default_batch_size')}, "
                              f"max_steps={self.config.get('training.default_max_steps')}, "
                              f"lr={self.config.get('training.default_lr')}",
            'notes': '自动化管道生成'
        }
    
    def _execute_pipeline_steps(self, results: Dict[str, Any], source_model: str, 
                               target_model: str, dataset: str, eval_only: bool) -> bool:
        """执行管道步骤"""
        # 总是显示完整的6步进度条
        progress_bar = tqdm(total=6, desc="Pipeline Progress", position=1, leave=True, ncols=80)
        
        try:
            # 步骤1: 评估源基础模型
            if not self._step_eval_source_base(results, source_model, dataset, progress_bar):
                print("⚠️ 源基础模型评估失败，但继续执行")
            
            # 步骤2: 训练源LoRA
            if not eval_only:
                if not self._step_train_source_lora(results, source_model, dataset, progress_bar):
                    raise Exception("源模型训练失败")
            else:
                self._step_skip_with_reason("STEP 2/6: TRAIN SOURCE LORA", "仅评估模式，跳过训练", progress_bar)
            
            # 步骤3: 迁移LoRA
            if not eval_only:
                if not self._step_transfer_lora(results, source_model, target_model, dataset, progress_bar):
                    raise Exception("LoRA迁移失败")
            else:
                self._step_skip_with_reason("STEP 3/6: TRANSFER LORA", "仅评估模式，跳过迁移", progress_bar)
            
            # 步骤4: 评估目标基础模型
            if not self._step_eval_target_base(results, target_model, dataset, progress_bar):
                print("⚠️ 目标基础模型评估失败，但继续执行")
            
            # 步骤5: 评估迁移LoRA
            if not eval_only:
                if not self._step_eval_transferred_lora(results, target_model, dataset, progress_bar):
                    print("⚠️ 迁移LoRA评估失败，但继续执行")
            else:
                self._step_skip_with_reason("STEP 5/6: EVAL TRANSFERRED LORA", "仅评估模式，无迁移LoRA可评估", progress_bar)
            
            # 步骤6: 训练目标LoRA
            if not eval_only:
                if not self._step_train_target_lora(results, target_model, dataset, progress_bar):
                    print("⚠️ 目标模型训练失败，但继续执行")
            else:
                self._step_skip_with_reason("STEP 6/6: TRAIN TARGET LORA", "仅评估模式，跳过训练", progress_bar)
            
            # 最终保存完整结果
            progress_bar.set_description("Saving Results")
            self.results.save_results(results)
            progress_bar.close()
            
            # 打印总结
            self._print_summary(results)
            
            # 提示可选命令
            if not eval_only:
                self._print_optional_commands(source_model, target_model, dataset)
            
            print("\nPipeline completed successfully!")
            return True
            
        except Exception as e:
            progress_bar.close()
            print(f"\nPipeline failed: {e}")
            # 保存部分结果
            self.results.save_partial_results(results, f"失败: {e}")
            return False
    
    def _step_skip_with_reason(self, step_title: str, reason: str, progress_bar: tqdm):
        """显示跳过的步骤及原因"""
        print(f"\n{'='*60}")
        print(step_title)
        print("="*60)
        print(f"🚫 跳过原因: {reason}")
        print("="*60)
        progress_bar.update(1)
    
    def _step_train_source_lora(self, results: Dict[str, Any], source_model: str, 
                               dataset: str, progress_bar: tqdm) -> bool:
        """步骤2: 训练源LoRA"""
        print(f"\n{'='*60}")
        print("STEP 2/6: TRAIN SOURCE LORA")
        print("="*60)
        
        source_lora_path, source_lora_acc, status_msg = self.trainer.train_model(source_model, dataset)
        print(f"状态: {status_msg}")
        print(f"🔍 DEBUG: 训练器返回的准确率: {source_lora_acc}")
        if source_lora_path is None:
            return False
        
        results.update({
            'source_lora_path': source_lora_path,
            'source_lora_acc': source_lora_acc,
        })
        self.results.save_partial_results(results, "源LoRA训练完成")
        progress_bar.update(1)
        return True
    
    def _step_transfer_lora(self, results: Dict[str, Any], source_model: str,
                           target_model: str, dataset: str, progress_bar: tqdm) -> bool:
        """步骤3: 迁移LoRA"""
        print(f"\n{'='*60}")
        print("STEP 3/6: TRANSFER LORA")
        print("="*60)
        
        transferred_lora_path = self.transfer.transfer_lora(
            results['source_lora_path'], source_model, target_model, dataset
        )
        if transferred_lora_path is None:
            return False
        
        results['transferred_lora_path'] = transferred_lora_path
        self.results.save_partial_results(results, "LoRA迁移完成")
        progress_bar.update(1)
        return True
    
    def _step_eval_target_base(self, results: Dict[str, Any], target_model: str, 
                              dataset: str, progress_bar: tqdm) -> bool:
        """步骤4: 评估目标基础模型"""
        print(f"\n{'='*60}")
        print("STEP 4/6: EVAL TARGET BASE MODEL")
        print("="*60)
        
        target_acc = self.evaluator.evaluate_base_model(target_model, dataset)
        results['target_acc'] = target_acc
        self.results.save_partial_results(results, "目标基础模型评估完成")
        progress_bar.update(1)
        return target_acc is not None
    
    def _step_eval_transferred_lora(self, results: Dict[str, Any], target_model: str, 
                                   dataset: str, progress_bar: tqdm) -> bool:
        """步骤5: 评估迁移LoRA"""
        print(f"\n{'='*60}")
        print("STEP 5/6: EVAL TRANSFERRED LORA")
        print("="*60)
        
        transferred_acc = self.evaluator.evaluate_lora_model(
            results['transferred_lora_path'], target_model, dataset
        )
        results['transferred_acc'] = transferred_acc
        self.results.save_partial_results(results, "迁移LoRA评估完成")
        progress_bar.update(1)
        return transferred_acc is not None
    
    def _step_train_target_lora(self, results: Dict[str, Any], target_model: str, 
                               dataset: str, progress_bar: tqdm) -> bool:
        """步骤6: 训练目标LoRA"""
        print(f"\n{'='*60}")
        print("STEP 6/6: TRAIN TARGET LORA")
        print("="*60)
        
        target_lora_path, target_lora_acc, status_msg = self.trainer.train_model(target_model, dataset)
        print(f"状态: {status_msg}")
        if target_lora_path is None:
            target_lora_acc = None
        
        results.update({
            'target_lora_path': target_lora_path,
            'target_lora_acc': target_lora_acc,
        })
        self.results.save_partial_results(results, "目标LoRA训练完成")
        progress_bar.update(1)
        return True  # 即使失败也继续
    
    def _step_eval_source_base(self, results: Dict[str, Any], source_model: str, 
                              dataset: str, progress_bar: tqdm) -> bool:
        """步骤1: 评估源基础模型"""
        print(f"\n{'='*60}")
        print("STEP 1/6: EVAL SOURCE BASE MODEL")
        print("="*60)
        
        source_acc = self.evaluator.evaluate_base_model(source_model, dataset)
        results['source_acc'] = source_acc
        self.results.save_partial_results(results, "源基础模型评估完成")
        progress_bar.update(1)
        return source_acc is not None
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print experiment summary"""
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        
        source_name = ModelUtils.get_model_short_name(results['source_model'])
        target_name = ModelUtils.get_model_short_name(results['target_model'])
        
        # Handle potentially None values
        source_acc = results.get('source_acc', 0) or 0
        target_acc = results.get('target_acc', 0) or 0
        source_lora_acc = results.get('source_lora_acc')
        target_lora_acc = results.get('target_lora_acc')
        transferred_acc = results.get('transferred_acc')
        
        print(f"Source Model ({source_name}):     {source_acc:.4f}")
        if source_lora_acc is not None:
            improvement = (source_lora_acc - source_acc) * 100
            sign = "+" if improvement >= 0 else ""
            print(f"Source + LoRA:              {source_lora_acc:.4f} ({sign}{improvement:.2f}%)")
        
        print(f"Target Model ({target_name}):     {target_acc:.4f}")
        
        if transferred_acc is not None:
            improvement = (transferred_acc - target_acc) * 100
            sign = "+" if improvement >= 0 else ""
            print(f"Target + Transferred LoRA:  {transferred_acc:.4f} ({sign}{improvement:.2f}%)")
        
        if target_lora_acc is not None:
            improvement = (target_lora_acc - target_acc) * 100
            sign = "+" if improvement >= 0 else ""
            print(f"Target + Direct LoRA:       {target_lora_acc:.4f} ({sign}{improvement:.2f}%)")
        
        print("=" * 60)
        print(f"Detailed results: results/experiment_summary.md")
    
    def _print_optional_commands(self, source_model: str, target_model: str, dataset: str):
        """Print optional target model LoRA training commands"""
        target_name = ModelUtils.get_model_short_name(target_model)
        
        print(f"\n{'-'*60}")
        print(f"OPTIONAL: Train {target_name} LoRA for Comparison")
        print("-" * 60)
        
        # Training command
        train_cmd = f"python {self.config.get('paths.train_script')} " \
                   f"--dataset {dataset} " \
                   f"--base_model {target_model} " \
                   f"--bs {self.config.get('training.default_batch_size')} " \
                   f"--max_steps {self.config.get('training.default_max_steps')}"
        
        print(f"Train {target_name} LoRA:")
        print(f"  {train_cmd}")
        
        # Evaluation command 
        eval_cmd = f"python {self.config.get('paths.eval_script')} " \
                  f"--models_list [trained_model_path] " \
                  f"--dataset {dataset} " \
                  f"--sample_ratio {self.config.get('evaluation.sample_ratio')} " \
                  f"--base_model {target_model}"
        
        print(f"\nEvaluate {target_name} LoRA:")
        print(f"  {eval_cmd}")
