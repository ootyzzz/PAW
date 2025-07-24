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
        """运行完整管道 - 新流程：训练源LoRA → 迁移 → 评估目标基础 → 评估迁移LoRA → 训练目标LoRA → 评估源基础
        
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
        print("="*60)
        
        # 检查是否有历史记录
        if not eval_only:
            existing = self.results.check_existing_experiment(source_model, target_model, dataset)
            if existing is not None:
                print(f"\nWarning: Found existing experiment (timestamp: {existing['timestamp']})")
                response = input("Continue anyway? (y/N): ").strip().lower()
                if response not in ['y', 'yes']:
                    print("Experiment cancelled.")
                    return False
        
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
        # 设置进度条
        total_steps = 6 if not eval_only else 4
        progress_bar = tqdm(total=total_steps, desc="Pipeline Progress", position=0, leave=True)
        
        try:
            if not eval_only:
                # 步骤1: 训练源LoRA
                if not self._step_train_source_lora(results, source_model, dataset, progress_bar):
                    raise Exception("源模型训练失败")
                
                # 步骤2: 迁移LoRA
                if not self._step_transfer_lora(results, source_model, target_model, dataset, progress_bar):
                    raise Exception("LoRA迁移失败")
            
            # 步骤3: 评估目标基础模型
            if not self._step_eval_target_base(results, target_model, dataset, progress_bar):
                print("⚠️ 目标基础模型评估失败，但继续执行")
            
            if not eval_only:
                # 步骤4: 评估迁移LoRA
                if not self._step_eval_transferred_lora(results, target_model, dataset, progress_bar):
                    print("⚠️ 迁移LoRA评估失败，但继续执行")
                
                # 步骤5: 训练目标LoRA
                if not self._step_train_target_lora(results, target_model, dataset, progress_bar):
                    print("⚠️ 目标模型训练失败，但继续执行")
            
            # 步骤6: 评估源基础模型
            if not self._step_eval_source_base(results, source_model, dataset, progress_bar):
                print("⚠️ 源基础模型评估失败，但继续执行")
            
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
    
    def _step_train_source_lora(self, results: Dict[str, Any], source_model: str, 
                               dataset: str, progress_bar: tqdm) -> bool:
        """步骤1: 训练源LoRA"""
        print(f"\n{'='*60}")
        print("STEP 1/6: TRAIN SOURCE LORA")
        print("="*60)
        
        source_lora_path, source_lora_acc = self.trainer.train_model(source_model, dataset)
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
        """步骤2: 迁移LoRA"""
        print(f"\n{'='*60}")
        print("STEP 2/6: TRANSFER LORA")
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
        """步骤3: 评估目标基础模型"""
        print(f"\n{'='*60}")
        print("STEP 3/6: EVAL TARGET BASE MODEL")
        print("="*60)
        
        target_acc = self.evaluator.evaluate_base_model(target_model, dataset)
        results['target_acc'] = target_acc
        self.results.save_partial_results(results, "目标基础模型评估完成")
        progress_bar.update(1)
        return target_acc is not None
    
    def _step_eval_transferred_lora(self, results: Dict[str, Any], target_model: str, 
                                   dataset: str, progress_bar: tqdm) -> bool:
        """步骤4: 评估迁移LoRA"""
        print(f"\n{'='*60}")
        print("STEP 4/6: EVAL TRANSFERRED LORA")
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
        """步骤5: 训练目标LoRA"""
        print(f"\n{'='*60}")
        print("STEP 5/6: TRAIN TARGET LORA")
        print("="*60)
        
        target_lora_path, target_lora_acc = self.trainer.train_model(target_model, dataset)
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
        """步骤6: 评估源基础模型"""
        print(f"\n{'='*60}")
        print("STEP 6/6: EVAL SOURCE BASE MODEL")
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
            print(f"Source + LoRA:              {source_lora_acc:.4f} (+{improvement:.2f}%)")
        
        print(f"Target Model ({target_name}):     {target_acc:.4f}")
        
        if transferred_acc is not None:
            improvement = (transferred_acc - target_acc) * 100
            print(f"Target + Transferred LoRA:  {transferred_acc:.4f} (+{improvement:.2f}%)")
        
        if target_lora_acc is not None:
            improvement = (target_lora_acc - target_acc) * 100
            print(f"Target + Direct LoRA:       {target_lora_acc:.4f} (+{improvement:.2f}%)")
        
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
        print()
        print("NOTE: After training, you can compare 'Target+LoRA' vs 'Target+Transferred LoRA' performance")
