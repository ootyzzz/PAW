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
        
        print(f"\n🎯 开始LoRA迁移实验")
        print(f"📋 实验ID: {self.experiment_id}")
        print(f"🎲 源模型: {source_model}")
        print(f"🎯 目标模型: {target_model}")
        print(f"📚 数据集: {dataset}")
        print("=" * 80)
        
        # 检查是否有历史记录
        if not eval_only:
            existing = self.results.check_existing_experiment(source_model, target_model, dataset)
            if existing is not None:
                print(f"⚠️ 发现相同实验记录 (时间: {existing['timestamp']})")
                response = input("是否继续? (y/N): ").strip().lower()
                if response not in ['y', 'yes']:
                    print("🚫 实验取消")
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
        progress_bar = tqdm(total=total_steps, desc="🚀 LoRA迁移管道", position=0, leave=True)
        
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
            progress_bar.set_description("💾 保存最终结果")
            self.results.save_results(results)
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
            self.results.save_partial_results(results, f"失败: {e}")
            return False
    
    def _step_train_source_lora(self, results: Dict[str, Any], source_model: str, 
                               dataset: str, progress_bar: tqdm) -> bool:
        """步骤1: 训练源LoRA"""
        progress_bar.set_description("🎯 步骤1: 训练源LoRA")
        
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
        progress_bar.set_description("🔄 步骤2: 迁移LoRA")
        
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
        progress_bar.set_description("📊 步骤3: 评估目标基础模型")
        
        target_acc = self.evaluator.evaluate_base_model(target_model, dataset)
        results['target_acc'] = target_acc
        self.results.save_partial_results(results, "目标基础模型评估完成")
        progress_bar.update(1)
        return target_acc is not None
    
    def _step_eval_transferred_lora(self, results: Dict[str, Any], target_model: str, 
                                   dataset: str, progress_bar: tqdm) -> bool:
        """步骤4: 评估迁移LoRA"""
        progress_bar.set_description("📊 步骤4: 评估迁移LoRA")
        
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
        progress_bar.set_description("🎯 步骤5: 训练目标LoRA")
        
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
        progress_bar.set_description("📊 步骤6: 评估源基础模型")
        
        source_acc = self.evaluator.evaluate_base_model(source_model, dataset)
        results['source_acc'] = source_acc
        self.results.save_partial_results(results, "源基础模型评估完成")
        progress_bar.update(1)
        return source_acc is not None
    
    def _print_summary(self, results: Dict[str, Any]):
        """打印实验总结"""
        print(f"\n🎉 实验完成! 总结如下:")
        print("=" * 60)
        
        source_name = ModelUtils.get_model_short_name(results['source_model'])
        target_name = ModelUtils.get_model_short_name(results['target_model'])
        
        # 处理可能为None的值
        source_acc = results.get('source_acc', 0) or 0
        target_acc = results.get('target_acc', 0) or 0
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
    
    def _print_optional_commands(self, source_model: str, target_model: str, dataset: str):
        """打印可选的目标模型LoRA训练命令"""
        target_name = ModelUtils.get_model_short_name(target_model)
        
        print(f"\n💡 可选：训练目标模型 {target_name} 的LoRA进行对比")
        print("=" * 60)
        
        # 训练命令
        train_cmd = f"python {self.config.get('paths.train_script')} " \
                   f"--dataset {dataset} " \
                   f"--base_model {target_model} " \
                   f"--bs {self.config.get('training.default_batch_size')} " \
                   f"--max_steps {self.config.get('training.default_max_steps')}"
        
        print(f"🎯 训练 {target_name} LoRA:")
        print(f"   {train_cmd}")
        
        # 评估命令 
        eval_cmd = f"python {self.config.get('paths.eval_script')} " \
                  f"--models_list [训练后的模型路径] " \
                  f"--dataset {dataset} " \
                  f"--sample_ratio {self.config.get('evaluation.sample_ratio')} " \
                  f"--base_model {target_model}"
        
        print(f"\n📊 评估 {target_name} LoRA:")
        print(f"   {eval_cmd}")
        print()
        print("💡 训练完成后可以对比 '目标模型+LoRA' vs '目标模型+迁移LoRA' 的性能差异")
