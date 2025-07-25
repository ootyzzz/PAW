"""
模型训练模块
负责LoRA训练的调度和管理
"""

import os
from typing import Optional, Tuple
from .config import PipelineConfig
from .utils import ModelUtils, CommandRunner, OutputParser


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, config: PipelineConfig, verbose: bool = True):
        self.config = config
        self.runner = CommandRunner(verbose=verbose)
        self.verbose = verbose
    
    def train_model(self, model_path: str, dataset: str) -> Tuple[Optional[str], Optional[float], str]:
        """训练模型+LoRA
        
        Args:
            model_path: 模型路径
            dataset: 数据集名称
            
        Returns:
            Tuple[模型路径, 准确率, 状态消息]
        """
        model_name = ModelUtils.get_model_short_name(model_path)
        
        if self.verbose:
            print(f"\nTraining {model_name} + LoRA (dataset: {dataset})")
        
        # Check for existing training results
        existing_path = self._check_existing_training(model_name, dataset)
        if existing_path:
            if self.verbose:
                print(f"Found existing training results: {existing_path}")
            return existing_path, None, f"发现已有训练结果: {existing_path}"  # Return path and empty accuracy (needs evaluation)
        
        # 构建训练命令
        cmd = self._build_train_command(model_path, dataset)
        
        # 执行训练
        output = self.runner.run_command(
            cmd, 
            f"训练 {model_name} LoRA",
            cwd=os.path.dirname(self.config.get('paths.train_script'))
        )
        
        if output is None:
            return None, None, "训练失败：命令执行错误"
        
        # 解析输出获取准确率
        accuracy = OutputParser.parse_training_accuracy(output)
        
        # 查找生成的模型路径
        final_model_path = self._find_latest_model(model_name, dataset)
        
        if final_model_path:
            return final_model_path, accuracy, f"训练完成，模型保存至: {final_model_path}"
        else:
            return None, accuracy, "训练执行完成，但未找到输出模型"
    
    def _check_existing_training(self, model_name: str, dataset: str) -> Optional[str]:
        """检查是否已有训练结果"""
        runs_dir = os.path.join(self.config.get('paths.runs_dir'), dataset, model_name)
        
        if not os.path.exists(runs_dir):
            return None
        
        # 查找最新的训练结果
        existing_runs = [d for d in os.listdir(runs_dir) 
                        if os.path.isdir(os.path.join(runs_dir, d))]
        
        if not existing_runs:
            return None
        
        latest_run = sorted(existing_runs)[-1]
        final_model_path = os.path.join(runs_dir, latest_run, "final_model")
        
        if os.path.exists(final_model_path):
            return final_model_path
        
        return None
    
    def _build_train_command(self, model_path: str, dataset: str) -> str:
        """构建训练命令"""
        train_script = os.path.basename(self.config.get('paths.train_script'))
        
        cmd = f"TQDM_DISABLE=1 python {train_script} " \
              f"--dataset {dataset} " \
              f"--base_model {model_path} " \
              f"--bs {self.config.get('training.default_batch_size')} " \
              f"--max_steps {self.config.get('training.default_max_steps')}"
        
        return cmd
    
    def _find_latest_model(self, model_name: str, dataset: str) -> Optional[str]:
        """查找最新生成的模型路径"""
        runs_dir = os.path.join(self.config.get('paths.runs_dir'), dataset, model_name)
        
        if not os.path.exists(runs_dir):
            return None
        
        runs = [d for d in os.listdir(runs_dir) 
               if os.path.isdir(os.path.join(runs_dir, d))]
        
        if not runs:
            return None
        
        latest_run = sorted(runs)[-1]
        final_model_path = os.path.join(runs_dir, latest_run, "final_model")
        
        if os.path.exists(final_model_path):
            return final_model_path
        
        return None
    
    def check_step_completed(self, model_path: str, dataset: str) -> Tuple[bool, Optional[str]]:
        """检查训练步骤是否已完成
        
        Returns:
            Tuple[是否完成, 模型路径]
        """
        model_name = ModelUtils.get_model_short_name(model_path)
        existing_path = self._check_existing_training(model_name, dataset)
        
        if existing_path:
            return True, existing_path
        
        return False, None
