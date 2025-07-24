"""
模型评估模块
负责模型性能评估
"""

import os
from typing import Optional
from .config import PipelineConfig
from .utils import ModelUtils, CommandRunner, OutputParser


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, config: PipelineConfig, verbose: bool = True):
        self.config = config
        self.runner = CommandRunner(verbose=verbose)
        self.verbose = verbose
    
    def evaluate_model(self, model_path: str, base_model: str, dataset: str, is_lora: bool = True) -> Optional[float]:
        """评估模型性能
        
        Args:
            model_path: 模型路径 (LoRA路径或基础模型路径)
            base_model: 基础模型路径 (LoRA评估时需要)
            dataset: 数据集名称
            is_lora: 是否是LoRA模型
            
        Returns:
            准确率 或 None 如果失败
        """
        model_type = "LoRA" if is_lora else "基础模型"
        model_name = ModelUtils.get_model_short_name(base_model)
        
        if self.verbose:
            print(f"\n📊 开始评估 {model_name} {model_type}")
        
        # 构建评估命令
        cmd = self._build_eval_command(model_path, base_model, dataset, is_lora)
        
        # 执行评估
        output = self.runner.run_command(
            cmd,
            f"评估 {model_name} {model_type}"
        )
        
        if output is None:
            return None
        
        # 解析评估输出获取准确率
        accuracy = OutputParser.parse_evaluation_accuracy(output)
        return accuracy
    
    def _build_eval_command(self, model_path: str, base_model: str, dataset: str, is_lora: bool) -> str:
        """构建评估命令"""
        eval_script = self.config.get('paths.eval_script')
        sample_ratio = self.config.get('evaluation.sample_ratio')
        
        if is_lora:
            # 评估LoRA模型
            cmd = f"python {eval_script} " \
                  f"--models_list {model_path} " \
                  f"--dataset {dataset} " \
                  f"--sample_ratio {sample_ratio} " \
                  f"--base_model {base_model}"
        else:
            # 评估基础模型
            cmd = f"python {eval_script} " \
                  f"--models_list {base_model} " \
                  f"--dataset {dataset} " \
                  f"--sample_ratio {sample_ratio}"
        
        return cmd
    
    def evaluate_base_model(self, model_path: str, dataset: str) -> Optional[float]:
        """评估基础模型的便捷方法"""
        return self.evaluate_model(model_path, model_path, dataset, is_lora=False)
    
    def evaluate_lora_model(self, lora_path: str, base_model: str, dataset: str) -> Optional[float]:
        """评估LoRA模型的便捷方法"""
        return self.evaluate_model(lora_path, base_model, dataset, is_lora=True)
