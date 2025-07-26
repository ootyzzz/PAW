"""
模型训练模块
负责LoRA训练的调度和管理
"""

import os
import yaml
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
        existing_path = self._check_existing_training(model_path, dataset)
        if existing_path:
            if self.verbose:
                print(f"状态: 发现已有训练结果: {existing_path}")
            # 尝试从已有结果中读取准确率
            existing_accuracy = self._read_accuracy_from_existing(existing_path)
            return existing_path, existing_accuracy, f"发现已有训练结果: {existing_path}"
        
        # 构建训练命令
        cmd = self._build_train_command(model_path, dataset)
        
        # 执行训练
        output = self.runner.run_command(
            cmd, 
            f"训练 {model_name} LoRA",
            cwd="."  # 在PAW根目录执行
        )
        
        if output is None:
            return None, None, "训练失败：命令执行错误"
        
        # 解析输出获取准确率
        accuracy = OutputParser.parse_training_accuracy(output)
        
        # 查找生成的模型路径 - 使用模型的短名称
        model_short_name = os.path.basename(model_path.rstrip('/'))
        final_model_path = self._find_latest_model(model_short_name, dataset)
        
        if final_model_path:
            return final_model_path, accuracy, f"训练完成，模型保存至: {final_model_path}"
        else:
            return None, accuracy, "训练执行完成，但未找到输出模型"
    
    def _check_existing_training(self, model_path: str, dataset: str) -> Optional[str]:
        """检查是否已有相同配置的训练结果
        
        目前只比对batch_size和max_steps - 可扩展到更多参数
        """
        model_short_name = os.path.basename(model_path.rstrip('/'))
        
        # 检查新格式路径: runs/{dataset}/{model_name}/
        new_format_dir = os.path.join("runs", dataset, model_short_name)
        # 检查旧格式路径: train_lora/runs/{dataset}/{model_name}/
        old_format_dir = os.path.join(self.config.get('paths.runs_dir'), dataset, model_short_name)
        
        # 获取当前配置
        current_batch_size = self.config.get('training.default_batch_size')
        current_max_steps = self.config.get('training.default_max_steps')
        
        # 优先检查新格式，然后检查旧格式
        for runs_dir in [new_format_dir, old_format_dir]:
            if not os.path.exists(runs_dir):
                continue
            
            # 查找所有训练结果目录
            existing_runs = [d for d in os.listdir(runs_dir) 
                            if os.path.isdir(os.path.join(runs_dir, d))]
            
            if not existing_runs:
                continue
            
            # 检查每个训练结果，比对配置
            for run_dir in sorted(existing_runs, reverse=True):  # 从最新开始
                run_path = os.path.join(runs_dir, run_dir)
                final_model_path = os.path.join(run_path, "final_model")
                
                # 检查final_model是否存在
                if not os.path.exists(final_model_path):
                    continue
                    
                # 查找配置文件进行比对
                config_files = [
                    os.path.join(run_path, "hparams.yaml"),      # Lightning默认参数文件
                    os.path.join(run_path, "trainer_state.json"), # Transformers训练状态
                    os.path.join(run_path, "training_args.json"), # 训练参数
                    os.path.join(run_path, "config.json"),       # 通用配置
                ]
                
                for config_file in config_files:
                    if os.path.exists(config_file):
                        try:
                            if self._config_matches(config_file, current_batch_size, current_max_steps):
                                if self.verbose:
                                    print(f"   发现匹配配置的训练结果: {final_model_path}")
                                    print(f"   配置文件: {os.path.basename(config_file)}")
                                return final_model_path
                        except Exception as e:
                            if self.verbose:
                                print(f"   配置文件读取失败 {config_file}: {e}")
                            continue
        
        return None
    
    def _config_matches(self, config_file: str, target_batch_size: int, target_max_steps: int) -> bool:
        """检查配置文件是否匹配当前训练配置
        
        目前只比对batch_size和max_steps
        """
        import json
        import yaml
        
        try:
            # 根据文件扩展名选择解析方式
            if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            else:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            
            # 查找batch_size的各种可能键名
            batch_size_keys = ['batch_size', 'per_device_train_batch_size', 'train_batch_size', 'bs']
            found_batch_size = None
            
            for key in batch_size_keys:
                if key in config:
                    found_batch_size = config[key]
                    break
            
            # 查找max_steps的各种可能键名
            max_steps_keys = ['max_steps', 'total_steps', 'training_steps']
            found_max_steps = None
            
            for key in max_steps_keys:
                if key in config:
                    found_max_steps = config[key]
                    break
            
            # 比对配置
            batch_match = (found_batch_size is None or found_batch_size == target_batch_size)
            steps_match = (found_max_steps is None or found_max_steps == target_max_steps)
            
            if self.verbose and (found_batch_size is not None or found_max_steps is not None):
                print(f"     配置比对: batch_size {found_batch_size} vs {target_batch_size}, "
                      f"max_steps {found_max_steps} vs {target_max_steps}")
            
            return batch_match and steps_match
            
        except Exception as e:
            if self.verbose:
                print(f"     配置解析失败: {e}")
            return False
    
    def _read_accuracy_from_existing(self, model_path: str) -> Optional[float]:
        """从已有训练结果中读取准确率"""
        import json
        import os
        
        # 查找可能包含准确率的文件
        result_files = [
            os.path.join(os.path.dirname(model_path), "trainer_state.json"),
            os.path.join(os.path.dirname(model_path), "training_results.json"),
            os.path.join(os.path.dirname(model_path), "eval_results.json"),
        ]
        
        for result_file in result_files:
            if os.path.exists(result_file):
                try:
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                    
                    # 查找准确率的各种可能键名
                    accuracy_keys = [
                        'test/accuracy', 'test_accuracy', 'eval_accuracy', 
                        'accuracy', 'final_accuracy', 'best_accuracy'
                    ]
                    
                    for key in accuracy_keys:
                        if key in data:
                            accuracy = data[key]
                            if isinstance(accuracy, (int, float)) and 0 <= accuracy <= 1:
                                return float(accuracy)
                            
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
        
        # 如果没有找到，返回None，但至少我们尝试过了
        return None
    
    def _build_train_command(self, model_path: str, dataset: str) -> str:
        """构建训练命令"""
        train_script = self.config.get('paths.train_script')
        
        cmd = f"TQDM_DISABLE=1 python {train_script} " \
              f"--dataset {dataset} " \
              f"--base_model {model_path} " \
              f"--bs {self.config.get('training.default_batch_size')} " \
              f"--max_steps {self.config.get('training.default_max_steps')}"
        
        return cmd
    
    def _find_latest_model(self, model_name: str, dataset: str) -> Optional[str]:
        """查找最新生成的模型路径"""
        # 检查新格式路径: runs/{dataset}/{model_name}/
        new_format_dir = os.path.join("runs", dataset, model_name)
        # 检查旧格式路径: train_lora/runs/{dataset}/{model_name}/
        old_format_dir = os.path.join(self.config.get('paths.runs_dir'), dataset, model_name)
        
        # 优先检查新格式
        for runs_dir in [new_format_dir, old_format_dir]:
            if not os.path.exists(runs_dir):
                continue
            
            runs = [d for d in os.listdir(runs_dir) 
                   if os.path.isdir(os.path.join(runs_dir, d))]
            
            if not runs:
                continue
            
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
