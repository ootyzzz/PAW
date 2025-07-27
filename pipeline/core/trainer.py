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
        model_short_name = ModelUtils.get_model_short_name(model_path)
        final_model_path = self._find_latest_model(model_short_name, dataset)
        
        if final_model_path:
            return final_model_path, accuracy, f"训练完成，模型保存至: {final_model_path}"
        else:
            return None, accuracy, "训练执行完成，但未找到输出模型"
    
    def _check_existing_training(self, model_path: str, dataset: str) -> Optional[str]:
        """检查是否已有相同配置的训练结果
        
        目前只比对batch_size和max_steps - 可扩展到更多参数
        """
        model_short_name = ModelUtils.get_model_short_name(model_path)
        
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
                    os.path.join(run_path, "config.yaml"),       # PAW项目配置文件
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
            
            # 查找batch_size的各种可能键名和路径
            found_batch_size = None
            batch_size_paths = [
                'batch_size', 'per_device_train_batch_size', 'train_batch_size', 'bs',
                'training.batch_size', 'experiment.batch_size'
            ]
            
            for path in batch_size_paths:
                value = self._get_nested_value(config, path)
                if value is not None:
                    found_batch_size = value
                    break
            
            # 查找max_steps的各种可能键名和路径
            found_max_steps = None
            max_steps_paths = [
                'max_steps', 'total_steps', 'training_steps',
                'training.max_steps', 'experiment.max_steps'
            ]
            
            for path in max_steps_paths:
                value = self._get_nested_value(config, path)
                if value is not None:
                    found_max_steps = value
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
        import yaml
        
        # 1. 首先查找标准的结果文件
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
                                if self.verbose:
                                    print(f"   从{os.path.basename(result_file)}读取准确率: {accuracy:.4f}")
                                return float(accuracy)
                            
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
        
        # 2. 查找tensorboard的hparams.yaml文件
        hparams_file = os.path.join(os.path.dirname(model_path), "tensorboard_logs", "hparams.yaml")
        if os.path.exists(hparams_file):
            try:
                with open(hparams_file, 'r') as f:
                    data = yaml.safe_load(f)
                # hparams通常不包含准确率，但我们检查一下
                if 'accuracy' in data:
                    accuracy = data['accuracy']
                    if isinstance(accuracy, (int, float)) and 0 <= accuracy <= 1:
                        if self.verbose:
                            print(f"   从hparams.yaml读取准确率: {accuracy:.4f}")
                        return float(accuracy)
            except Exception:
                pass
        
        # 3. 尝试从swanlab元数据中读取（虽然通常不包含准确率）
        swanlab_dir = os.path.join(os.path.dirname(model_path), "swanlab_logs")
        if os.path.exists(swanlab_dir):
            for run_dir in os.listdir(swanlab_dir):
                metadata_file = os.path.join(swanlab_dir, run_dir, "files", "swanlab-metadata.json")
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r') as f:
                            data = json.load(f)
                        # 检查是否有准确率信息
                        if 'accuracy' in data:
                            accuracy = data['accuracy']
                            if isinstance(accuracy, (int, float)) and 0 <= accuracy <= 1:
                                if self.verbose:
                                    print(f"   从swanlab元数据读取准确率: {accuracy:.4f}")
                                return float(accuracy)
                    except Exception:
                        continue
        
        # 4. 如果都没找到，尝试运行一个快速评估来获取准确率
        if self.verbose:
            print(f"   未找到已保存的准确率，尝试快速评估...")
        
        return self._quick_evaluate_model(model_path)
    
    def _quick_evaluate_model(self, model_path: str) -> Optional[float]:
        """对已有模型进行快速评估以获取准确率"""
        try:
            # 从模型路径推断数据集和基础模型
            path_parts = model_path.split(os.sep)
            dataset = None
            model_name = None
            
            # 解析路径：runs/arc-challenge/gemma-2-2b-it/211804/final_model
            for i, part in enumerate(path_parts):
                if part == "runs" and i + 2 < len(path_parts):
                    dataset = path_parts[i + 1]
                    model_name = path_parts[i + 2]
                    break
            
            if not dataset or not model_name:
                if self.verbose:
                    print(f"   无法从路径解析数据集和模型名: {model_path}")
                return None
            
            # 构建基础模型路径
            base_model_path = os.path.join(self.config.get('paths.models_dir'), model_name)
            
            # 构建评估命令
            eval_script = self.config.get('paths.eval_script')
            sample_ratio = self.config.get('evaluation.sample_ratio', 0.05)
            
            cmd = f"python {eval_script} " \
                  f"--models_list {model_path} " \
                  f"--dataset {dataset} " \
                  f"--sample_ratio {sample_ratio} " \
                  f"--base_model {base_model_path}"
            
            if self.verbose:
                print(f"   运行快速评估: {cmd}")
            
            # 运行评估
            output = self.runner.run_command(
                cmd,
                f"快速评估 {model_name}",
                cwd="."
            )
            
            if output:
                # 解析评估输出获取准确率
                accuracy = OutputParser.parse_evaluation_accuracy(output)
                if accuracy is not None:
                    if self.verbose:
                        print(f"   快速评估获得准确率: {accuracy:.4f}")
                    return accuracy
            
        except Exception as e:
            if self.verbose:
                print(f"   快速评估失败: {e}")
        
        # 如果快速评估也失败了，返回一个合理的默认值或None
        if self.verbose:
            print(f"   无法获取准确率，将使用默认值")
        return None  # 或者返回一个合理的默认值，比如 0.5
    
    def _build_train_command(self, model_path: str, dataset: str) -> str:
        """构建训练命令"""
        train_script = self.config.get('paths.train_script')
        
        # 基础命令
        cmd = f"TQDM_DISABLE=1 python {train_script} " \
              f"--dataset {dataset} " \
              f"--base_model {model_path} " \
              f"--bs {self.config.get('training.default_batch_size')} " \
              f"--max_steps {self.config.get('training.default_max_steps')}"
        
        # 如果配置中有LoRA设置，添加配置文件参数
        if self.config.has_lora_config():
            config_file = self.config.get_config_file_path()
            if config_file and os.path.exists(config_file):
                cmd += f" --config {config_file}"
                if self.verbose:
                    print(f"📝 使用LoRA配置文件: {config_file}")
                    lora_config = self.config.get('lora', {})
                    print(f"   - 目标层: {lora_config.get('target_modules', ['q_proj', 'v_proj'])}")
                    print(f"   - 秩 (r): {lora_config.get('r', 16)}")
                    print(f"   - Alpha: {lora_config.get('lora_alpha', 32)}")
        else:
            # 如果没有LoRA配置，使用默认的lightning配置文件
            default_config = "./train_lora/config/lightning_config.yaml"
            if os.path.exists(default_config):
                cmd += f" --config {default_config}"
                if self.verbose:
                    print(f"📝 使用默认LoRA配置文件: {default_config}")
        
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
            
            # 按文件修改时间排序，确保获取最新的训练结果
            def get_mtime(run_name):
                run_path = os.path.join(runs_dir, run_name)
                try:
                    return os.path.getmtime(run_path)
                except:
                    # 如果获取修改时间失败，尝试解析时间戳
                    if len(run_name) == 6 and run_name.isdigit():
                        # 6位数字格式：HHMMSS
                        return float(run_name)
                    elif len(run_name) >= 15 and '_' in run_name:
                        # 完整格式：YYYYMMDD_HHMMSS
                        timestamp_part = run_name.split('_')[-1]
                        if timestamp_part.isdigit():
                            return float(timestamp_part)
                    return 0
            
            latest_run = sorted(runs, key=get_mtime)[-1]
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
    
    def _get_nested_value(self, config: dict, path: str):
        """从嵌套字典中获取值，支持点分隔的路径"""
        keys = path.split('.')
        current = config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
