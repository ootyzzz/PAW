"""
配置管理模块
处理所有pipeline的配置加载和验证
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List


class PipelineConfig:
    """Pipeline配置管理器"""
    
    def __init__(self, config_path: str = None):
        """初始化配置"""
        if config_path is None:
            # 默认使用pipeline目录下的配置
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'pipeline_config.yaml')
        
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            # 如果配置文件不存在，创建默认配置
            self._create_default_config()
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _create_default_config(self):
        """创建默认配置文件"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        default_config = {
            'paths': {
                'models_dir': '/root/autodl-tmp/models',
                'train_script': '/root/PAW/train_lora/train_cs_lora_lightning.py',
                'eval_script': '/root/PAW/eval/lightning_eval.py',
                'transfer_script': '/root/PAW/lora_adapter/scripts/transfer_lora_x.py',
                'runs_dir': '/root/PAW/train_lora/runs',
                'transferred_lora_dir': '/root/autodl-tmp/transferred_lora',
                'results_dir': '/root/PAW/results'
            },
            'training': {
                'default_batch_size': 4,
                'default_max_steps': 20,
                'default_lr': '1e-5',
                'datasets': ['piqa', 'arc-challenge', 'arc-easy', 'hellaswag', 'winogrande']
            },
            'evaluation': {
                'sample_ratio': 0.05,
                'default_batch_size': 8
            },
            'transfer': {
                'similarity_threshold': 0.0001
            },
            'results': {
                'csv_file': 'experiment_results.csv',
                'markdown_file': 'experiment_summary.md'
            },
            'general': {
                'timestamp': None
            }
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
    
    def _validate_config(self):
        """验证配置的有效性"""
        required_paths = ['models_dir', 'train_script', 'eval_script', 'transfer_script']
        
        for path_key in required_paths:
            if path_key not in self.config['paths']:
                raise ValueError(f"配置缺少必需路径: {path_key}")
            
            path_value = self.config['paths'][path_key]
            if not os.path.exists(path_value):
                print(f"⚠️ 路径不存在: {path_key} = {path_value}")
    
    def get(self, key: str, default=None):
        """获取配置值，支持点号分隔的嵌套键"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """设置配置值"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_models_list(self) -> List[str]:
        """获取可用模型列表"""
        models_dir = self.get('paths.models_dir')
        if not os.path.exists(models_dir):
            return []
        
        models = []
        for item in os.listdir(models_dir):
            model_path = os.path.join(models_dir, item)
            if os.path.isdir(model_path):
                models.append(item)
        
        return sorted(models)
    
    def get_model_path(self, model_name: str) -> str:
        """获取模型完整路径"""
        if model_name.startswith('/'):
            return model_name  # 已经是绝对路径
        
        return os.path.join(self.get('paths.models_dir'), model_name)
    
    def save(self):
        """保存配置到文件"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)


# 快速测试配置类
class QuickTestConfig(PipelineConfig):
    """快速测试专用配置"""
    
    def __init__(self):
        # 不调用父类的__init__，直接设置快速测试配置
        self.config_path = None
        self.config = self._create_quick_test_config()
    
    def _create_quick_test_config(self) -> Dict[str, Any]:
        """创建快速测试配置"""
        return {
            'paths': {
                'models_dir': '/root/autodl-tmp/models',
                'train_script': '/root/PAW/train_lora/train_cs_lora_lightning.py',
                'eval_script': '/root/PAW/eval/lightning_eval.py',
                'transfer_script': '/root/PAW/lora_adapter/scripts/transfer_lora_x.py',
                'runs_dir': '/root/PAW/train_lora/runs',
                'transferred_lora_dir': '/root/autodl-tmp/transferred_lora',
                'results_dir': '/root/PAW/results'
            },
            'training': {
                'default_batch_size': 4,
                'default_max_steps': 20,  # 快速测试用更少步数
                'default_lr': '1e-5',
                'datasets': ['piqa', 'arc-challenge', 'arc-easy']
            },
            'evaluation': {
                'sample_ratio': 0.05,  # 5%采样，快速评估
                'default_batch_size': 8
            },
            'transfer': {
                'similarity_threshold': 0.0001
            },
            'results': {
                'csv_file': 'experiment_results.csv',
                'markdown_file': 'experiment_summary.md'
            },
            'recommended_models': {
                'source': 'Qwen-Qwen2.5-0.5B',
                'target': 'Qwen_Qwen2.5-1.5B', 
                'dataset': 'piqa'
            }
        }
