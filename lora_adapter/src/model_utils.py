#!/usr/bin/env python3
"""
模型权重加载和处理工具
"""

import torch
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logger = logging.getLogger(__name__)


class ModelWeightLoader:
    """模型权重加载器"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_lora_weights(self, lora_path: str) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        加载LoRA权重
        
        Args:
            lora_path: LoRA模型路径
            
        Returns:
            lora_weights: LoRA权重字典
            config: LoRA配置
        """
        lora_path = Path(lora_path)
        
        # 加载LoRA配置
        config_path = lora_path / "adapter_config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 加载LoRA权重
        weights_path = lora_path / "adapter_model.safetensors"
        lora_weights = {}
        
        with safe_open(weights_path, framework="pt", device=str(self.device)) as f:
            for key in f.keys():
                lora_weights[key] = f.get_tensor(key)
        
        logger.info(f"加载LoRA权重: {len(lora_weights)}个参数")
        logger.info(f"LoRA配置: rank={config.get('r', 'unknown')}, alpha={config.get('lora_alpha', 'unknown')}")
        
        return lora_weights, config
    
    def load_base_model_weights(self, model_path: str, target_layers: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        加载基础模型权重
        
        Args:
            model_path: 模型路径
            target_layers: 目标层列表，None表示加载所有层
            
        Returns:
            base_weights: 基础模型权重字典
        """
        logger.info(f"加载基础模型权重: {model_path}")
        
        # 使用transformers加载模型，优化内存使用
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # 使用float16减少内存
            device_map="cpu",  # 强制使用CPU避免GPU内存不足
            trust_remote_code=True,
            low_cpu_mem_usage=True  # 启用低内存模式
        )
        
        # 提取权重
        base_weights = {}
        for name, param in model.named_parameters():
            if target_layers is None or any(layer in name for layer in target_layers):
                base_weights[name] = param.detach().clone()
        
        logger.info(f"提取基础权重: {len(base_weights)}个参数")
        
        # 清理内存
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return base_weights
    
    def get_attention_layer_names(self, model_path: str) -> List[str]:
        """
        获取模型的attention层名称
        
        Args:
            model_path: 模型路径
            
        Returns:
            attention_layers: attention层名称列表
        """
        logger.info(f"分析模型结构: {model_path}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map=None,
            trust_remote_code=True
        )
        
        attention_layers = []
        for name, _ in model.named_parameters():
            # 识别attention相关层
            if any(attn_keyword in name.lower() for attn_keyword in 
                   ['attn', 'attention', 'self_attn', 'q_proj', 'k_proj', 'v_proj', 'o_proj']):
                attention_layers.append(name)
        
        logger.info(f"发现attention层: {len(attention_layers)}个")
        
        # 清理内存
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return attention_layers
    
    def map_layer_names(self, source_layers: List[str], target_layers: List[str]) -> Dict[str, str]:
        """
        映射源模型和目标模型的层名称
        
        Args:
            source_layers: 源模型层名称
            target_layers: 目标模型层名称
            
        Returns:
            layer_mapping: 层名称映射字典
        """
        layer_mapping = {}
        
        # 简单的层映射策略：基于层编号和类型
        for source_layer in source_layers:
            best_match = self._find_best_layer_match(source_layer, target_layers)
            if best_match:
                layer_mapping[source_layer] = best_match
        
        logger.info(f"层映射: {len(layer_mapping)}个映射关系")
        return layer_mapping
    
    def _find_best_layer_match(self, source_layer: str, target_layers: List[str]) -> Optional[str]:
        """寻找最佳的层匹配"""
        # 提取层信息
        source_info = self._parse_layer_info(source_layer)
        
        best_match = None
        best_score = 0
        
        for target_layer in target_layers:
            target_info = self._parse_layer_info(target_layer)
            score = self._compute_layer_similarity(source_info, target_info)
            
            if score > best_score:
                best_score = score
                best_match = target_layer
        
        # 只返回足够相似的匹配
        return best_match if best_score > 0.5 else None
    
    def _parse_layer_info(self, layer_name: str) -> Dict[str, str]:
        """解析层名称信息"""
        info = {
            'layer_num': 'unknown',
            'module_type': 'unknown',
            'proj_type': 'unknown'
        }
        
        # 提取层编号
        import re
        layer_match = re.search(r'layers?\.(\d+)', layer_name.lower())
        if layer_match:
            info['layer_num'] = layer_match.group(1)
        
        # 提取模块类型
        if 'self_attn' in layer_name.lower():
            info['module_type'] = 'self_attn'
        elif 'attn' in layer_name.lower():
            info['module_type'] = 'attn'
        
        # 提取投影类型
        for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            if proj in layer_name.lower():
                info['proj_type'] = proj
                break
        
        return info
    
    def _compute_layer_similarity(self, info1: Dict[str, str], info2: Dict[str, str]) -> float:
        """计算层相似性分数"""
        score = 0.0
        
        # 层编号匹配
        if info1['layer_num'] == info2['layer_num'] and info1['layer_num'] != 'unknown':
            score += 0.4
        
        # 模块类型匹配
        if info1['module_type'] == info2['module_type'] and info1['module_type'] != 'unknown':
            score += 0.4
        
        # 投影类型匹配
        if info1['proj_type'] == info2['proj_type'] and info1['proj_type'] != 'unknown':
            score += 0.2
        
        return score


def save_transferred_lora(lora_weights: Dict[str, torch.Tensor], 
                         config: Dict, 
                         output_path: str):
    """
    保存迁移后的LoRA权重
    
    Args:
        lora_weights: LoRA权重字典
        config: LoRA配置
        output_path: 输出路径
    """
    from safetensors.torch import save_file
    import json
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存权重
    weights_file = output_path / "adapter_model.safetensors"
    save_file(lora_weights, weights_file)
    
    # 保存配置
    config_file = output_path / "adapter_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"迁移后的LoRA权重已保存到: {output_path}")
