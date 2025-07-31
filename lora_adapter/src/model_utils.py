#!/usr/bin/env python3
"""
模型权重加载工具
"""

import torch
import json
from pathlib import Path
from typing import Dict, Tuple
from safetensors import safe_open
from transformers import AutoModelForCausalLM
import logging

logger = logging.getLogger(__name__)

class ModelWeightLoader:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_lora_weights(self, lora_path: str) -> Tuple[Dict[str, torch.Tensor], Dict]:
        lora_path = Path(lora_path)
        config_path = lora_path / "adapter_config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)

        weights_path = lora_path / "adapter_model.safetensors"
        lora_weights = {}
        with safe_open(weights_path, framework="pt", device=str(self.device)) as f:
            for key in f.keys():
                lora_weights[key] = f.get_tensor(key)

        return lora_weights, config

    def load_base_model_weights(self, model_path: str) -> Dict[str, torch.Tensor]:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        base_weights = {name: param.detach().clone() for name, param in model.named_parameters()}
        del model
        torch.cuda.empty_cache()
        return base_weights

def save_transferred_lora(lora_weights: Dict[str, torch.Tensor], config: Dict, output_path: str):
    from safetensors.torch import save_file
    import json

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    weights_file = output_path / "adapter_model.safetensors"
    save_file(lora_weights, weights_file)

    config_file = output_path / "adapter_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"迁移后的LoRA权重已保存到: {output_path}")
