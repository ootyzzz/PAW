"""
main.py
pipeline 主入口，调度各数据集训练流程
"""
import os
from adapters.arc_adapter import arc_adapter
from adapters.boolq_adapter import boolq_adapter
from adapters.obqa_adapter import obqa_adapter
from adapters.hellaswag_adapter import hellaswag_adapter
from adapters.piqa_adapter import piqa_adapter
from adapters.winogrande_adapter import winogrande_adapter
from model.train import train_loop

# 加载数据集样本（需补充实际数据加载逻辑）

import json

def load_samples(dataset_name):
    """
    加载指定数据集的样本
    默认从 data/{dataset_name}/samples.json 加载
    """
    data_path = os.path.join('data', dataset_name, 'samples.json')
    if not os.path.exists(data_path):
        print(f"[警告] 未找到数据文件: {data_path}")
        return []
    with open(data_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    return samples

# 加载 LoRA checkpoint 路径（需补充实际 checkpoint 枚举逻辑）

def load_lora_checkpoints(dataset_name):
    """
    加载指定数据集的 LoRA checkpoint 路径
    默认从 data/{dataset_name}/checkpoints/ 目录下枚举所有 .pt 文件
    """
    ckpt_dir = os.path.join('data', dataset_name, 'checkpoints')
    if not os.path.exists(ckpt_dir):
        print(f"[警告] 未找到 checkpoint 目录: {ckpt_dir}")
        return []
    ckpt_files = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith('.pt')]
    if not ckpt_files:
        print(f"[警告] {ckpt_dir} 下无 .pt 文件")
    return ckpt_files

# 获取对应数据集的 prompt adapter
def get_adapter(dataset_name):
    adapters = {
        'ARC-e': arc_adapter,
        'ARC-c': arc_adapter,
        'BoolQ': boolq_adapter,
        'OBQA': obqa_adapter,
        'HellaSwag': hellaswag_adapter,
        'PIQA': piqa_adapter,
        'WinoGrande': winogrande_adapter,
    }
    return adapters[dataset_name]

# 构建 prompt batch
def prepare_prompt_batches(samples, adapter, batch_size=8):
    """
    按 batch_size 构建 prompt batch
    """
    prompts = [adapter(s) for s in samples]
    return [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]

def main():
    """
    主流程，遍历所有数据集，训练 Hyper-Convolution Decoder
    """
    datasets = ['ARC-e', 'ARC-c', 'BoolQ', 'OBQA', 'HellaSwag', 'PIQA', 'WinoGrande']
    for ds in datasets:
        samples = load_samples(ds)
        lora_ckpts = load_lora_checkpoints(ds)
        adapter = get_adapter(ds)
        prompt_batches = prepare_prompt_batches(samples, adapter)
        train_loop(prompt_batches, lora_ckpts)

if __name__ == '__main__':
    main()
