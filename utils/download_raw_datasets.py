"""
download_raw_datasets.py
下载 7 个 Commonsense 数据集原始文件到 data/raw/ 目录。
"""
from datasets import load_dataset
import os
import json

def save_raw(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)



# ARC-e
arc = load_dataset('ai2_arc', 'ARC-Easy')
for split in arc:
    save_raw(arc[split].to_list(), f'data/raw/ARC-e_{split}.json')

# ARC-c
arc_c = load_dataset('ai2_arc', 'ARC-Challenge')
for split in arc_c:
    save_raw(arc_c[split].to_list(), f'data/raw/ARC-c_{split}.json')

# BoolQ
boolq = load_dataset('boolq')
for split in boolq:
    save_raw(boolq[split].to_list(), f'data/raw/BoolQ_{split}.json')

# OBQA
obqa = load_dataset('openbookqa', 'main')
for split in obqa:
    save_raw(obqa[split].to_list(), f'data/raw/OBQA_{split}.json')

# HellaSwag
hella = load_dataset('hellaswag')
for split in hella:
    save_raw(hella[split].to_list(), f'data/raw/HellaSwag_{split}.json')

# PIQA
# piqa = load_dataset('piqa')
# for split in piqa:
#     save_raw(piqa[split].to_list(), f'data/raw/PIQA_{split}.json')

# WinoGrande
wino = load_dataset('winogrande', 'winogrande_xl')
for split in wino:
    save_raw(wino[split].to_list(), f'data/raw/WinoGrande_{split}.json')

print('所有原始数据集已下载到 data/raw/')
