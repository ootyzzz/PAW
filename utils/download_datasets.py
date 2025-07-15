"""
download_datasets.py
格式化 data/raw/ 下的原始数据集，生成 pipeline 适配的 samples.json。
"""
import os
import json

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

boolq_samples = [{"question": x['question'], "choices": ["Yes", "No"]} for x in boolq]
save_json(boolq_samples, 'data/BoolQ/samples.json')

import glob

def merge_and_format(files, formatter):
    samples = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            samples.extend([formatter(x) for x in data])
    return samples

# ARC-e
arc_files = glob.glob('data/raw/ARC-e_*.json')
arc_samples = merge_and_format(arc_files, lambda x: {"question": x['question'], "choices": x['choices']})
save_json(arc_samples, 'data/ARC-e/samples.json')

# ARC-c
arc_c_files = glob.glob('data/raw/ARC-c_*.json')
arc_c_samples = merge_and_format(arc_c_files, lambda x: {"question": x['question'], "choices": x['choices']})
save_json(arc_c_samples, 'data/ARC-c/samples.json')

# BoolQ
boolq_files = glob.glob('data/raw/BoolQ_*.json')
boolq_samples = merge_and_format(boolq_files, lambda x: {"question": x['question'], "choices": ["Yes", "No"]})
save_json(boolq_samples, 'data/BoolQ/samples.json')

# OBQA
obqa_files = glob.glob('data/raw/OBQA_*.json')
obqa_samples = merge_and_format(obqa_files, lambda x: {"question": x['question_stem'], "choices": x['choices']['text']})
save_json(obqa_samples, 'data/OBQA/samples.json')

# HellaSwag
hella_files = glob.glob('data/raw/HellaSwag_*.json')
hella_samples = merge_and_format(hella_files, lambda x: {"context": x['ctx_a'], "choices": [x['endings'][0], x['endings'][1], x['endings'][2], x['endings'][3]]})
save_json(hella_samples, 'data/HellaSwag/samples.json')

# PIQA
piqa_files = glob.glob('data/raw/PIQA_*.json')
piqa_samples = merge_and_format(piqa_files, lambda x: {"question": x['goal'], "choices": [x['sol1'], x['sol2']]})
save_json(piqa_samples, 'data/PIQA/samples.json')

# WinoGrande
wino_files = glob.glob('data/raw/WinoGrande_*.json')
wino_samples = merge_and_format(wino_files, lambda x: {"question": x['sentence'], "choices": [x['option1'], x['option2']]})
save_json(wino_samples, 'data/WinoGrande/samples.json')

print('所有数据集已格式化完成！')
