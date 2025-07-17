import os
import json
from datasets import Dataset

def parquet_to_jsonl(parquet_file, jsonl_file):
    # 读取 parquet 为 Huggingface Dataset 对象
    dataset = Dataset.from_parquet(parquet_file)

    # 保存为 JSONL
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f" Converted {parquet_file} → {jsonl_file} ({len(dataset)} samples)")

def process_local_hellaswag(parquet_dir, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # split: (输入文件名, 输出文件名)
    parquet_files = {
        "train": ("train-00000-of-00001.parquet", f"hellaswag_train.jsonl"),
        "validation": ("validation-00000-of-00001.parquet", f"hellaswag_validation.jsonl"),
        "test": ("test-00000-of-00001.parquet", f"hellaswag_test.jsonl"),
    }

    for split, (input_file, output_file) in parquet_files.items():
        parquet_path = os.path.join(parquet_dir, input_file)
        jsonl_path = os.path.join(output_dir, output_file)

        if os.path.exists(parquet_path):
            parquet_to_jsonl(parquet_path, jsonl_path)
        else:
            print(f"⚠️ File not found: {parquet_path}")
            

if __name__ == "__main__":
    # 输入路径是 parquet 所在目录
    parquet_base_dir = "./utils/download_raw_datasets/hellaswag/data"
    # 输出路径是之前 download 脚本用的目录
    output_dir = "./raw_datasets/hellaswag"
    process_local_hellaswag(parquet_base_dir, output_dir)
