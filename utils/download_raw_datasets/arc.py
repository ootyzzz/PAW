import os
from datasets import load_dataset

def setup_env():
    # 设置代理
    os.environ['https_proxy'] = 'http://agent.baidu.com:8891'
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def download_ai2_arc(output_dir="./raw_datasets"):
    setup_env()

    # 分别下载 ARC-Challenge 和 ARC-Easy
    subsets = ["ARC-Challenge", "ARC-Easy"]
    for subset in subsets:
        print(f" Downloading {subset} ...")
        dataset = load_dataset("ai2_arc", subset)

        # 保存为 JSONL 文件到 output_dir
        for split in dataset:
            save_path = os.path.join(output_dir, f"{subset.lower()}_{split}.jsonl")
            os.makedirs(output_dir, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                for item in dataset[split]:
                    f.write(f"{item}\n")
            print(f" Saved {subset}/{split} to {save_path}")

if __name__ == "__main__":
    download_ai2_arc()