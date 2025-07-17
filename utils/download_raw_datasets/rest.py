import os
import json
from datasets import load_dataset, DownloadConfig

def setup_env():
    # 设置代理
    os.environ['HTTP_PROXY'] = 'http://agent.baidu.com:8891'
    os.environ['HTTPS_PROXY'] = 'http://agent.baidu.com:8891'

    # 使用 Huggingface 镜像（清华）
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HUB_BASE'] = 'https://hf-mirror.com'
    os.environ['HUGGINGFACE_HUB_CACHE'] = './hf_cache'
    os.environ['HF_DATASETS_CACHE'] = './hf_cache'

def download_all_raw_datasets(output_dir="./raw_datasets"):
    setup_env()

    # 明确传入 download_config，设置 proxy
    download_config = DownloadConfig(
        proxies={
            "http": os.environ['HTTP_PROXY'],
            "https": os.environ['HTTPS_PROXY'],
        },
        resume_download=True
    )

    datasets_to_download = {
        # "boolq":           ("boolq",   None),
        # "openbookqa":      ("openbookqa",   "main"),
        "hellaswag":       ("hellaswag",    None),
        "piqa":            ("piqa",         None),
        # "winogrande":      ("winogrande",   "winogrande_xl"),
    }

    for local_name, (repo_id, config) in datasets_to_download.items():
        ds_dir = os.path.join(output_dir, local_name)
        os.makedirs(ds_dir, exist_ok=True)

        print(f" Downloading {local_name} ...")

        try:
            if config:
                ds = load_dataset(repo_id, config, download_config=download_config)
            else:
                ds = load_dataset(repo_id, download_config=download_config)

            for split, split_ds in ds.items():
                save_path = os.path.join(ds_dir, f"{local_name}_{split}.jsonl")
                with open(save_path, "w", encoding="utf-8") as f:
                    for item in split_ds:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                print(f" Saved {local_name}/{split} → {save_path} ({len(split_ds)} samples)")

        except Exception as e:
            print(f" Failed to download {local_name}: {str(e)}")

if __name__ == "__main__":
    download_all_raw_datasets()
