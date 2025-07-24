import os
from modelscope.hub.snapshot_download import snapshot_download

def main():
    model_id = 'LLM-Research/Llama-3.2-3B-Instruct'
    download_dir = '/root/autodl-tmp/models/Llama-3.2-3B-Instruct'

    os.makedirs(download_dir, exist_ok=True)

    try:
        print(f"开始下载模型: {model_id}")
        model_path = snapshot_download(model_id, cache_dir=download_dir)
        print(f"模型已下载到: {model_path}")
        files = os.listdir(model_path)
        print("主要文件:")
        for f in files:
            print(f"  - {f}")
    except Exception as e:
        print(f"下载失败: {e}")

if __name__ == "__main__":
    main()