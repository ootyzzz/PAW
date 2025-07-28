import os
from modelscope.hub.snapshot_download import snapshot_download

def main():
    model_id = 'Qwen/Qwen2.5-7B-Instruct'
    download_dir = '../../../autodl-tmp/models/Qwen2.5-7B-Instruct'

    os.makedirs(download_dir, exist_ok=True)

    try:
        print(f"开始下载模型: {model_id}")
        # 移除不支持的参数，直接使用基础下载功能
        model_path = snapshot_download(model_id, cache_dir=download_dir)
        print(f"模型已完整下载到: {model_path}")
        
        # 列出下载的主要文件
        print("\n下载的主要文件:")
        for root, dirs, files in os.walk(model_path):
            if root == model_path:
                for f in files:
                    print(f"  - {f}")
                for d in dirs:
                    print(f"  - 子目录: {d}")
                break

    except Exception as e:
        print(f"下载过程出错: {e}")
        print("提示：若需重新下载，可删除目录后重试：", download_dir)

if __name__ == "__main__":
    main()