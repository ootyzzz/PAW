import os
from modelscope.hub.file_download import model_file_download

def main():
    model_id = 'LLM-Research/Llama-3.2-3B-Instruct'
    download_dir = '/root/autodl-tmp/models/Llama-3.2-3B-Instruct'
    
    os.makedirs(download_dir, exist_ok=True)
    
    # 需要下载的文件列表（避开 consolidated.00.pth）
    required_files = [
        'LICENSE.txt',
        'README.md',
        'config.json',
        'configuration.json',
        'generation_config.json',
        'special_tokens_map.json',
        'tokenizer.json',
        'tokenizer_config.json',
        'USE_POLICY.md',
        'model.safetensors.index.json',
        'original/params.json',
        'original/orig_params.json',
        'original/tokenizer.model',
        'model-00001-of-00002.safetensors',
        'model-00002-of-00002.safetensors'
    ]

    try:
        print(f"开始下载模型: {model_id}")
        print(f"目标路径: {download_dir}")
        
        downloaded_count = 0
        for file in required_files:
            target_path = os.path.join(download_dir, file)
            if os.path.exists(target_path):
                print(f"已存在，跳过: {file}")
                downloaded_count += 1
                continue
            
            try:
                print(f"正在下载: {file}")
                model_file_download(
                    model_id=model_id,
                    file_path=file,
                    local_dir=download_dir
                )
                downloaded_count += 1
                print(f"下载成功 ({downloaded_count}/{len(required_files)}): {file}")
            except Exception as e:
                print(f"下载失败: {file}，错误: {e}")
                # 继续尝试其他文件

        print(f"\n文件处理完成，成功下载 {downloaded_count}/{len(required_files)} 个文件")
        print(f"模型文件保存于: {download_dir}")
        
    except Exception as e:
        print(f"执行中断: {e}")

if __name__ == "__main__":
    main()
