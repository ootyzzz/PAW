import os
from modelscope.hub.file_download import model_file_download

def main():
    model_id = 'LLM-Research/Meta-Llama-3.1-8B-Instruct'
    download_dir = '../../../autodl-tmp/models/Meta-Llama-3.1-8B-Instruct'
    
    # 确保目录存在
    os.makedirs(download_dir, exist_ok=True)
    
    # 手动指定需要下载的关键文件（排除大型pth文件）
    # 这些文件是模型运行必需的配置、tokenizer等小文件，以及safetensors权重
    required_files = [
        'LICENSE',
        'README.md',
        'config.json',
        'configuration.json',
        'generation_config.json',
        'special_tokens_map.json',
        'tokenizer.json',
        'tokenizer.model',
        'tokenizer_config.json',
        'USE_POLICY.md',
        'model.safetensors.index.json',
        'original/params.json',
        'original/tokenizer.model',  # 注意：这里是tokenizer，不是权重
        'model-00001-of-00004.safetensors',
        'model-00002-of-00004.safetensors',
        'model-00003-of-00004.safetensors',
        'model-00004-of-00004.safetensors'
    ]
    
    try:
        print(f"开始下载模型: {model_id}")
        print(f"目标路径: {download_dir}")
        
        downloaded_count = 0
        for file in required_files:
            target_path = os.path.join(download_dir, file)
            # 跳过已下载的文件
            if os.path.exists(target_path):
                print(f"已存在，跳过: {file}")
                downloaded_count += 1
                continue
            
            # 下载单个文件
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
                # 继续下载其他文件，不中断
            
        print(f"\n所有文件处理完成，成功下载 {downloaded_count}/{len(required_files)} 个文件")
        print(f"模型文件已保存到: {download_dir}")
        
    except Exception as e:
        print(f"执行中断: {e}")

if __name__ == "__main__":
    main()