"""
数据处理模块
包含数据集加载和处理相关功能
"""

from .config import *


def get_test_file_path(dataset_name: str) -> str:
    """获取测试文件路径"""
    # 尝试多个可能的数据目录路径
    possible_paths = [
        f"data_to_lora/cs/{dataset_name}",  # 从PAW根目录运行
        f"../data_to_lora/cs/{dataset_name}",  # 从pipeline目录运行
        f"../../data_to_lora/cs/{dataset_name}",  # 从子目录运行
    ]
    
    for data_dir in possible_paths:
        test_file = f"{data_dir}/{dataset_name}_test_formatted.jsonl"
        validation_file = f"{data_dir}/{dataset_name}_validation_formatted.jsonl"
        
        if os.path.exists(test_file):
            return test_file
        elif os.path.exists(validation_file):
            print(f"📝 使用validation文件作为test: {validation_file}")
            return validation_file
    
    # 如果都找不到，给出详细的错误信息
    raise FileNotFoundError(f"数据集 {dataset_name} 找不到test或validation文件。尝试过的路径: {possible_paths}")


class SimpleDataset(Dataset):
    """简单的数据集类，适用于评估"""
    def __init__(self, data_file: str, sample_ratio: float = 1.0):
        self.data = self._load_data(data_file)
        original_size = len(self.data)
        
        # 硬性限制：无论数据集有多少样本，最多只使用前1000个
        max_samples = 1000
        if len(self.data) > max_samples:
            self.data = self.data[:max_samples]
            print(f"  📊 限制样本数量: {max_samples}/{original_size} (使用前{max_samples}个样本)")
        else:
            print(f"  📊 使用完整数据: {len(self.data)}样本 (数据集小于{max_samples}个样本)")
        
        # 如果需要进一步采样加速评估
        if sample_ratio < 1.0:
            current_size = len(self.data)
            sample_size = max(1, int(current_size * sample_ratio))
            # 使用固定种子保证采样可重复
            random.seed(42)
            self.data = random.sample(self.data, sample_size)
            print(f"  📊 进一步采样: {sample_size}/{current_size} ({sample_ratio*100:.1f}%)")
    
    def _load_data(self, data_file: str) -> List[Dict[str, Any]]:
        """从JSONL文件加载数据"""
        data = []
        if os.path.exists(data_file):
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx].copy()
