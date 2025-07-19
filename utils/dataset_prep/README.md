# 数据集预处理工具

这个文件夹包含了用于处理commonsense数据集的工具。

## 📁 文件说明

- **`cs_mixer.py`** - 混合并标准化7个commonsense数据集
- **`cs_formatter.py`** - 最终数据清理和格式化
- **`cs_pipeline.py`** - 完整的数据处理流水线

## 🚀 使用方法

### 完整流程（推荐）
```bash
python utils/dataset_prep/cs_pipeline.py
```

### 单独运行步骤
```bash
# 步骤1: 混合数据集
python utils/dataset_prep/cs_mixer.py

# 步骤2: 格式化数据
python utils/dataset_prep/cs_formatter.py
```

## 📊 输出文件

- `raw_datasets/commonsense/cs_mixed.jsonl` - 混合后的数据
- `raw_datasets/commonsense/cs_formatted.jsonl` - 最终格式化的数据

## 🛠️ 参数选项

```bash
python utils/dataset_prep/cs_pipeline.py --help
```

常用参数：
- `--datasets_dir` - 数据集根目录（默认：raw_datasets）
- `--output_dir` - 输出目录（默认：raw_datasets/commonsense）
- `--seed` - 随机种子（默认：42）
- `--max_samples` - 每个数据集最大样本数
- `--skip_mix` - 跳过混合步骤
- `--skip_clean` - 跳过清理步骤
