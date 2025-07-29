# Lightning模型评估工具

## 快速使用

### 基本用法
```bash
# 单个基础模型评估
python eval/lightning_eval.py --base_model ../autodl-tmp/models/Qwen2.5-7B-Instruct --dataset arc-challenge --batch_size 4

# 使用LoRA参数
python eval/lightning_eval.py --lora ../autodl-tmp/models/Qwen2.5-7B-Instruct --dataset arc-challenge --batch_size 4

# 快速测试（使用1%数据）
python eval/lightning_eval.py --base_model ../autodl-tmp/models/Qwen2.5-7B-Instruct --dataset arc-challenge --batch_size 4 --sample_ratio 0.01

# LoRA模型评估
python eval/lightning_eval.py --lora /path/to/lora/model --base_model ../autodl-tmp/models/Qwen2.5-7B-Instruct --dataset arc-challenge --batch_size 4
```

### 环境要求
```bash
conda activate cuda312
```

## 工具概述

`lightning_eval.py` 是基于PyTorch Lightning的模型评估工具，支持评估基础模型和LoRA微调模型。

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--lora` | str* | None | LoRA模型路径列表（可选） |
| `--base_model` | str | None | 基础模型路径（必需，除非提供lora） |
| `--dataset` | str | arc-challenge | 数据集名称 |
| `--output_dir` | str | eval/results | 评估结果输出目录 |
| `--sample_ratio` | float | 1.0 | 数据采样比例（0.01-1.0） |
| `--batch_size` | int | 8 | 批处理大小 |

## 支持的模型类型

- **基础模型**: 标准的预训练语言模型
- **LoRA模型**: 包含 `adapter_config.json` 的LoRA微调模型
- **HuggingFace模型**: 支持通过模型名称直接加载

## 支持的数据集

- `arc-challenge`: ARC Challenge数据集
- `arc-easy`: ARC Easy数据集
- `boolq`: BoolQ数据集
- `hellaswag`: HellaSwag数据集
- `openbookqa`: OpenBookQA数据集
- `piqa`: PIQA数据集
- `winogrande`: WinoGrande数据集

数据集文件位置：`data_to_lora/cs/{dataset}/`

## 输出结果

### 控制台输出
- 实时评估进度
- 模型加载状态
- 评估指标（Loss、Accuracy、Perplexity）
- 性能统计（评估时间、样本/秒）

### 文件输出
- **CSV格式**: `lightning_evaluation_results_YYYYMMDD_HHMMSS.csv`
  - 包含字段：Model, Dataset, Loss, Accuracy, Perplexity, Eval_Time(s), Samples/Sec, Batch_Size, Timestamp
- **总结果文件**: `results/experiment_results.csv` (仅当sample_ratio=1.0时)

## 项目结构

```
PAW/eval/
├── core/                           # 核心模块
│   ├── __init__.py                # 模块初始化
│   ├── config.py                  # 配置和依赖管理
│   ├── data.py                    # 数据加载和处理
│   ├── evaluator.py               # Lightning评估器
│   ├── model_loader.py            # 模型加载工具
│   └── batch_eval.py              # 批量评估逻辑
├── lightning_eval.py              # 主入口脚本
├── results/                       # 评估结果目录
└── README.md                      # 本文档
```

## 核心功能

### 模型加载
- 自动检测模型类型（基础模型/LoRA模型）
- 支持本地路径和HuggingFace模型名称
- 智能设备分配和精度优化（fp16混合精度，Gemma模型使用fp32）

### 数据处理
- 支持数据采样加速评估
- 自动数据格式转换
- 批量数据加载优化

### 评估指标
- **Loss**: 交叉熵损失
- **Accuracy**: 预测准确率
- **Perplexity**: 困惑度

### 性能优化
- PyTorch Lightning并行处理
- 混合精度训练（16-bit，Gemma模型除外）
- GPU内存优化
- 批量处理优化

## 当前状态

### 已修复的问题
- ✅ batch_size参数现在可以正确传递和使用
- ✅ CSV输出功能正常工作，使用原生CSV模块避免pandas数据类型问题
- ✅ 支持完整的评估指标输出

### 推荐配置
- **服务器环境**: batch_size=4（大部分模型），batch_size=2（Gemma模型）
- **4090显卡**: batch_size=1-2（根据模型大小调整）
- **模型路径**: 使用 `../autodl-tmp/models/` 作为模型基础目录

## 注意事项

1. **环境依赖**: 必须在 `cuda312` conda环境中运行
2. **内存管理**: 大模型评估时注意GPU内存使用，适当调整batch_size
3. **路径格式**: 模型路径必须存在或为有效的HuggingFace模型名称
4. **LoRA模型**: 需要指定 `--base_model` 参数

## 故障排除

### 常见问题

**模型加载失败**
- 检查模型路径是否正确
- 确认LoRA模型的基础模型路径
- 验证模型文件完整性

**内存不足**
- 减小 `--batch_size` 参数（推荐从4降到2或1）
- 使用 `--sample_ratio` 减少数据量进行快速测试
- 检查GPU内存使用情况

**数据集不存在**
- 确认数据集文件在 `data_to_lora/cs/{dataset}/` 目录下
- 检查数据文件格式是否正确

**CSV输出问题**
- 当前版本使用原生CSV模块，避免了pandas数据类型转换问题
- CSV文件保存在 `eval/results/` 目录下，文件名包含时间戳

## 批量评估建议

对于批量评估多个模型，建议使用shell脚本或手动运行：

```bash
# 示例：评估多个模型在单个数据集上
for model in Qwen2.5-7B-Instruct baichuan-7B chatglm3-6b; do
    python eval/lightning_eval.py --base_model ../autodl-tmp/models/$model --dataset arc-challenge --batch_size 4
done
