# Lightning模型评估工具

## 快速使用

### 基本用法
```bash
# 单个模型评估
python eval/lightning_eval_new.py --models_list /root/autodl-tmp/models/Qwen_Qwen2.5-1.5B --dataset arc-challenge

# 多个模型评估
python eval/lightning_eval_new.py --models_list /root/autodl-tmp/models/Qwen_Qwen2.5-1.5B /root/autodl-tmp/models/gemma-2-2b-it --dataset arc-challenge

# 快速测试（使用1%数据）
python eval/lightning_eval_new.py --models_list /root/autodl-tmp/models/Qwen_Qwen2.5-1.5B --dataset arc-challenge --sample_ratio 0.01

# LoRA模型评估
python eval/lightning_eval_new.py --models_list /root/PAW/runs/arc-challenge/Qwen_Qwen2.5-1.5B/181133/final_model --base_model /root/autodl-tmp/models/Qwen_Qwen2.5-1.5B --dataset arc-challenge
```

### 环境要求
```bash
conda activate cuda312
```

## 工具概述

`lightning_eval_new.py` 是基于PyTorch Lightning的模型评估工具，支持批量评估多个模型，包括基础模型和LoRA微调模型。

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--models_list` | str+ | 必需 | 要评估的模型路径列表 |
| `--dataset` | str | arc-challenge | 数据集名称 |
| `--output_dir` | str | eval/results | 评估结果输出目录 |
| `--base_model` | str | None | LoRA模型的基础模型路径（可选） |
| `--sample_ratio` | float | 1.0 | 数据采样比例（0.01-1.0） |
| `--batch_size` | int | 8 | 批处理大小 |

## 支持的模型类型

- **基础模型**: 标准的预训练语言模型
- **LoRA模型**: 包含 `adapter_config.json` 的LoRA微调模型
- **HuggingFace模型**: 支持通过模型名称直接加载

## 支持的数据集

- `arc-challenge`: ARC Challenge数据集
- 其他数据集需要在 `data_to_lora/cs/` 目录下有对应的测试文件

## 输出结果

### 控制台输出
- 实时评估进度
- 模型加载状态
- 评估指标（Loss、Accuracy、Perplexity）
- 性能统计（评估时间、样本/秒）

### 文件输出
- **JSON格式**: `lightning_evaluation_summary_YYYYMMDD_HHMMSS.json`
- **CSV格式**: `lightning_evaluation_results_YYYYMMDD_HHMMSS.csv`
- **单模型结果**: `{model_name}_{dataset}_evaluation_results.json`

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
├── lightning_eval_new.py          # 主入口脚本
├── results/                       # 评估结果目录
└── README.md                      # 本文档
```

## 核心功能

### 模型加载
- 自动检测模型类型（基础模型/LoRA模型）
- 支持本地路径和HuggingFace模型名称
- 智能设备分配和精度优化

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
- 混合精度训练（16-bit）
- GPU内存优化
- 批量处理优化

## 注意事项

1. **环境依赖**: 必须在 `cuda312` conda环境中运行
2. **内存管理**: 大模型评估时注意GPU内存使用
3. **路径格式**: 模型路径必须存在或为有效的HuggingFace模型名称
4. **LoRA模型**: 需要指定 `--base_model` 参数或确保adapter_config.json中包含基础模型路径

## 故障排除

### 常见问题

**模型加载失败**
- 检查模型路径是否正确
- 确认LoRA模型的基础模型路径
- 验证模型文件完整性

**内存不足**
- 减小 `--batch_size` 参数
- 使用 `--sample_ratio` 减少数据量
- 检查GPU内存使用情况

**数据集不存在**
- 确认数据集文件在 `data_to_lora/cs/{dataset}/` 目录下
- 检查数据文件格式是否正确
