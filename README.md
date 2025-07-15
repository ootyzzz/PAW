## LoRA 微调训练脚本

项目已提供 Qwen-0.5B LoRA 微调训练脚本：

```bash
python utils/train_lora.py
```

脚本默认对 BoolQ 数据集进行微调，训练完成后会自动保存 LoRA checkpoint 到 `data/BoolQ/checkpoints/epoch_10_lora.pt`。
如需训练其他数据集，请修改 `dataset_name` 及数据预处理部分。

## LoRA checkpoint 结构说明

- 每个 checkpoint 为 HuggingFace 格式权重文件（.pt），内容为 LoRA adapter 层参数。
- 目录结构示例：
  ```
  data/BoolQ/checkpoints/
    ├── epoch_1_lora.pt
    ├── epoch_2_lora.pt
    └── ...
  ```
- 加载方式参考 PEFT 官方文档或 pipeline 说明。
# DnD Hyper-Convolution Decoder Pipeline (Qwen-0.5B, Commonsense)

本项目严格复现 DnD 原论文（arXiv:2506.16406v1）在 Commonsense 任务下的 Hyper-Convolution Parameter Generator 训练流程，适用于 Qwen-0.5B 模型。

## 目录结构
```
P2W/
├── adapters/           # 各数据集 prompt 构建适配器
├── data/               # 数据集存放目录（需自行准备）
├── lora/               # LoRA 权重展平与 checkpoint 工具
├── model/              # Hyper-Convolution Decoder 及训练主循环
├── utils/              # embedding、损失、调度等工具
├── main.py             # pipeline 主入口
├── requirements.txt    # 依赖包清单
└── README.md           # 项目说明文档
```

## 依赖安装
请先安装依赖：
```bash
pip install -r requirements.txt
```


## 数据准备


### 数据集下载与格式化分步说明

1. 下载原始数据集（HuggingFace Hub）：
   ```bash
   python utils/download_raw_datasets.py
   ```
   运行后会自动下载原始数据到 `data/raw/` 目录。

2. 格式化数据集，生成 pipeline 适配的 samples.json：
   ```bash
   python utils/download_datasets.py
   ```
   运行后会自动生成 `data/{数据集名}/samples.json`，格式已适配 main.py。

每个样本格式为字典，例如：
```python
{"question": "...", "choices": ["A", "B"]}
```
或 HellaSwag:
```python
{"context": "...", "choices": ["A", "B", "C", "D"]}
```

### LoRA checkpoint
LoRA checkpoint（.pt 文件）需按数据集分别存放于 `data/{数据集名}/checkpoints/` 目录下。

## 主要模块说明
- `adapters/`：每个数据集一个 adapter，统一 prompt 格式。
- `utils/embedding.py`：MiniLM-L6-v2 文本编码器，输出 embedding。
- `lora/flatten.py`：LoRA 权重展平为一维向量。
- `lora/checkpoint_utils.py`：checkpoint 保存、加载、枚举。
- `model/hyperconv_decoder.py`：Hyper-Convolution Decoder，严格按论文结构实现。
- `model/train.py`：训练主循环，完成 prompt batch 与 LoRA checkpoint 的配对训练。
- `main.py`：遍历所有数据集，调度训练流程。

## 运行方法
1. 按如下格式准备数据和 checkpoint，`main.py` 会自动加载：
   - 数据样本：放在 `data/{数据集名}/samples.json`，内容为样本列表（每个样本为字典，如 `{ "question": "...", "choices": ["A", "B"] }`）。
   - LoRA checkpoint：放在 `data/{数据集名}/checkpoints/` 目录下，所有 `.pt` 文件会自动枚举。
2. 运行主入口：
   ```bash
   python main.py
   ```
3. 训练过程中会自动保存最佳模型到 `best_model.pt`。

## 代码注释
所有核心文件均已添加详细中文注释，便于理解和二次开发。

## 复现指南
- 严格遵循 DnD 论文流程，包括 prompt batch 构建、same-task random-pair 配对、MiniLM embedding、Hyper-Convolution Decoder 结构、MSE loss、AdamW 优化、Cosine LR 调度等。
- 推荐每个数据集独立管理 prompt batch 和 LoRA checkpoint，保证 shape 和顺序一致。

## 常见问题
- 若无数据和 checkpoint，训练循环不会实际执行。
- 若需 GPU 加速，torch 会自动检测 CUDA。
- 如需英文注释或样例数据，请联系维护者。

---

**本项目适合科研复现、模型微调、结构探索等场景。**
