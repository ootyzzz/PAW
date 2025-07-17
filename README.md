# P2W LoRA训练项目唯一入口说明

本项目所有操作（环境配置、模型下载、训练、性能监控等）均通过 `baidu_gpu_lora_training.ipynb` notebook 完成。

- 不再使用 Makefile、setup.sh、train.py 等单独入口脚本。
- 只需依次运行 notebook 各单元格，即可完成全部流程。
- 底层功能如模型下载、训练等，均可在 notebook 内通过 `!python scripts/xxx.py` 或 `subprocess` 调用。

## 快速开始

1. 打开并运行 `baidu_gpu_lora_training.ipynb`
2. 按顺序执行各单元格，完成环境检查、数据准备、模型下载、训练、性能分析等全部流程。
3. 如需自定义参数或流程，可直接在 notebook 中修改相关代码。

---

> 其他脚本仅作为 notebook 的底层工具，不再单独暴露入口。

---

# P2W: 现代化LoRA训练框架

一个基于Hugging Face生态的现代化LoRA训练框架，专为foundation models（如Qwen2.5）微调而设计。

## ✨ 特性

- 🚀 **现代化架构**: 基于Hugging Face Transformers + PEFT + Accelerate
- 📊 **完整监控**: 集成WandB、TensorBoard等训练监控工具
- 🔧 **易于配置**: 使用YAML配置文件，支持多种训练策略
- 📈 **实验管理**: 完整的实验跟踪和结果管理
- 🎯 **专业级**: 支持混合精度、梯度检查点、分布式训练
- 🔄 **可扩展**: 模块化设计，易于扩展新功能

## 🏗️ 项目结构

```
P2W/
├── configs/                 # 配置文件
├── models/                  # Foundation models 存储
├── src/                     # 源代码
│   ├── core/               # 核心训练模块
│   ├── data/               # 数据处理
│   ├── utils/              # 工具函数
│   └── adapters/           # 数据适配器
├── scripts/                # 执行脚本
├── experiments/            # 实验管理
├── logs/                   # 日志文件
├── checkpoints/            # 检查点
└── outputs/                # 输出结果
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements_modern.txt
```

或者安装开发版本：

```bash
pip install -e .
```

### 2. 下载模型

```bash
python scripts/model_manager.py --action download --model Qwen/Qwen2.5-0.5B
```

### 3. 配置训练

编辑 `configs/training_config.yaml`：

```yaml
model:
  name: "Qwen/Qwen2.5-0.5B"
  cache_dir: "./models"

lora:
  r: 8
  alpha: 16
  dropout: 0.05

training:
  output_dir: "./checkpoints/qwen2.5_lora"
  num_train_epochs: 3
  per_device_train_batch_size: 4
  learning_rate: 2.0e-4
  run_name: "qwen2.5_lora_exp1"
```

### 4. 开始训练

```bash
python scripts/train_lora.py --config configs/training_config.yaml --dataset your_dataset
```

## 📊 训练监控

本框架集成了多种监控工具：

### WandB 监控
- 实时loss曲线
- 学习率变化
- 梯度分布
- 系统资源使用

### TensorBoard
- 标量指标
- 模型结构
- 嵌入可视化

### 日志文件
- 详细的训练日志
- 错误信息记录
- 性能指标

## 🔧 高级功能

### 分布式训练
```bash
accelerate launch --multi_gpu scripts/train_lora.py
```

### 混合精度训练
```yaml
training:
  fp16: true
  gradient_checkpointing: true
```

### 量化支持
```yaml
model:
  load_in_8bit: true
  # 或者
  load_in_4bit: true
```

## 📈 实验管理

每个实验都会在 `experiments/` 目录下创建独立的文件夹：

```
experiments/
├── qwen2.5_lora_exp1/
│   ├── config.yaml      # 实验配置
│   ├── results/         # 训练结果
│   ├── checkpoints/     # 检查点
│   └── logs/           # 日志
└── qwen2.5_lora_exp2/
    └── ...
```

## 🎯 支持的模型

- Qwen2.5 (0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B)
- Llama2/Llama3 系列
- Mistral 系列
- Phi 系列
- 其他基于Transformers的模型

## 📝 配置说明

### 模型配置
```yaml
model:
  name: "Qwen/Qwen2.5-0.5B"
  cache_dir: "./models"
  torch_dtype: "auto"
  device_map: "auto"
```

### LoRA配置
```yaml
lora:
  r: 8                    # LoRA秩
  alpha: 16               # LoRA缩放因子
  dropout: 0.05           # Dropout率
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

### 训练配置
```yaml
training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  learning_rate: 2.0e-4
  warmup_steps: 100
  evaluation_strategy: "steps"
  logging_steps: 10
```

## 🔍 最佳实践

1. **模型选择**: 从小模型开始（如Qwen2.5-0.5B）
2. **批量大小**: 根据显存调整batch_size
3. **学习率**: LoRA推荐2e-4到5e-4
4. **监控**: 使用WandB追踪实验
5. **检查点**: 定期保存检查点
6. **评估**: 设置合适的评估间隔

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

## 🙏 致谢

感谢以下开源项目：
- Hugging Face Transformers
- PEFT
- Accelerate
- WandB
- PyTorch Lightning
