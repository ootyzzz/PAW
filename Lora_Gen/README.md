# LoRA Parameter Generator

基于 HyperConv 的 LoRA 参数生成器，使用 PyTorch Lightning + SwanLab 训练框架。

## 📁 项目结构

```
Lora_Gen/
├── core/                           # 核心模块
│   ├── hyperconv_decoder.py       # HyperConv 解码器
│   ├── generator.py                # LoRA 参数生成器
│   ├── lightning_module.py         # Lightning 训练模块  
│   ├── data_module.py              # 数据模块
│   └── utils/                      # 工具函数
├── config/                         # 配置文件
│   └── generator_config.yaml       # 默认配置
├── data/                           # 数据目录
│   ├── train_prompts.jsonl         # 训练 prompts
│   └── val_prompts.jsonl           # 验证 prompts
├── experiments/                    # 实验结果
│   └── lora_generator_YYYYMMDD_HHMMSS/
│       ├── checkpoints/            # Lightning checkpoints
│       ├── tensorboard_logs/       # TensorBoard 日志
│       ├── swanlab_logs/           # SwanLab 日志
│       ├── results/                # 最终结果
│       └── config.yaml             # 实验配置
├── scripts/                        # 工具脚本
│   └── prepare_data.py             # 数据准备工具
├── train_generator.py              # 训练脚本
├── inference.py                    # 推理脚本
└── README.md                       # 本文档
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install pytorch-lightning>=2.0.0
pip install swanlab>=0.3.0
pip install sentence-transformers
pip install transformers>=4.20.0
pip install peft
pip install PyYAML
```

### 2. 数据准备

```bash
# 准备训练数据
python scripts/prepare_data.py prepare \
    --source data_to_lora/cs/arc-challenge/arc-challenge_train_formatted.jsonl \
    --output_dir Lora_Gen/data \
    --samples_per_prompt 4 \
    --test_ratio 0.1

# 分析checkpoint文件
python scripts/prepare_data.py analyze \
    --checkpoint_dir runs/arc-challenge_lora_20250721_005053/checkpoints
```

### 3. 训练模型

```bash
# 基础训练
python train_generator.py --config config/generator_config.yaml

# 指定checkpoint目录
python train_generator.py \
    --config config/generator_config.yaml \
    --checkpoint_dir runs/arc-challenge_lora_20250721_005053/checkpoints

# 干运行（验证配置）
python train_generator.py --config config/generator_config.yaml --dry_run
```

### 4. 模型推理

```bash
# 单个prompt推理
python inference.py \
    --model experiments/lora_generator_xxx/results/final_model/generator_model.pt \
    --prompt "Question: What is the capital of France?"

# 批量推理
python inference.py \
    --model experiments/lora_generator_xxx/results/final_model/generator_model.pt \
    --prompt_file test_prompts.txt \
    --output results/generated_params.pt
```

## 🏗️ 架构设计

### 模型架构

```
Input Prompts
    ↓
Text Encoder (Sentence-BERT)
    ↓
Input Projection
    ↓
Sequence Expander  
    ↓
HyperConv Block 1 (ConvW → ConvH → ConvL)
    ↓
HyperConv Block 2 (ConvW → ConvH → ConvL)
    ↓  
HyperConv Block 3 (ConvW → ConvH → ConvL)
    ↓
Output Projection
    ↓
Tokenized LoRA Parameters [B, 512, 384]
```

### 训练流程

1. **数据准备**: 将 ARC Challenge 数据随机组合成 non-overlapping prompts
2. **Checkpoint Tokenization**: 将 50 个 LoRA checkpoints tokenize 成固定格式
3. **监督训练**: 使用 MSE loss 训练生成器生成目标参数tokens
4. **实验管理**: Lightning + SwanLab 自动记录训练过程和指标

### 损失函数

```python
# MSE Loss (主要损失)
mse_loss = F.mse_loss(generated_params, target_params)

# L1 Loss (稀疏性正则化) 
l1_loss = F.l1_loss(generated_params, target_params)

# 总损失
total_loss = mse_weight * mse_loss + l1_weight * l1_loss
```

## ⚙️ 配置说明

### 模型配置

```yaml
model:
  text_encoder_name: "all-MiniLM-L6-v2"  # Sentence-BERT模型
  hidden_dim: 384                         # 隐藏层维度
  max_seq_len: 512                        # 最大序列长度
  num_hyperconv_blocks: 3                 # HyperConv块数量
  output_dim: 384                         # 输出维度
  freeze_text_encoder: true               # 冻结文本编码器
```

### 训练配置

```yaml
training:
  max_epochs: 100
  learning_rate: 1e-4
  weight_decay: 0.01
  mse_weight: 1.0
  l1_weight: 0.1
  optimizer_type: "adamw"
  scheduler_type: "cosine"
```

### 数据配置

```yaml
data:
  checkpoint_dir: "runs/arc-challenge_lora_20250721_005053/checkpoints"
  batch_size: 8
  samples_per_prompt: 4
  max_checkpoints: 50
  cache_tokenized: true
```

## 📊 SwanLab 集成

### 个人使用

```bash
# 默认使用个人项目（自动生成项目名）
python train_generator.py --config config/generator_config.yaml
```

### 团队协作

```yaml
# 修改配置文件
logging:
  swanlab:
    project: "team-lora-experiments"    # 团队项目名
    workspace: "your-team-workspace"    # 团队工作区
```

或使用环境变量：

```bash
export SWANLAB_PROJECT="team-lora-experiments"
export SWANLAB_WORKSPACE="your-team-workspace"
python train_generator.py --config config/generator_config.yaml
```

## 🔧 工具脚本

### 数据分析

```bash
# 分析checkpoint统计信息
python scripts/prepare_data.py analyze --checkpoint_dir your_checkpoint_dir

# 测试tokenization
python scripts/prepare_data.py test \
    --checkpoint_file your_checkpoint.ckpt \
    --max_tokens 512 \
    --token_dim 384
```

### 批量训练

```bash
# 训练多个配置
for config in config/*.yaml; do
    echo "训练配置: $config"
    python train_generator.py --config $config
done
```

## 📈 实验管理

### 实验目录结构

每个实验会自动创建独立的目录：

```
experiments/lora_generator_20250721_143052/
├── checkpoints/                    # Lightning检查点
│   ├── epoch=10-val_loss=0.1234.ckpt
│   └── last.ckpt
├── tensorboard_logs/               # TensorBoard日志
├── swanlab_logs/                   # SwanLab日志  
├── results/                        # 最终结果
│   └── final_model/
│       └── generator_model.pt
└── config.yaml                     # 实验配置快照
```

### 指标监控

训练过程中自动记录的指标：

- **Loss**: MSE loss, L1 loss, Total loss
- **Metrics**: MAE, RMSE, Relative error
- **Learning**: Learning rate, Epoch, Step
- **Hardware**: GPU utilization, Memory usage

## 🎯 使用案例

### 案例1: 基础训练

```bash
# 1. 准备数据
python scripts/prepare_data.py prepare \
    --source data_to_lora/cs/arc-challenge/arc-challenge_train_formatted.jsonl \
    --output_dir Lora_Gen/data

# 2. 训练模型
python train_generator.py --config config/generator_config.yaml

# 3. 推理测试
python inference.py \
    --model experiments/lora_generator_xxx/results/final_model/generator_model.pt \
    --prompt "Question: What is photosynthesis?"
```

### 案例2: 自定义配置

```bash
# 1. 复制默认配置
cp config/generator_config.yaml config/my_config.yaml

# 2. 修改配置
# 编辑 my_config.yaml，调整超参数

# 3. 训练
python train_generator.py --config config/my_config.yaml
```

### 案例3: 团队协作

```yaml
# team_config.yaml
logging:
  swanlab:
    project: "lora-research-2025"
    workspace: "ai-lab"
    experiment_name: "hyperconv-v1"
```

```bash
python train_generator.py --config team_config.yaml
```

## ⚠️ 注意事项

1. **内存使用**: 每个checkpoint约1GB，建议使用缓存机制
2. **计算资源**: 推荐使用GPU训练，CPU训练会很慢
3. **数据质量**: 确保prompts和checkpoints匹配且有效
4. **实验管理**: 建议为不同实验使用描述性的配置文件名

## 🐛 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减小batch size
   # 在配置文件中设置: batch_size: 4
   ```

2. **Checkpoint加载失败**
   ```bash
   # 检查checkpoint格式
   python scripts/prepare_data.py test --checkpoint_file your_file.ckpt
   ```

3. **SwanLab连接问题**
   ```bash
   # 检查网络连接和API key
   swanlab login
   ```

## 📚 相关论文

- HyperConv: [论文链接]
- LoRA: Low-Rank Adaptation of Large Language Models
- Parameter Generation for Few-shot Learning

## 🤝 贡献

欢迎提交 Issues 和 Pull Requests！

## 📄 许可证

MIT License
