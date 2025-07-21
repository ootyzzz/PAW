# 📁 目录结构对比：传统 vs Lightning + SwanLab

## 🔄 保存方式演进

### 传统方式 (experiments/)
```
./experiments/
├── icoding/
│   └── arc-challenge/
│       └── icoding_20250720_143052_lora/
│           ├── models/              # 训练模型保存
│           ├── checkpoints/         # 自定义checkpoint格式
│           └── logs/               # 日志和batch追踪
│               ├── batch_tracking_arc-challenge.jsonl
│               ├── checkpoint_mapping_arc-challenge.json
│               └── line_checkpoint_mapping_arc-challenge.json
```

### Lightning + SwanLab 方式 (runs/)
```
./runs/
├── arc-challenge_lora_20250720_143052/     # 清晰的实验命名
│   ├── checkpoints/                        # Lightning标准checkpoint (.ckpt)
│   │   ├── checkpoint-step-076.ckpt       # PyTorch Lightning格式
│   │   ├── checkpoint-step-077.ckpt       # 支持断点续训
│   │   ├── ...                            
│   │   └── checkpoint-step-125.ckpt       # 后50步，每步一个
│   ├── tensorboard_logs/                  # TensorBoard标准格式
│   │   └── events.out.tfevents.1234567890
│   ├── swanlab_logs/                      # SwanLab实验数据
│   │   ├── swanlab_metadata.json          # 实验元数据
│   │   └── logs/                          # 训练日志
│   ├── final_model/                       # HuggingFace标准格式
│   │   ├── config.json                    # 模型配置
│   │   ├── pytorch_model.bin              # 模型权重
│   │   ├── tokenizer.json                 # 分词器
│   │   ├── tokenizer_config.json          
│   │   ├── special_tokens_map.json        
│   │   └── ...                            
│   └── config.yaml                        # 完整实验配置记录
└── swanlab_workspace/                     # 全局SwanLab工作区
    ├── projects/
    └── cache/
```

## 🎯 主要改进

### 1. **目录命名更清晰**
- **传统**: `icoding/arc-challenge/icoding_20250720_143052_lora/`
- **Lightning**: `arc-challenge_lora_20250720_143052/`
- ✅ 更简洁，直接表达数据集和时间

### 2. **标准化格式**
| 文件类型 | 传统格式 | Lightning格式 | 优势 |
|----------|----------|---------------|------|
| Checkpoint | 自定义格式 | `.ckpt` (Lightning标准) | 断点续训、兼容性 |
| 日志 | JSON/JSONL | TensorBoard + SwanLab | 可视化、标准工具 |
| 最终模型 | 混合格式 | HuggingFace标准 | 易于分享、部署 |
| 配置 | 分散保存 | 统一config.yaml | 完整记录、可复现 |

### 3. **工具兼容性**
```powershell
# TensorBoard可视化
tensorboard --logdir ./runs/arc-challenge_lora_20250720_143052/tensorboard_logs

# HuggingFace Hub上传
huggingface-cli upload ./runs/arc-challenge_lora_20250720_143052/final_model

# Lightning断点续训
python train.py --resume_from_checkpoint ./runs/arc-challenge_lora_20250720_143052/checkpoints/checkpoint-step-100.ckpt
```

## 📊 存储空间对比

### 传统方式
```
models/              ~500MB     # LoRA权重
checkpoints/         ~25GB      # 50个检查点 × ~500MB
logs/               ~50MB      # JSON追踪文件
--------------------------------------------
总计:               ~25.5GB
```

### Lightning方式
```
final_model/         ~500MB     # HuggingFace格式
checkpoints/         ~25GB      # 50个.ckpt文件
tensorboard_logs/    ~10MB      # 二进制日志
swanlab_logs/        ~20MB      # 实验数据
config.yaml          ~5KB       # 配置记录
--------------------------------------------
总计:               ~25.5GB     # 相似，但更规范
```

## 🔄 迁移建议

### 保持并存
建议保留两种方式并存，根据需求选择：

```powershell
# 需要详细batch追踪 → 使用传统版本
python train_cs_lora_icoding.py --dataset arc-challenge --track_batches

# 需要现代化训练和监控 → 使用Lightning版本
python train_cs_lora_lightning.py --dataset arc-challenge
```

### 批量实验对比
```powershell
# 传统批量训练
foreach ($dataset in @("arc-challenge", "arc-easy")) {
    python train_cs_lora_icoding.py --dataset $dataset --track_batches
}

# Lightning批量训练
foreach ($dataset in @("arc-challenge", "arc-easy")) {
    python train_cs_lora_lightning.py --dataset $dataset
}
```

## 🎨 SwanLab vs 传统日志

### 传统监控方式
```powershell
# 查看训练进度
tail -f ./experiments/icoding/arc-challenge/*/logs/batch_tracking_*.jsonl

# 分析checkpoint映射
cat ./experiments/icoding/arc-challenge/*/logs/checkpoint_mapping_*.json | jq '.'
```

### SwanLab监控方式
```python
# 实时Web界面
# 自动打开 http://localhost:5092
# 或访问 https://swanlab.cn

# 特性:
✅ 实时训练曲线
✅ 超参数对比
✅ 实验版本管理
✅ 模型性能分析
✅ 协作和分享
```

## 🚀 未来规划

### 短期 (下个版本)
- [ ] 在Lightning版本中恢复batch追踪功能
- [ ] 提供传统→Lightning格式转换工具
- [ ] 支持从传统checkpoint恢复

### 中期
- [ ] 支持更多监控平台 (MLflow, W&B)
- [ ] 自动实验对比和报告生成
- [ ] 分布式训练优化

### 长期
- [ ] 统一两种方式的优点
- [ ] 云端实验管理
- [ ] 自动化超参数搜索

## 💡 推荐使用场景

| 场景 | 推荐版本 | 原因 |
|------|----------|------|
| 研究阶段，需要详细追踪 | 传统版本 | batch-level追踪，精确映射 |
| 生产训练，需要监控 | Lightning版本 | 现代化，可视化，稳定 |
| 多人协作 | Lightning版本 | 标准格式，易于分享 |
| 大规模训练 | Lightning版本 | 分布式，混合精度 |
| 快速实验 | Lightning版本 | 自动化程度高 |
| 调试训练过程 | 传统版本 | 详细的内部状态记录 |

## 🔧 配置迁移

如需从传统配置迁移到Lightning配置：

```yaml
# 传统 training_config.yaml
training:
  per_device_train_batch_size: 32
  max_steps: 125
  stage1:
    steps: 75
    learning_rate: 1.0e-4

# Lightning lightning_config.yaml  
training:
  batch_size: 32          # 简化命名
  max_steps: 125
  stage1_steps: 75        # 扁平化结构
  learning_rate_stage1: 1.0e-4
```

总体而言，Lightning + SwanLab 版本代表了更现代化、标准化的训练方式，而传统版本在某些特定需求（如精确追踪）方面仍有优势。建议根据具体需求选择合适的版本。
