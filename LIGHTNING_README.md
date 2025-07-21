# Lightning + SwanLab 版本安装和使用指南

## 📦 安装依赖

在原有环境基础上，安装额外的Lightning和SwanLab依赖：

```powershell
# 安装Lightning框架
pip install pytorch-lightning>=2.0.0 lightning>=2.0.0

# 安装SwanLab（国产实验管理平台）
pip install swanlab>=0.3.0

# 安装其他依赖
pip install tensorboard>=2.10.0 torchmetrics>=0.11.0

# 或者一次性安装
pip install -r requirements_lightning.txt
```

## 🚀 使用方法

### 基本命令（与原版完全兼容）

```powershell
# 单个数据集训练
python train_cs_lora_lightning.py --dataset arc-challenge

# 自定义batch size
python train_cs_lora_lightning.py --dataset arc-challenge --batch_size 16

# 测试模式
python train_cs_lora_lightning.py --dataset arc-challenge --test_mode

# 干运行
python train_cs_lora_lightning.py --dataset arc-challenge --dry_run
```

### 批量训练

```powershell
# PowerShell批量执行
foreach ($dataset in @("arc-challenge", "arc-easy", "boolq", "hellaswag", "openbookqa", "piqa", "winogrande")) {
    Write-Host "🚀 开始训练 $dataset..."
    python train_cs_lora_lightning.py --dataset $dataset
    Write-Host "✅ $dataset 训练完成"
}
```

## 📁 Lightning + SwanLab 目录结构

相比原来的 `./experiments/` 目录，新版本使用更标准的目录结构：

```
./runs/                                    # 主实验目录
├── arc-challenge_lora_20250720_143052/    # 单个实验目录
│   ├── checkpoints/                       # Lightning检查点 (.ckpt格式)
│   │   ├── checkpoint-step-076.ckpt      # 第76步检查点
│   │   ├── checkpoint-step-077.ckpt      # 第77步检查点
│   │   ├── ...
│   │   └── checkpoint-step-125.ckpt      # 最后一步检查点
│   ├── tensorboard_logs/                 # TensorBoard日志
│   │   └── events.out.tfevents.*
│   ├── swanlab_logs/                     # SwanLab日志和可视化
│   │   ├── swanlab_metadata.json
│   │   └── logs/
│   ├── final_model/                      # 最终HuggingFace格式模型
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   ├── tokenizer.json
│   │   └── ...
│   └── config.yaml                       # 完整实验配置
└── swanlab_workspace/                     # SwanLab工作区（全局）
```

## 🎯 主要优势

### 1. **现代化框架**
- ✅ PyTorch Lightning: 自动处理设备分配、分布式训练、混合精度
- ✅ SwanLab: 国产实验管理平台，支持中文，界面友好
- ✅ 代码更简洁，样板代码更少

### 2. **实时监控**
- 📊 训练指标实时可视化（loss、learning rate、stage等）
- 📈 支持TensorBoard和SwanLab双重日志
- 🔍 实验对比和版本管理

### 3. **标准化保存**
- 🗂️ 遵循社区最佳实践的目录结构
- 💾 Lightning checkpoint格式，支持断点续训
- 🤗 HuggingFace兼容的最终模型格式

### 4. **兼容性**
- ✅ 命令行参数与原版完全兼容
- ✅ 使用相同的数据路径和配置文件
- ✅ 保持相同的125步训练策略

### 5. **扩展性**
- 🔧 易于添加新的回调和监控
- 🚀 天然支持多GPU和分布式训练
- 📊 易于集成其他监控工具

## 🔧 配置说明

### SwanLab 配置
首次使用需要注册SwanLab账号：
```powershell
# 登录SwanLab（可选，也可以匿名使用）
swanlab login
```

### 检查点策略
- **Steps 1-75**: 不保存检查点（Stage 1高学习率阶段）
- **Steps 76-125**: 每步保存检查点（Stage 2低学习率阶段）
- 最终模型保存为HuggingFace格式

### 学习率调度
- **Stage 1 (Steps 1-75)**: 1e-4
- **Stage 2 (Steps 76-125)**: 1e-5
- 自动切换，无需手动干预

## 📊 监控和可视化

### TensorBoard
```powershell
# 启动TensorBoard查看训练曲线
tensorboard --logdir ./runs/[experiment_name]/tensorboard_logs
```

### SwanLab Dashboard
训练时会自动生成SwanLab链接，可以在浏览器中实时查看：
- 训练loss曲线
- 学习率变化
- 阶段切换指示
- 系统资源使用情况

## 🆚 与原版对比

| 特性 | 原版 | Lightning版 |
|------|------|-------------|
| 框架 | 自定义训练循环 | PyTorch Lightning |
| 监控 | 命令行输出 | SwanLab + TensorBoard |
| 保存格式 | 自定义目录结构 | 标准化目录结构 |
| 断点续训 | 不支持 | 原生支持 |
| 分布式 | 需手动配置 | 自动支持 |
| 混合精度 | 需手动配置 | 自动支持 |
| Batch追踪 | 详细追踪 | 下版本支持 |

## ⚠️ 注意事项

1. **兼容性**: `--track_batches` 参数暂时被忽略，batch追踪功能将在下个版本实现
2. **数据路径**: 继续使用 `data_to_lora/cs/` 目录，无需修改现有数据
3. **依赖**: 需要额外安装Lightning和SwanLab
4. **存储**: Lightning版本生成的文件可能比原版多（因为包含更多元数据）

## 🚧 下个版本计划

- [ ] 恢复详细的batch追踪功能
- [ ] 支持更多的实验管理平台（MLflow、Weights & Biases等）
- [ ] 添加验证集评估
- [ ] 支持更多的调度策略
- [ ] 分布式训练优化
