# PAW项目开发背景信息总结

## 🏗️ 项目架构概览

### **项目名称**: PAW (Parameter-Aware Weight Transfer)
**核心功能**: LoRA权重迁移和评估pipeline

### **主要目录结构**:
```
PAW/
├── pipeline/                    # 主要pipeline逻辑
│   ├── transfer_pipeline.py    # 主入口脚本
│   ├── config/                 # 配置文件
│   │   ├── pipeline_config.yaml    # 本地配置
│   │   ├── server.yaml             # 服务器配置
│   │   └── quick_test_config.yaml  # 快速测试配置
│   └── core/                   # 核心模块
├── eval/                       # 评估模块
│   ├── lightning_eval.py       # Lightning评估入口
│   └── core/                   # 评估核心逻辑
├── train_lora/                 # LoRA训练模块
├── lora_adapter/               # LoRA迁移模块
└── data_to_lora/              # 数据集
```

## 🔧 技术栈

### **核心框架**:
- **PyTorch Lightning**: 训练和评估框架
- **Transformers**: 模型加载和处理
- **PEFT**: LoRA适配器管理
- **SwanLab**: 实验跟踪

### **支持的模型**:
- **Qwen系列**: Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen2.5-7B
- **Llama系列**: Meta-Llama-3-8B-Instruct
- **Gemma系列**: gemma-2-2b-it

### **数据集**:
- **arc-challenge**: 主要测试数据集
- **arc-easy, piqa, hellaswag, winogrande**: 其他支持的数据集

## 🌐 环境配置

### **本地环境**:
- **硬件**: 4090 GPU
- **环境**: conda activate cuda312
- **用途**: 开发和快速测试

### **服务器环境**:
- **硬件**: 4×A800, 320GB总显存
- **环境**: conda activate dl_env
- **用途**: 大规模训练和评估

### **开发流程**:
1. 本地开发 → 2. 本地测试 → 3. Push到GitHub → 4. 服务器Pull → 5. 服务器运行

## 📊 Pipeline工作流程

### **6步完整流程**:
1. **EVAL SOURCE BASE MODEL**: 评估源基础模型
2. **TRAIN SOURCE LORA**: 训练源模型的LoRA
3. **TRANSFER LORA**: 迁移LoRA到目标模型
4. **EVAL TARGET BASE MODEL**: 评估目标基础模型
5. **EVAL TRANSFERRED LORA**: 评估迁移后的LoRA
6. **TRAIN TARGET LORA**: 训练目标模型的LoRA（对比基线）

### **配置参数**:
- **训练**: batch_size=2, max_steps=400, lr=1e-5
- **评估**: sample_ratio=1.0, batch_size=8
- **迁移**: similarity_threshold=0.0001

## 🚨 已知问题和解决方案

### **1. Progress Bar问题**:
- **现象**: Lightning默认进度条重复打印，刷屏严重
- **尝试方案**: 自定义QuietProgressCallback
- **状态**: 仍未完全解决

### **2. DataLoader兼容性**:
- **问题**: `trainer.test_dataloaders[0]` IndexError
- **原因**: Lightning在不同版本/配置下DataLoader格式不同
- **解决**: 添加兼容性检查

### **3. 模型加载**:
- **LoRA模型**: 需要base_model_path参数
- **路径问题**: 本地用`../autodl-tmp/models/`, 服务器用`../models/`
- **特殊处理**: Gemma模型需要特殊配置

### **4. 内存管理**:
- **GPU内存**: 使用torch.cuda.empty_cache()清理
- **模型缓存**: 全局缓存避免重复加载

## 📝 配置文件说明

### **pipeline_config.yaml** (本地):
```yaml
paths:
  models_dir: '../autodl-tmp/models'
training:
  default_batch_size: 4
  default_max_steps: 200
evaluation:
  sample_ratio: 0.05
default_experiment:
  source_model: 'gemma-2-2b-it'
  target_model: 'Qwen_Qwen2.5-1.5B'
```

### **server.yaml** (服务器):
```yaml
paths:
  models_dir: '../models'
training:
  default_batch_size: 2
  default_max_steps: 400
evaluation:
  sample_ratio: 1.0
default_experiment:
  source_model: 'Meta-Llama-3-8B-Instruct'
  target_model: 'Qwen2.5-7B-Instruct'
```

## 🎯 关键命令

### **本地快速测试**:
```bash
python transfer_pipeline.py --quick_test
```

### **服务器完整运行**:
```bash
python transfer_pipeline.py --config '/root/paddlejob/workspace/env_run/Projects/PAW/pipeline/config/server.yaml'
```

### **单独评估**:
```bash
python ./eval/lightning_eval.py --models_list ../models/MODEL_NAME --dataset arc-challenge --sample_ratio 1.0
```

## 🔍 调试和监控

### **日志系统**:
- **SwanLab**: 在线实验跟踪
- **本地日志**: 详细的调试输出
- **内存监控**: GPU和RAM使用情况

### **结果保存**:
- **JSON**: 详细结果数据
- **CSV**: 表格格式结果
- **Markdown**: 实验总结报告

这些信息应该能帮助你快速了解项目的整体架构和开发环境！