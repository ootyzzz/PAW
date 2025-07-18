# LoRA微调Qwen2.5-0.5B实施方案

## 目标
使用LoRA技术微调Qwen2.5-0.5B模型，数据集为commonsense数据，采用分阶段学习率策略。

## 数据准备
- [x] 数据集路径: `C:\Users\feifa\GitHub\P2W\raw_datasets\commonsense\cs_all_unbalanced.jsonl`
- [x] 数据格式: {"instruction": "...", "input": "...", "output": "..."}
- [x] 模型路径: `C:\Users\feifa\GitHub\P2W\models\Qwen-Qwen2.5-0.5B`

## 训练策略
### 学习率调度
- [ ] 前75步: lr = 1e-4, 不保存checkpoint
- [ ] 后50步: lr = 1e-5, 每步保存checkpoint
- [ ] 总计125步训练

### 技术组件
- [ ] 使用`scripts/experiment_manager.py`作为主要接口
- [ ] 使用`scripts/model_manager.py`管理模型加载
- [ ] 参考`lora/checkpoint_utils.py`实现checkpoint保存逻辑
- [ ] 基于`configs/training_config.yaml`配置训练参数

## 实施步骤

### 1. 数据处理组件
- [x] 创建`utils/data_processor.py`
  - [x] 实现commonsense数据集加载
  - [x] 数据格式转换和tokenization
  - [x] 数据验证和预处理

### 2. 自定义学习率调度器
- [x] 在`utils/scheduler.py`中添加`TwoStageScheduler`
  - [x] 前75步: 1e-4学习率
  - [x] 后50步: 1e-5学习率
  - [x] 平滑过渡机制

### 3. 增强checkpoint管理
- [x] 扩展`lora/checkpoint_utils.py`
  - [x] 添加条件保存逻辑
  - [x] 实现步骤计数器
  - [x] 优化存储管理

### 4. LoRA训练器集成
- [x] 更新`core/train.py`
  - [x] 集成两阶段训练逻辑
  - [x] 添加进度监控
  - [x] 实现动态checkpoint策略

### 5. 实验管理接口
- [x] 增强`scripts/experiment_manager.py`
  - [x] 添加commonsense实验配置
  - [x] 集成新的训练流程
  - [x] 实现实验监控和日志

### 6. 模型管理增强
- [x] 更新`scripts/model_manager.py`
  - [x] 优化Qwen2.5模型加载
  - [x] 添加LoRA配置管理
  - [x] 实现模型状态检查

### 7. 配置文件调整
- [x] 更新`configs/training_config.yaml`
  - [x] 设置commonsense数据集路径
  - [x] 配置两阶段学习率参数
  - [x] 优化LoRA和训练参数

### 8. 主训练脚本
- [x] 创建`train_commonsense_lora.py`
  - [x] 整合所有组件
  - [x] 实现完整训练流程
  - [x] 添加错误处理和日志

## 验证步骤
按顺序执行以下验证命令，确保每个步骤都通过：

### 1. 环境和依赖验证
```bash
# 运行完整环境验证
python validate_setup.py
```
**期望结果**: 所有验证项显示 ✅，生成 `validation_report.json`

### 2. 数据加载测试
```bash
# 仅验证数据和模型，不训练
python train_commonsense_lora.py --validate_only
```
**期望结果**: 
- 模型路径验证通过
- 数据格式验证通过  
- LoRA兼容性检查通过

### 3. 配置文件验证
```bash
# 检查配置文件语法和路径
python -c "import yaml; print('✅ 配置文件格式正确' if yaml.safe_load(open('configs/training_config.yaml')) else '❌')"
```
**期望结果**: 显示 `✅ 配置文件格式正确`

### 4. Dry Run测试（不实际训练）
```bash
# 执行完整流程但不训练
python train_commonsense_lora.py --dry_run --experiment_name "test_dry_run"
```
**期望结果**: 
- 实验创建成功
- 所有组件初始化正常
- 显示 "🏃 Dry run完成，未实际训练"

### 5. 小规模训练测试（可选）
如果前面步骤都通过，可以测试实际训练：
```bash
# 开始完整训练
python train_commonsense_lora.py --experiment_name "commonsense_lora_test"
```
**期望结果**:
- 训练开始并显示进度条
- 前75步不保存checkpoint
- 第76步开始每步保存checkpoint
- 训练完成后显示结果摘要

### 手动验证检查清单

#### 在运行训练前确认：
- [ ] GPU可用性: `python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"` 
- [ ] 磁盘空间: 确保至少有10GB可用空间（用于checkpoints）
- [ ] 内存: 确保有足够内存（推荐16GB+）

#### 在训练过程中监控：
- [ ] 查看日志文件: `./experiments/[实验名]/logs/training_*.log`
- [ ] 监控checkpoint保存: `./experiments/[实验名]/checkpoints/`
- [ ] 检查进度显示: 确保学习率在第76步从1e-4变为1e-5

#### 训练完成后验证：
- [ ] 检查最终模型: `./experiments/[实验名]/models/final_model/`
- [ ] 验证checkpoint数量: 应该有50个checkpoint文件（步骤75-124）
- [ ] 查看训练报告: `./experiments/[实验名]/results/training_report.json`

## 预期输出
训练成功完成后，你将得到以下文件和结果：

### 📁 文件结构
```
experiments/
└── [实验名称]/
    ├── checkpoints/           # Checkpoint文件
    │   ├── checkpoint_step_0075_*.pt
    │   ├── checkpoint_step_0076_*.pt
    │   ├── ...
    │   ├── checkpoint_step_0124_*.pt
    │   └── checkpoint_info.json
    ├── models/               # 最终模型
    │   └── final_model/
    │       ├── adapter_config.json
    │       ├── adapter_model.safetensors
    │       └── tokenizer files...
    ├── logs/                 # 训练日志
    │   └── training_*.log
    ├── results/              # 结果文件
    │   ├── training_results.json
    │   ├── training_report.json
    │   └── data_validation.json
    ├── config.yaml           # 实验配置
    └── metadata.json         # 实验元数据
```

### 📊 关键指标
- **训练步数**: 125步 (75步 + 50步)
- **Checkpoint文件**: 50个 (步骤75-124)
- **最终LoRA权重**: 保存在 `final_model/` 目录
- **训练日志**: 详细记录每步的损失和学习率变化

### 📋 验证成功的标志
- [x] 学习率在第76步从1e-4切换到1e-5
- [x] 前75步没有保存checkpoint
- [x] 后50步每步都保存checkpoint  
- [x] 训练过程无错误中断
- [x] 最终模型文件完整生成

---
*创建时间: 2025-07-18*
*状态: ✅ 实施完成，等待验证*

## 🚀 快速开始指南

### 第一步：环境验证
```bash
python validate_setup.py
```

### 第二步：测试运行
```bash
python train_commonsense_lora.py --validate_only
```

### 第三步：开始训练
```bash
python train_commonsense_lora.py --experiment_name "my_commonsense_lora"
```

### 第四步：监控进度
- 查看实时日志: `tail -f ./experiments/[实验名]/logs/training_*.log`
- 检查checkpoint: `ls -la ./experiments/[实验名]/checkpoints/`

**预计训练时间**: 约30-60分钟（取决于硬件）
**所需磁盘空间**: 约5-10GB（包括checkpoints）
