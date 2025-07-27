# 调试增强说明文档

## 概述

为了解决服务器上LoRA迁移管道评估失败的问题，我们对评估系统进行了全面的调试增强。本文档详细说明了所做的修改和如何使用这些增强功能。

## 问题背景

- **本地环境**: 4090, conda activate cuda312 - 运行正常
- **服务器环境**: 4*A800, 320G显存, conda activate dl_env - 评估阶段失败
- **错误现象**: 模型加载成功但评估时显示ERROR，缺乏详细错误信息

## 修改内容

### 0. 防重复加载机制 (`evaluator.py`)

添加了模型加载状态管理：
- `_model_loaded` 标志防止重复加载
- 模型和tokenizer的空值检查
- 跳过已加载模型的重复初始化

### 1. 增强的内存监控 (`evaluator.py`)

添加了 `log_memory_usage()` 函数，在关键步骤监控：
- GPU内存使用情况 (已分配/已保留/总容量)
- RAM内存使用情况 (已使用/总容量/使用率)

### 2. 详细异常处理装饰器 (`evaluator.py`)

添加了 `detailed_exception_handler` 装饰器，提供：
- 异常类型和详细信息
- 完整的Python traceback
- 异常发生时的系统状态

### 3. 模型加载过程详细日志 (`evaluator.py`)

#### 路径验证增强:
- 检查模型路径是否存在
- 显示绝对路径
- 列出模型目录内容
- 验证关键文件存在性

#### 分步加载日志:
- **LoRA模型**: 4个详细步骤
  1. 加载PEFT配置
  2. 加载tokenizer
  3. 加载基础模型
  4. 加载LoRA适配器

- **常规模型**: 3个详细步骤
  1. 加载tokenizer
  2. 准备模型参数
  3. 加载模型

#### 特殊模型处理:
- **Llama模型**: 特殊配置优化
- **Gemma模型**: 现有的特殊处理保持不变

### 4. 测试步骤详细监控 (`evaluator.py`)

在 `test_step()` 方法中添加：
- 每个batch的处理状态
- 损失和准确率计算的详细步骤
- batch内容验证
- 内存使用监控

### 5. 批量评估增强错误处理 (`batch_eval.py`)

在模型评估循环中添加：
- 评估器初始化的详细日志
- 异常时的系统状态检查
- 模型路径存在性验证
- GPU内存状态报告

## 使用方法

### 1. 直接运行原有管道

现在运行原有的管道命令会自动获得详细的调试信息：

```bash
# 服务器上运行
cd /root/PAW
python pipeline/transfer_pipeline.py
```

### 2. 使用测试脚本

我们提供了一个专门的测试脚本来验证调试功能：

```bash
cd /root/PAW
python test_debug_eval.py
```

### 3. 单独测试评估功能

```bash
cd /root/PAW
python eval/lightning_eval.py --models_list ../models/Meta-Llama-3-8B-Instruct --dataset arc-challenge --sample_ratio 0.05
```

## 预期输出

运行修改后的代码，你将看到：

### 模型加载阶段:
```
📦 开始加载模型: ../models/Meta-Llama-3-8B-Instruct
🔍 [模型加载前] GPU内存: 0.00GB / 0.00GB / 80.00GB
🔍 模型路径检查: ../models/Meta-Llama-3-8B-Instruct
🔍 绝对路径: /root/models/Meta-Llama-3-8B-Instruct
🔍 是否为本地路径: True
🔍 模型目录内容: ['config.json', 'pytorch_model-00001-of-00004.bin', ...]
✅ 找到关键文件: config.json
🔍 步骤1: 加载tokenizer...
🔍 [tokenizer加载前] GPU内存: 0.00GB / 0.00GB / 80.00GB
✅ tokenizer加载成功
🔍 [tokenizer加载后] GPU内存: 0.05GB / 0.10GB / 80.00GB
```

### 如果出现错误:
```
❌ 函数 _load_model 发生异常:
❌ 异常类型: RuntimeError
❌ 异常信息: CUDA out of memory. Tried to allocate 2.00 GiB
❌ 详细traceback:
Traceback (most recent call last):
  File "...", line ..., in wrapper
    return func(*args, **kwargs)
  ...
❌ 系统信息:
🔍 [异常发生时] GPU内存: 78.50GB / 80.00GB / 80.00GB
🔍 [异常发生时] RAM使用: 45.2GB / 64.0GB (70.6%)
```

## 故障排除指南

### 1. 内存不足问题
如果看到GPU内存接近满载，可以：
- 减少batch_size
- 使用更小的sample_ratio
- 检查是否有其他进程占用GPU

### 2. 模型加载失败
如果模型路径检查失败：
- 验证配置文件中的路径设置
- 检查模型文件完整性
- 确认权限设置

### 3. 兼容性问题
如果特定模型类型出现问题：
- 查看特殊模型处理日志
- 检查transformers版本兼容性
- 验证模型格式

## 配置文件

确保使用正确的配置文件：
- **本地**: `pipeline/config/pipeline_config.yaml`
- **服务器**: `pipeline/config/server.yaml`

## 下一步

运行修改后的代码，收集详细的错误信息，然后我们可以：
1. 根据具体错误类型制定针对性解决方案
2. 优化内存使用策略
3. 调整模型加载参数
4. 实施特定的兼容性修复

## 注意事项

- 调试日志会增加输出量，但不会显著影响性能
- 内存监控功能在没有GPU的环境中会自动跳过
- 所有原有功能保持不变，只是增加了调试信息
