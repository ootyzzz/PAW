# P2W 项目环境设置

## 📁 环境设置文件夹说明

这个文件夹包含了所有与Python和CUDA环境设置相关的工具和脚本。

## 📋 文件列表

### 🔧 主要工具

- **`validate_cuda.py`** - CUDA环境完整验证脚本
  - 检查环境一致性
  - 验证CUDA和PyTorch配置
  - 测试RTX 5060 GPU功能

- **`start_work.bat`** - 一键启动工作环境
  - 自动激活conda环境
  - 运行环境验证
  - 准备工作环境

- **`quick_check.bat`** - 快速环境检查
  - 简单的环境状态检查
  - 适合日常快速验证

## 🚀 使用方法

### 每日工作流程

1. **双击 `start_work.bat`** - 一键启动完整环境
2. 或者在项目根目录运行：`python setup_env\validate_cuda.py`

### 快速检查

双击 `quick_check.bat` 或运行：
```bash
cd "C:\Users\feifa\GitHub\P2W"
python setup_env\validate_cuda.py
```

### 手动环境激活

如果遇到环境问题：
```bash
conda deactivate
conda activate cuda312
python setup_env\validate_cuda.py
```

## ✅ 验证通过标准

环境正确配置时，应该看到：
- ✅ 环境一致
- ✅ Python路径与conda环境匹配  
- ✅ 命令行python版本与脚本运行版本一致
- ✅ 检测到RTX 5060 GPU
- ✅ CUDA计算验证成功

## ⚠️ 常见问题

**环境不一致问题：**
```bash
conda deactivate && conda activate cuda312
```

**路径问题：**
确保在项目根目录 `C:\Users\feifa\GitHub\P2W` 运行脚本

**CUDA问题：**
检查NVIDIA驱动是否最新，PyTorch版本是否与CUDA兼容
