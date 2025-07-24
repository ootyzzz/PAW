# Lightning + SwanLab Setup

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