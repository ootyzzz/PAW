#!/usr/bin/env python3
"""
测试脚本：验证警告屏蔽是否生效
"""

import warnings
import os

# 屏蔽 Transformers 警告
warnings.filterwarnings("ignore", message=".*cache_implementation.*")
warnings.filterwarnings("ignore", message=".*generation flags are not valid.*")
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# 测试导入
try:
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
    print("✅ 成功设置 Transformers 日志级别为 ERROR")
except ImportError:
    print("⚠️ 未找到 transformers 库")

print("✅ 警告屏蔽配置完成")
print("🔕 以下类型的警告将被屏蔽:")
print("   - cache_implementation 相关警告")
print("   - generation flags are not valid 相关警告")
print("   - 其他 transformers 详细日志信息")
