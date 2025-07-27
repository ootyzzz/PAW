#!/usr/bin/env python3
"""
测试调试版本的评估脚本
用于验证增强的错误处理是否正常工作
"""

import os
import sys

# 添加路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, 'eval'))

def test_debug_eval():
    """测试调试版本的评估功能"""
    print("🧪 测试调试版本的评估功能")
    print("=" * 50)
    
    # 导入评估模块
    try:
        from eval.core.batch_eval import evaluate_models
        print("✅ 成功导入评估模块")
    except Exception as e:
        print(f"❌ 导入评估模块失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试一个存在的模型路径（如果有的话）
    test_models = [
        "../models/Meta-Llama-3-8B-Instruct",  # 服务器上的模型路径
        "../autodl-tmp/models/Qwen-Qwen2.5-0.5B",  # 本地模型路径
    ]
    
    # 找到第一个存在的模型进行测试
    existing_model = None
    for model_path in test_models:
        if os.path.exists(model_path):
            existing_model = model_path
            print(f"✅ 找到测试模型: {model_path}")
            break
    
    if not existing_model:
        print("⚠️ 未找到可用的测试模型，使用不存在的路径测试错误处理")
        existing_model = "/nonexistent/model/path"
    
    # 运行评估测试
    try:
        print(f"\n🔍 开始测试评估: {existing_model}")
        results = evaluate_models(
            models_list=[existing_model],
            dataset_name="arc-challenge",
            sample_ratio=0.01,  # 只使用1%的数据进行快速测试
            batch_size=2
        )
        
        print(f"\n✅ 评估测试完成")
        print(f"📊 结果: {results}")
        return True
        
    except Exception as e:
        print(f"\n❌ 评估测试失败: {e}")
        print(f"❌ 异常类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 开始调试测试...")
    success = test_debug_eval()
    
    if success:
        print("\n🎉 调试测试成功完成!")
    else:
        print("\n❌ 调试测试失败!")
    
    sys.exit(0 if success else 1)
