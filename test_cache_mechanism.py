#!/usr/bin/env python3
"""
测试全局缓存机制
验证模型是否会被重复加载
"""

import os
import sys

# 添加路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, 'eval'))

def test_cache_mechanism():
    """测试缓存机制"""
    print("🧪 测试全局缓存机制")
    print("=" * 50)
    
    # 导入评估模块
    try:
        from eval.core.evaluator import LightningModelEvaluator, _MODEL_CACHE
        print("✅ 成功导入评估模块")
        print(f"🔍 初始缓存状态: {len(_MODEL_CACHE)} 个模型")
    except Exception as e:
        print(f"❌ 导入评估模块失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试模型路径
    test_model_path = "../models/Meta-Llama-3-8B-Instruct"
    
    if not os.path.exists(test_model_path):
        print(f"⚠️ 测试模型不存在: {test_model_path}")
        print("🔍 尝试使用虚拟路径测试缓存逻辑...")
        test_model_path = "/nonexistent/test/model"
    
    try:
        print(f"\n🔍 第一次创建评估器: {test_model_path}")
        evaluator1 = LightningModelEvaluator(test_model_path)
        print(f"✅ 第一次创建成功")
        print(f"🔍 缓存状态: {len(_MODEL_CACHE)} 个模型")
        
        print(f"\n🔍 第二次创建评估器: {test_model_path}")
        evaluator2 = LightningModelEvaluator(test_model_path)
        print(f"✅ 第二次创建成功")
        print(f"🔍 缓存状态: {len(_MODEL_CACHE)} 个模型")
        
        print(f"\n🔍 第三次创建评估器: {test_model_path}")
        evaluator3 = LightningModelEvaluator(test_model_path)
        print(f"✅ 第三次创建成功")
        print(f"🔍 缓存状态: {len(_MODEL_CACHE)} 个模型")
        
        # 检查缓存键
        cache_keys = list(_MODEL_CACHE.keys())
        print(f"\n🔍 缓存键列表: {cache_keys}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 缓存测试失败: {e}")
        print(f"❌ 异常类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 开始缓存机制测试...")
    success = test_cache_mechanism()
    
    if success:
        print("\n🎉 缓存机制测试完成!")
    else:
        print("\n❌ 缓存机制测试失败!")
    
    sys.exit(0 if success else 1)
