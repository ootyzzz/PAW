"""
模型加载模块
包含模型加载相关的工具函数
"""

from .config import *


def load_model_for_eval(model_path, device="auto", **kwargs):
    """加载模型并准备好评估"""
    print(f"⏳ 加载模型 {model_path}...")
    
    # 确定设备
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 检查是否是本地路径
    is_local_path = os.path.exists(model_path)
    
    try:
        # 环境变量设置，避免tokenizer警告和优化
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        # 加载tokenizer (特别处理本地模型路径)
        tokenizer_kwargs = {"trust_remote_code": True}
        if is_local_path:
            tokenizer_kwargs["local_files_only"] = True
            
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
        except Exception as e:
            print(f"警告: 标准tokenizer加载失败，尝试使用预训练tokenizer: {e}")
            tokenizer_kwargs["use_fast"] = False
            tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
        
        # 模型名称转小写用于判断
        model_name_lower = model_path.lower() if isinstance(model_path, str) else ""
        
        # 根据模型类型设置不同的加载参数
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            "device_map": device,
            "trust_remote_code": True,
        }
        
        # 添加本地文件参数
        if is_local_path:
            model_kwargs["local_files_only"] = True
        
        # Gemma模型需要特殊处理
        if "gemma" in model_name_lower:
            print("🦙 检测到Gemma模型，使用特殊加载设置")
            model_kwargs.update({
                "attn_implementation": "eager",  # 避免使用flash attention
            })
        
        # 检查是否为LoRA微调模型
        adapter_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_path):
            print("🔍 检测到LoRA适配器配置")
            # 加载基础模型
            base_model_path = kwargs.get('base_model_path')
            if not base_model_path:
                # 尝试从adapter_config.json中找到基础模型
                try:
                    with open(adapter_path, 'r') as f:
                        adapter_config = json.load(f)
                    base_model_path = adapter_config.get('base_model_name_or_path')
                    print(f"📄 从adapter_config.json获取到基础模型: {base_model_path}")
                except Exception as e:
                    raise ValueError(f"LoRA模型需要指定base_model_path参数: {e}")
            
            # 检查基础模型路径
            if not os.path.exists(base_model_path) and "/" not in base_model_path:
                # 可能是相对路径，尝试autodl-tmp中的常见位置
                for prefix in ["/root/autodl-tmp/models/", "/root/autodl-tmp/"]:
                    test_path = f"{prefix}{base_model_path}"
                    if os.path.exists(test_path):
                        base_model_path = test_path
                        print(f"🔍 定位到基础模型: {base_model_path}")
                        break
            
            # 加载基础模型
            print(f"🔄 加载基础模型: {base_model_path}")
            
            # 特殊处理本地基础模型
            base_kwargs = model_kwargs.copy()
            if os.path.exists(base_model_path):
                base_kwargs["local_files_only"] = True
                
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                **base_kwargs
            )
            
            # 加载LoRA权重
            print(f"🔄 加载LoRA权重: {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
        else:
            # 直接加载模型
            print(f"🔄 加载标准模型: {model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
        
        # 将模型置于评估模式
        model.eval()
        
        print(f"✅ 模型 {model_path} 加载完成")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        traceback.print_exc()
        raise
