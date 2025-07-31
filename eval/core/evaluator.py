"""
Lightning评估模块
包含LightningModelEvaluator类和相关评估逻辑
"""

from .config import *
import psutil
import gc


# 全局模型缓存，防止重复加载
_MODEL_CACHE = {}
_LOADING_LOCK = {}  # 加载锁，防止并发加载


def log_memory_usage(stage="", verbose=False):
    """记录内存使用情况 - 精简版"""
    if not verbose:
        return
    try:
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / 1024**3
            gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            ram_usage = psutil.virtual_memory()
            print(f"💾 [{stage}] GPU: {gpu_allocated:.1f}GB/{gpu_total:.1f}GB | RAM: {ram_usage.percent:.1f}%")
    except Exception:
        pass


def detailed_exception_handler(func):
    """详细异常处理装饰器"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"❌ 函数 {func.__name__} 发生异常:")
            print(f"❌ 异常类型: {type(e).__name__}")
            print(f"❌ 异常信息: {str(e)}")
            print(f"❌ 详细traceback:")
            traceback.print_exc()
            print(f"❌ 系统信息:")
            log_memory_usage("异常发生时")
            raise
    return wrapper


def analyze_lora_adapter(model_path):
    """分析LoRA适配器结构并生成信息卡片"""
    try:
        from peft import PeftConfig
        import json
        
        # 读取adapter配置
        config_path = Path(model_path) / "adapter_config.json"
        if not config_path.exists():
            return None
            
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 读取adapter模型文件获取权重信息
        adapter_model_path = Path(model_path) / "adapter_model.safetensors"
        if adapter_model_path.exists():
            try:
                from safetensors import safe_open
                with safe_open(adapter_model_path, framework="pt") as f:
                    keys = list(f.keys())
            except:
                keys = []
        else:
            keys = []
        
        # 分析权重结构
        component_stats = {
            'q_proj': {'layers': set()},
            'k_proj': {'layers': set()},
            'v_proj': {'layers': set()},
            'o_proj': {'layers': set()},
            'gate_proj': {'layers': set()},
            'up_proj': {'layers': set()},
            'down_proj': {'layers': set()}
        }
        
        # 统计实际存在的权重
        for key in keys:
            for comp in component_stats.keys():
                if comp in key:
                    import re
                    layer_match = re.search(r'layers\.(\d+)\.', key)
                    if layer_match:
                        layer_num = int(layer_match.group(1))
                        component_stats[comp]['layers'].add(layer_num)
        
        target_modules = config.get('target_modules', [])
        
        # 生成卡片
        print("\n" + "=" * 60)
        print("🎯 LoRA 适配器信息卡片")
        print("=" * 60)
        print(f"📁 路径: {Path(model_path).name}")
        print(f"🔧 LoRA rank: {config.get('r', 'N/A')}")
        print(f"🔧 LoRA alpha: {config.get('lora_alpha', 'N/A')}")
        print(f"🔧 目标模块: {', '.join(target_modules)}")
        print(f"🔧 基础模型: {config.get('base_model_name_or_path', 'N/A')}")
        
        print("\n📊 LoRA组件分布:")
        for comp, stats in component_stats.items():
            if comp in target_modules and stats['layers']:
                layers = sorted(list(stats['layers']))
                layer_count = len(layers)
                if layer_count > 0:
                    layer_range = f"{min(layers)}-{max(layers)} 共 {layer_count} 层" if len(layers) > 1 else f"第{layers[0]}层"
                    print(f"  {comp:>10}: {layer_range}")
        
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"⚠️ LoRA信息分析失败: {e}")
        return False


class LightningModelEvaluator(pl.LightningModule):
    """Lightning模型评估模块"""
    
    def __init__(self, model_path: str, base_model_path: str = None, max_length: int = 512):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_path = model_path
        self.base_model_path = base_model_path
        self.max_length = max_length
        
        # 创建模型名称用于报告
        self.model_name = Path(model_path).name
        
        # 模型加载状态标志
        self._model_loaded = False
        self.model = None
        self.tokenizer = None
        
        # 创建缓存键
        self.cache_key = f"{self.model_path}_{self.base_model_path or 'none'}"
        
        # 加载模型和tokenizer
        self._load_model()
    
    def setup(self, stage=None):
        """Lightning生命周期方法 - 确保模型已加载"""
        if not self._model_loaded:
            print(f"🔄 Lightning setup阶段重新加载模型...")
            self._load_model()
    
    def on_test_start(self):
        """测试开始时的钩子"""
        if not self._model_loaded:
            print(f"🔄 测试开始时重新加载模型...")
            self._load_model()
        
        # 初始化累积统计变量
        self._total_correct = 0
        self._total_samples = 0
        print(f"🚀 开始测试评估，将显示实时累积准确率...")

    def on_test_end(self):
        """测试结束时的钩子"""
        if hasattr(self, '_total_correct') and hasattr(self, '_total_samples'):
            if self._total_samples > 0:
                final_accuracy = self._total_correct / self._total_samples
                print(f"\n✅ 测试完成！")
                print(f"📊 最终累积准确率: {final_accuracy:.4f} ({self._total_correct}/{self._total_samples})")
                print(f"📊 正确样本数: {self._total_correct}")
                print(f"📊 总样本数: {self._total_samples}")
            else:
                print(f"\n⚠️ 测试完成，但没有有效样本")
        
    @detailed_exception_handler
    def _load_model(self):
        """加载模型和tokenizer"""
        # 创建缓存键
        cache_key = f"{self.model_path}_{self.base_model_path or 'none'}"
        
        # 检查全局缓存
        if cache_key in _MODEL_CACHE:
            print(f"✅ 从缓存加载模型: {Path(self.model_path).name}")
            cached_data = _MODEL_CACHE[cache_key]
            self.model = cached_data['model']
            self.tokenizer = cached_data['tokenizer']
            self._model_loaded = True
            return
        
        # 检查是否已经加载过模型
        if self._model_loaded and self.model is not None and self.tokenizer is not None:
            print(f"✅ 模型已加载: {Path(self.model_path).name}")
            return
        
        print(f"📦 加载模型: {Path(self.model_path).name}")
        
        # 检查模型路径
        is_local_path = os.path.exists(self.model_path)
        if not is_local_path:
            raise FileNotFoundError(f"模型路径不存在: {self.model_path}")
        
        try:
            # 模型加载参数
            load_kwargs = {
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "trust_remote_code": True,
                "use_cache": True,
                "device_map": None,  # 禁用自动设备映射，让Lightning控制
            }
            
            # 检查是否是LoRA模型
            config_path = Path(self.model_path) / "adapter_config.json"
            
            if config_path.exists():
                # 显示LoRA信息卡片
                analyze_lora_adapter(self.model_path)
                
                # LoRA模型加载流程
                print("🔧 LoRA模型加载中...")
                try:
                    # 加载PEFT配置获取基础模型信息
                    peft_config = PeftConfig.from_pretrained(self.model_path)
                    detected_base_model = peft_config.base_model_name_or_path
                    
                    # 使用提供的基础模型路径或检测到的路径
                    actual_base_model = self.base_model_path or detected_base_model
                    
                    # 确认基础模型路径
                    if not os.path.exists(actual_base_model) and "/" not in actual_base_model:
                        for prefix in ["/root/autodl-tmp/models/", "/root/autodl-tmp/"]:
                            test_path = f"{prefix}{actual_base_model}"
                            if os.path.exists(test_path):
                                actual_base_model = test_path
                                break
                    
                    # 加载tokenizer
                    tokenizer_kwargs = {"trust_remote_code": True}
                    self.tokenizer = AutoTokenizer.from_pretrained(actual_base_model, **tokenizer_kwargs)
                    
                    # 特殊处理Gemma模型
                    if "gemma" in actual_base_model.lower():
                        load_kwargs.update({
                            "attn_implementation": "eager",
                            "use_cache": False,
                            "_attn_implementation_internal": "eager"
                        })
                    
                    # 加载基础模型
                    base_model = AutoModelForCausalLM.from_pretrained(actual_base_model, **load_kwargs)
                    
                    # 过滤PEFT警告并加载PEFT模型
                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="Found missing adapter keys")
                        self.model = PeftModel.from_pretrained(base_model, self.model_path)
                    print(f"✅ LoRA模型加载完成")
                    
                except Exception as e:
                    raise RuntimeError(f"无法加载LoRA模型: {self.model_path}，错误: {e}")
            else:
                # 常规模型加载流程
                print("📦 常规模型加载中...")
                
                # 加载tokenizer
                tokenizer_kwargs = {"trust_remote_code": True}
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, **tokenizer_kwargs)
                except Exception as e:
                    tokenizer_kwargs["use_fast"] = False
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, **tokenizer_kwargs)
                
                # 针对特殊模型的处理
                model_name_lower = self.model_path.lower()
                if "gemma" in model_name_lower:
                    load_kwargs.update({
                        "attn_implementation": "eager",
                        "use_cache": False,
                        "_attn_implementation_internal": "eager"
                    })
                elif "llama" in model_name_lower:
                    load_kwargs.update({"use_cache": True})
                
                # 加载模型
                self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)
                print("✅ 常规模型加载完成")
        
            # 确保模型处于评估模式
            self.model.eval()
            
            # 手动将模型移动到第一个GPU（如果可用）
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                self.model = self.model.to(device)
            
            # 设置pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 设置加载完成标志
            self._model_loaded = True
            
            # 保存到全局缓存
            _MODEL_CACHE[cache_key] = {
                'model': self.model,
                'tokenizer': self.tokenizer
            }
            
        except Exception as e:
            print(f"❌ 模型加载失败: {self.model_path}")
            print(f"❌ 错误类型: {type(e).__name__}")
            print(f"❌ 错误信息: {str(e)}")
            print(f"❌ 详细错误:")
            traceback.print_exc()
            raise RuntimeError(f"无法加载模型 {self.model_path}: {str(e)}")

    def test_step(self, batch, batch_idx):
        """单个测试步骤"""
        try:
            # 计算指标
            loss = self._compute_loss(batch)
            accuracy = self._compute_accuracy(batch)
            perplexity = torch.exp(loss)
            batch_size = len(batch) if batch else 0
            
            # 累积准确率计算
            if not hasattr(self, '_total_correct'):
                self._total_correct = 0
                self._total_samples = 0
            
            # 更新累积统计
            if batch_size > 0:
                correct_count = int(accuracy.item() * batch_size)
                self._total_correct += correct_count
                self._total_samples += batch_size
                
                # 计算累积准确率
                cumulative_accuracy = self._total_correct / self._total_samples if self._total_samples > 0 else 0.0
                
                # 每50个batch显示一次进度和累积准确率
                if batch_idx % 50 == 0:
                    sample_info = ""
                    if batch and len(batch) > 0:
                        sample = batch[0]
                        if 'input' in sample:
                            sample_info = f" | 样本: {sample['input'][:50]}..."
                    print(f"📊 Step {batch_idx:4d} | 累积准确率: {cumulative_accuracy:.4f} ({self._total_correct}/{self._total_samples}){sample_info}")
                    log_memory_usage(f"step_{batch_idx}", verbose=False)
            
            # 验证batch内容
            if not batch:
                return {
                    'loss': torch.tensor(0.0),
                    'accuracy': torch.tensor(0.0),
                    'perplexity': torch.tensor(1.0),
                    'batch_size': 0
                }
            
            # 记录指标
            self.log('test/loss', loss, batch_size=batch_size)
            self.log('test/accuracy', accuracy, batch_size=batch_size)
            self.log('test/perplexity', perplexity, batch_size=batch_size)
            
            return {
                'loss': loss,
                'accuracy': accuracy,
                'perplexity': perplexity,
                'batch_size': batch_size
            }
        except Exception as e:
            print(f"❌ Step {batch_idx} 失败: {type(e).__name__}: {str(e)}")
            if batch_idx % 100 == 0:  # 只在关键步骤显示详细错误
                traceback.print_exc()
            
            return {
                'loss': torch.tensor(float('inf')),
                'accuracy': torch.tensor(0.0),
                'perplexity': torch.tensor(float('inf')),
                'batch_size': len(batch) if batch else 1
            }
        
    def _compute_loss(self, batch):
        """计算损失"""
        try:
            inputs = []
            labels = []
            
            for item in batch:
                # 处理多选题格式
                if 'input' in item and 'options' in item:
                    question = item['input']
                    options = item['options']
                    target = item.get('target', 'A')
                    
                    # 格式化问题、选项和答案
                    text = f"Question: {question}\n"
                    for option in options:
                        text += f"{option}\n"
                    text += f"Answer: {target}"
                else:
                    # 备选：使用任何文本字段
                    text = item.get('text', str(item))
                
                # Tokenize
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                inputs.append(encoding['input_ids'].squeeze())
                labels.append(encoding['input_ids'].squeeze())

            if inputs:
                # 确保所有tensor都在模型的主设备上
                model_device = next(self.model.parameters()).device
                input_ids = torch.stack(inputs).to(model_device)
                attention_mask = torch.ones_like(input_ids).to(model_device)
                labels = torch.stack(labels).to(model_device)
            else:
                return torch.tensor(0.0)
            
            # 计算损失
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            return outputs.loss
            
        except Exception as e:
            print(f"❌ _compute_loss失败: {e}")
            print(f"❌ batch大小: {len(batch) if batch else 'None'}")
            if batch:
                print(f"❌ 第一个样本: {batch[0] if len(batch) > 0 else 'Empty'}")
            traceback.print_exc()
            return torch.tensor(float('inf'))

    def _compute_accuracy(self, batch):
        """计算准确率 - 支持所有7个数据集格式"""
        if not isinstance(batch, list):
            return torch.tensor(0.5)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for item in batch:
                try:
                    # 解析数据项
                    question = item.get('input', '')
                    options = item.get('options', [])
                    correct_answer = item.get('target', '')
                    correct_idx = item.get('target_idx', 0)
                    dataset = item.get('dataset', '')
                    
                    if not options:
                        total += 1
                        continue
                    
                    # 根据数据集类型格式化提示词
                    if dataset in ['arc-challenge', 'arc-easy', 'openbookqa']:
                        # ARC和OpenBookQA：4选项，使用A/B/C/D格式
                        prompt = f"Question: {question}\n"
                        option_labels = ['A', 'B', 'C', 'D']
                        for i, option in enumerate(options):
                            if i < len(option_labels):
                                prompt += f"{option_labels[i]}. {option}\n"
                        prompt += "Answer:"
                        valid_answers = option_labels[:len(options)]
                        
                    elif dataset == 'boolq':
                        # BoolQ：True/False格式，映射到A/B
                        prompt = f"Question: {question}\n"
                        prompt += "A. False\n"
                        prompt += "B. True\n"
                        prompt += "Answer:"
                        valid_answers = ['A', 'B']  # A=False, B=True
                        
                    elif dataset == 'hellaswag':
                        # HellaSwag：4选项，使用A/B/C/D格式
                        prompt = f"Question: {question}\n"
                        option_labels = ['A', 'B', 'C', 'D']
                        for i, option in enumerate(options):
                            if i < len(option_labels):
                                prompt += f"{option_labels[i]}. {option}\n"
                        prompt += "Answer:"
                        valid_answers = option_labels[:len(options)]
                        
                    elif dataset in ['piqa', 'winogrande']:
                        # PIQA和WinoGrande：2选项，使用A/B格式
                        prompt = f"Question: {question}\n"
                        for i, option in enumerate(options[:2]):
                            prompt += f"{'A' if i == 0 else 'B'}. {option}\n"
                        prompt += "Answer:"
                        valid_answers = ['A', 'B']
                        
                    else:
                        # 默认格式：使用A/B格式
                        prompt = f"Question: {question}\n"
                        option_labels = ['A', 'B', 'C', 'D']
                        for i, option in enumerate(options):
                            if i < len(option_labels):
                                prompt += f"{option_labels[i]}. {option}\n"
                        prompt += "Answer:"
                        valid_answers = option_labels[:len(options)]
                    
                    # Tokenize
                    model_device = next(self.model.parameters()).device
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors='pt',
                        truncation=True,
                        max_length=self.max_length,
                        padding=True
                    )
                    inputs = {k: v.to(model_device) for k, v in inputs.items()}
                    
                    # 生成参数 - 加速版本：只生成1个token
                    generation_kwargs = {
                        "max_new_tokens": 1,  # 只生成1个token，大幅加速
                        "do_sample": False,
                        "pad_token_id": self.tokenizer.eos_token_id,
                        "use_cache": False,
                        "output_attentions": False,
                        "output_hidden_states": False,
                    }
                    
                    if "gemma" in self.model_path.lower():
                        generation_kwargs.update({
                            "temperature": 1.0,
                            "top_p": 1.0,
                            "repetition_penalty": 1.0,
                        })
                    
                    # 生成答案
                    outputs = self.model.generate(**inputs, **generation_kwargs)
                    
                    # 解码并清理生成的文本
                    generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                    # 移除开头的单引号和空格，只取前2个字符进行解析
                    generated_text = generated_text.lstrip("' ").strip().upper()[:2]
                    
                    # 提取预测答案
                    predicted_answer = None
                    for char in generated_text:
                        if char in valid_answers:
                            predicted_answer = valid_answers.index(char)
                            break
                    
                    # 如果没有找到明确答案，使用默认策略
                    if predicted_answer is None:
                        predicted_answer = 0  # 默认选择第一个选项
                    
                    # 调试信息输出
                    is_correct = predicted_answer == correct_idx
                    # if total < 10:
                        # print(f"\n🔍 样本 {total + 1} 调试信息:")
                        # print(f"📝 数据集: {dataset}")
                        # print(f"📝 问题: {question[:100]}...")
                        # print(f"📝 选项数量: {len(options)}")
                        # for i, opt in enumerate(options):
                        #     label = valid_answers[i] if i < len(valid_answers) else f"选项{i+1}"
                        #     print(f"📝 {label}: {opt[:50]}...")
                        # print(f"📝 正确答案: {correct_answer} (索引: {correct_idx})")
                        # print(f"📝 生成文本: '{generated_text}'")
                        # print(f"📝 预测答案: {predicted_answer} ({valid_answers[predicted_answer] if predicted_answer < len(valid_answers) else 'N/A'})")
                        # print(f"📝 是否正确: {'✅' if is_correct else '❌'}")
                        # print("-" * 50)
                    
                    # 与正确答案比较
                    if is_correct:
                        correct += 1
                    
                    total += 1
                    
                except Exception as e:
                    print(f"⚠️ 处理样本错误: {e}")
                    total += 1
                    continue
        
        if total == 0:
            return torch.tensor(0.0)
        
        accuracy = correct / total
        return torch.tensor(accuracy)

    def configure_optimizers(self):
        """配置优化器 - 评估模式不需要，但Lightning需要这个方法"""
        return None
