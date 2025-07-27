"""
Lightning评估模块
包含LightningModelEvaluator类和相关评估逻辑
"""

from .config import *
import psutil
import gc


# 全局模型缓存，防止重复加载
_MODEL_CACHE = {}


def log_memory_usage(stage=""):
    """记录内存使用情况"""
    try:
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / 1024**3
            gpu_reserved = torch.cuda.memory_reserved() / 1024**3
            gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🔍 [{stage}] GPU内存: {gpu_allocated:.2f}GB / {gpu_reserved:.2f}GB / {gpu_total:.2f}GB")
        
        ram_usage = psutil.virtual_memory()
        print(f"🔍 [{stage}] RAM使用: {ram_usage.used/1024**3:.2f}GB / {ram_usage.total/1024**3:.2f}GB ({ram_usage.percent:.1f}%)")
    except Exception as e:
        print(f"⚠️ 内存监控失败: {e}")


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
        
    @detailed_exception_handler
    def _load_model(self):
        """加载模型和tokenizer"""
        # 创建缓存键
        cache_key = f"{self.model_path}_{self.base_model_path or 'none'}"
        
        # 检查全局缓存
        if cache_key in _MODEL_CACHE:
            print(f"✅ 从缓存加载模型: {self.model_path}")
            cached_data = _MODEL_CACHE[cache_key]
            self.model = cached_data['model']
            self.tokenizer = cached_data['tokenizer']
            self._model_loaded = True
            log_memory_usage("缓存加载后")
            return
        
        # 检查是否已经加载过模型
        if self._model_loaded and self.model is not None and self.tokenizer is not None:
            print(f"✅ 模型已加载，跳过重复加载: {self.model_path}")
            return
        
        print(f"📦 开始加载模型: {self.model_path}")
        log_memory_usage("模型加载前")
        
        # 检查是否是本地路径还是Hugging Face模型ID
        is_local_path = os.path.exists(self.model_path)
        
        print(f"🔍 模型路径检查: {self.model_path}")
        print(f"🔍 绝对路径: {os.path.abspath(self.model_path)}")
        print(f"🔍 是否为本地路径: {is_local_path}")
        
        # 如果是本地路径，检查目录内容
        if is_local_path:
            try:
                files = os.listdir(self.model_path)
                print(f"🔍 模型目录内容: {files[:10]}...")  # 只显示前10个文件
                
                # 检查关键文件
                key_files = ['config.json', 'pytorch_model.bin', 'model.safetensors', 'tokenizer.json']
                for key_file in key_files:
                    if key_file in files:
                        print(f"✅ 找到关键文件: {key_file}")
                    else:
                        print(f"⚠️ 未找到文件: {key_file}")
            except Exception as e:
                print(f"⚠️ 无法读取模型目录: {e}")
        
        # 检查模型路径是否存在
        if not is_local_path:
            print(f"❌ 模型路径不存在: {self.model_path}")
            print(f"❌ 当前工作目录: {os.getcwd()}")
            print(f"❌ 尝试的绝对路径: {os.path.abspath(self.model_path)}")
            raise FileNotFoundError(f"模型路径不存在: {self.model_path}")
        
        try:
            # 模型加载参数
            load_kwargs = {
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "trust_remote_code": True,
                "use_cache": True,
                "device_map": "auto" if torch.cuda.is_available() else None,
            }
            
            print(f"🔍 模型加载参数: {load_kwargs}")
            
            # 检查是否是LoRA模型
            config_path = Path(self.model_path) / "adapter_config.json"
            print(f"🔍 检查LoRA配置文件: {config_path} (存在: {config_path.exists()})")
            
            if config_path.exists():
                # LoRA模型加载流程
                print("🔧 检测到LoRA模型，使用PEFT加载...")
                try:
                    print("🔍 步骤1: 加载PEFT配置...")
                    # 加载PEFT配置获取基础模型信息
                    peft_config = PeftConfig.from_pretrained(self.model_path)
                    detected_base_model = peft_config.base_model_name_or_path
                    print(f"🔍 检测到的基础模型: {detected_base_model}")
                    
                    # 使用提供的基础模型路径或检测到的路径
                    actual_base_model = self.base_model_path or detected_base_model
                    print(f"🔍 实际使用的基础模型路径: {actual_base_model}")
                    
                    # 确认基础模型路径
                    if not os.path.exists(actual_base_model) and "/" not in actual_base_model:
                        # 可能是相对路径，尝试autodl-tmp中的常见位置
                        for prefix in ["/root/autodl-tmp/models/", "/root/autodl-tmp/"]:
                            test_path = f"{prefix}{actual_base_model}"
                            if os.path.exists(test_path):
                                actual_base_model = test_path
                                print(f"🔍 找到基础模型: {actual_base_model}")
                                break
                    
                    print(f"🔍 步骤2: 加载tokenizer...")
                    log_memory_usage("tokenizer加载前")
                    
                    # 加载基础模型的tokenizer (移除local_files_only限制)
                    tokenizer_kwargs = {"trust_remote_code": True}
                    
                    self.tokenizer = AutoTokenizer.from_pretrained(actual_base_model, **tokenizer_kwargs)
                    print(f"✅ tokenizer加载成功")
                    log_memory_usage("tokenizer加载后")
                    
                    # 特殊处理Gemma模型
                    if "gemma" in actual_base_model.lower():
                        print("🦙 检测到Gemma模型，应用特殊配置...")
                        load_kwargs.update({
                            "attn_implementation": "eager",  # 避免使用flash attention
                            "use_cache": False,  # 禁用缓存机制
                            "_attn_implementation_internal": "eager"
                        })
                    
                    print(f"🔍 步骤3: 加载基础模型...")
                    print(f"📦 基础模型路径: {actual_base_model}")
                    print(f"📦 加载参数: {load_kwargs}")
                    log_memory_usage("基础模型加载前")
                    
                    # 加载基础模型
                    base_model = AutoModelForCausalLM.from_pretrained(
                        actual_base_model,
                        **load_kwargs
                    )
                    print(f"✅ 基础模型加载成功")
                    log_memory_usage("基础模型加载后")
                    
                    print(f"🔍 步骤4: 加载LoRA适配器...")
                    print(f"🔧 LoRA路径: {self.model_path}")
                    
                    # 加载PEFT模型
                    self.model = PeftModel.from_pretrained(base_model, self.model_path)
                    print(f"✅ LoRA适配器加载成功")
                    log_memory_usage("LoRA加载后")
                    
                except Exception as e:
                    print(f"❌ 作为PEFT模型加载失败: {e}")
                    raise RuntimeError(f"无法加载LoRA模型: {self.model_path}，LoRA模型必须与正确的基础模型匹配")
            else:
                # 常规模型加载流程
                print("📦 加载为常规模型...")
                
                print("🔍 步骤1: 加载tokenizer...")
                log_memory_usage("tokenizer加载前")
                
                # 处理tokenizer (移除严格的local_files_only限制)
                tokenizer_kwargs = {"trust_remote_code": True}
                    
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, **tokenizer_kwargs)
                    print("✅ tokenizer加载成功")
                except Exception as e:
                    print(f"⚠️ 标准tokenizer加载失败: {e}")
                    print("🔍 尝试使用备用tokenizer选项...")
                    tokenizer_kwargs["use_fast"] = False
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, **tokenizer_kwargs)
                    print("✅ 备用tokenizer加载成功")
                
                log_memory_usage("tokenizer加载后")
                
                print("🔍 步骤2: 准备模型加载参数...")
                # 针对特殊模型的处理
                model_name_lower = self.model_path.lower()
                special_kwargs = load_kwargs.copy()
                
                if "gemma" in model_name_lower:
                    print("🦙 检测到Gemma模型，应用特殊配置...")
                    special_kwargs.update({
                        "attn_implementation": "eager",  # 避免使用flash attention
                        "use_cache": False,  # 禁用缓存机制
                        "_attn_implementation_internal": "eager"
                    })
                elif "llama" in model_name_lower:
                    print("🦙 检测到Llama模型，应用特殊配置...")
                    # Llama模型可能需要特殊处理
                    special_kwargs.update({
                        "use_cache": True,  # Llama通常可以使用缓存
                    })
                
                print(f"🔍 最终加载参数: {special_kwargs}")
                
                print("🔍 步骤3: 加载模型...")
                log_memory_usage("模型加载前")
                    
                # 加载模型，移除严格的local_files_only限制
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    **special_kwargs
                )
                print("✅ 模型加载成功")
                log_memory_usage("模型加载后")
        
            # 确保模型处于评估模式
            self.model.eval()
            
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
            print(f"💾 模型已缓存: {cache_key}")
                
            print(f"✅ 模型加载成功: {self.model_path}")
            
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
            print(f"🔍 test_step开始 - batch_idx: {batch_idx}, batch_size: {len(batch) if batch else 0}")
            log_memory_usage(f"test_step_{batch_idx}_开始")
            
            # 验证batch内容
            if not batch:
                print(f"⚠️ 空batch，跳过处理")
                return {
                    'loss': torch.tensor(0.0),
                    'accuracy': torch.tensor(0.0),
                    'perplexity': torch.tensor(1.0),
                    'batch_size': 0
                }
            
            print(f"🔍 batch样本示例: {batch[0] if len(batch) > 0 else 'None'}")
            
            # 计算损失
            print(f"🔍 开始计算损失...")
            loss = self._compute_loss(batch)
            print(f"🔍 损失计算完成: {loss}")
            
            # 计算准确率
            print(f"🔍 开始计算准确率...")
            accuracy = self._compute_accuracy(batch)
            print(f"🔍 准确率计算完成: {accuracy}")
            
            # 计算困惑度
            perplexity = torch.exp(loss)
            print(f"🔍 困惑度计算完成: {perplexity}")
            
            batch_size = len(batch)
            
            # 记录指标
            self.log('test/loss', loss, batch_size=batch_size)
            self.log('test/accuracy', accuracy, batch_size=batch_size)
            self.log('test/perplexity', perplexity, batch_size=batch_size)
            
            log_memory_usage(f"test_step_{batch_idx}_完成")
            print(f"✅ test_step完成 - batch_idx: {batch_idx}")
            
            return {
                'loss': loss,
                'accuracy': accuracy,
                'perplexity': perplexity,
                'batch_size': batch_size
            }
        except Exception as e:
            print(f"❌ test_step失败 (batch_idx={batch_idx}): {e}")
            print(f"❌ 异常类型: {type(e).__name__}")
            print(f"❌ batch大小: {len(batch) if batch else 'None'}")
            if batch and len(batch) > 0:
                print(f"❌ 第一个样本: {batch[0]}")
            print(f"❌ 详细traceback:")
            traceback.print_exc()
            log_memory_usage(f"test_step_{batch_idx}_异常")
            
            # 返回默认值避免训练中断
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
                input_ids = torch.stack(inputs).to(self.device)
                attention_mask = torch.ones_like(input_ids).to(self.device)
                labels = torch.stack(labels).to(self.device)
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
        """计算准确率"""
        if not isinstance(batch, list):
            return torch.tensor(0.25)  # 4选1题的随机基线
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for item in batch:
                try:
                    # 解析数据项
                    question = item.get('input', '')
                    options = item.get('options', [])
                    correct_answer = item.get('target', 'A')
                    
                    if not options:
                        total += 1
                        continue
                    
                    # 格式化带选项的问题
                    prompt = f"Question: {question}\n"
                    for option in options:
                        prompt += f"{option}\n"
                    prompt += "Answer:"
                    
                    # Tokenize
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors='pt',
                        truncation=True,
                        max_length=self.max_length,
                        padding=True
                    ).to(self.device)
                    
                    # Gemma模型特殊处理
                    model_name_lower = self.model_path.lower()
                    generation_kwargs = {
                        "max_new_tokens": 3,  # 减少生成长度
                        "do_sample": False,
                        "pad_token_id": self.tokenizer.eos_token_id,
                        "use_cache": False,  # 禁用缓存
                        "output_attentions": False,
                        "output_hidden_states": False,
                    }
                    
                    if "gemma" in model_name_lower:
                        # Gemma模型特殊适配
                        generation_kwargs.update({
                            "temperature": 1.0,
                            "top_p": 1.0,
                            "repetition_penalty": 1.0,
                        })
                    
                    # 生成答案
                    outputs = self.model.generate(
                        **inputs,
                        **generation_kwargs
                    )
                    
                    # 解码生成的答案
                    generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                    generated_answer = generated_text.strip().upper()
                    
                    # 提取第一个字母 (A, B, C, 或 D)
                    predicted_answer = None
                    for char in generated_answer:
                        if char in ['A', 'B', 'C', 'D']:
                            predicted_answer = char
                            break
                    
                    # 如果没有找到明确答案，尝试匹配选项前缀
                    if predicted_answer is None:
                        for option in options:
                            if option.startswith('A:') and 'A' in generated_answer:
                                predicted_answer = 'A'
                            elif option.startswith('B:') and 'B' in generated_answer:
                                predicted_answer = 'B'
                            elif option.startswith('C:') and 'C' in generated_answer:
                                predicted_answer = 'C'
                            elif option.startswith('D:') and 'D' in generated_answer:
                                predicted_answer = 'D'
                            if predicted_answer:
                                break
                    
                    # 与正确答案比较
                    if predicted_answer == correct_answer:
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
