"""
Lightning评估模块
包含LightningModelEvaluator类和相关评估逻辑
"""

from .config import *


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
        
        # 加载模型和tokenizer
        self._load_model()
        
    def _load_model(self):
        """加载模型和tokenizer"""
        print(f"📦 加载模型: {self.model_path}")
        
        # 检查是否是本地路径还是Hugging Face模型ID
        is_local_path = os.path.exists(self.model_path)
        
        print(f"🔍 模型路径检查: {self.model_path}")
        print(f"🔍 是否为本地路径: {is_local_path}")
        
        # 检查模型路径是否存在
        if not is_local_path:
            print(f"❌ 模型路径不存在: {self.model_path}")
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
                    # 加载PEFT配置获取基础模型信息
                    peft_config = PeftConfig.from_pretrained(self.model_path)
                    detected_base_model = peft_config.base_model_name_or_path
                    
                    # 使用提供的基础模型路径或检测到的路径
                    actual_base_model = self.base_model_path or detected_base_model
                    
                    # 确认基础模型路径
                    if not os.path.exists(actual_base_model) and "/" not in actual_base_model:
                        # 可能是相对路径，尝试autodl-tmp中的常见位置
                        for prefix in ["/root/autodl-tmp/models/", "/root/autodl-tmp/"]:
                            test_path = f"{prefix}{actual_base_model}"
                            if os.path.exists(test_path):
                                actual_base_model = test_path
                                break
                    
                    print(f"📦 加载基础模型: {actual_base_model}")
                    
                    # 加载基础模型的tokenizer (移除local_files_only限制)
                    tokenizer_kwargs = {"trust_remote_code": True}
                    
                    self.tokenizer = AutoTokenizer.from_pretrained(actual_base_model, **tokenizer_kwargs)
                    
                    # 特殊处理Gemma模型
                    if "gemma" in actual_base_model.lower():
                        print("🦙 检测到Gemma模型，应用特殊配置...")
                        load_kwargs.update({
                            "attn_implementation": "eager",  # 避免使用flash attention
                            "use_cache": False,  # 禁用缓存机制
                            "_attn_implementation_internal": "eager"
                        })
                    
                    # 加载基础模型
                    base_model = AutoModelForCausalLM.from_pretrained(
                        actual_base_model,
                        **load_kwargs
                    )
                    
                    print(f"🔧 加载LoRA适配器: {self.model_path}")
                    # 加载PEFT模型
                    self.model = PeftModel.from_pretrained(base_model, self.model_path)
                    
                except Exception as e:
                    print(f"❌ 作为PEFT模型加载失败: {e}")
                    raise RuntimeError(f"无法加载LoRA模型: {self.model_path}，LoRA模型必须与正确的基础模型匹配")
            else:
                # 常规模型加载流程
                print("📦 加载为常规模型...")
                
                # 处理tokenizer (移除严格的local_files_only限制)
                tokenizer_kwargs = {"trust_remote_code": True}
                    
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, **tokenizer_kwargs)
                except Exception as e:
                    print(f"⚠️ 标准tokenizer加载失败: {e}")
                    print("尝试使用备用tokenizer选项...")
                    tokenizer_kwargs["use_fast"] = False
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, **tokenizer_kwargs)
                
                # 针对Gemma模型的特殊处理
                model_name_lower = self.model_path.lower()
                special_kwargs = load_kwargs.copy()
                
                if "gemma" in model_name_lower:
                    print("🔍 检测到Gemma模型，应用特殊配置...")
                    special_kwargs.update({
                        "attn_implementation": "eager",  # 避免使用flash attention
                        "use_cache": False,  # 禁用缓存机制
                        "_attn_implementation_internal": "eager"
                    })
                    
                # 加载模型，移除严格的local_files_only限制
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    **special_kwargs
                )
        
            # 确保模型处于评估模式
            self.model.eval()
            
            # 设置pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
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
            # 计算损失
            loss = self._compute_loss(batch)
            # 计算准确率
            accuracy = self._compute_accuracy(batch)
            perplexity = torch.exp(loss)
            
            batch_size = len(batch)
            
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
            print(f"❌ test_step失败 (batch_idx={batch_idx}): {e}")
            print(f"❌ batch内容: {batch}")
            traceback.print_exc()
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
