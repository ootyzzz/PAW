"""
train_lora.py
Qwen-0.5B LoRA 微调训练脚本，适配 Commonsense 数据集，A100 环境推荐。
"""
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# 配置
model_name = "Qwen/Qwen-0.5B"
dataset_name = "boolq"  # 可替换为其他数据集
output_dir = "data/BoolQ/checkpoints"

# 加载模型和 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# LoRA 配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],  # 视模型结构而定
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 加载数据集并格式化
if dataset_name == "boolq":
    ds = load_dataset("boolq")
    def preprocess(example):
        prompt = f"Q: {example['question']}\nA: Yes or No?\n"
        return {"input_ids": tokenizer(prompt, truncation=True, padding="max_length", max_length=128)["input_ids"]}
    train_ds = ds["train"].map(preprocess)
# 其他数据集可按需补充

# TrainingArguments
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    fp16=True,
    save_strategy="epoch",
    logging_steps=10,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    tokenizer=tokenizer
)

trainer.train()

# 保存 LoRA adapter 权重
model.save_pretrained(f"{output_dir}/epoch_10_lora.pt")
print("训练完成，LoRA checkpoint 已保存！")
