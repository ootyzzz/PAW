"""
HellaSwag 数据集专用 prompt 构建函数
输入: sample 字典，包含 'context' 和 'choices'
输出: 格式化后的 prompt string
"""
def hellaswag_adapter(sample):
    context = sample['context']
    choices = sample['choices']
    choices_str = ' or '.join(choices)
    return f"Q: {context}\nA: {choices_str}?\n"
