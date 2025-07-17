# File: adapters/arc_adapter.py (relative to workspace root)
"""
ARC-e/ARC-c 数据集专用 prompt 构建函数
输入: sample 字典，包含 'question' 和 'choices'
输出: 格式化后的 prompt string
"""
def arc_adapter(sample):
    question = sample['question']
    choices = sample['choices']
    choices_str = ' or '.join(choices)
    return f"Q: {question}\nA: {choices_str}?\n"
