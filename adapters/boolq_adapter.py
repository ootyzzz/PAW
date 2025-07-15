"""
BoolQ 数据集专用 prompt 构建函数
输入: sample 字典，包含 'question'
输出: 格式化后的 prompt string（Yes/No）
"""
def boolq_adapter(sample):
    question = sample['question']
    choice1, choice2 = 'Yes', 'No'
    return f"Q: {question}\nA: {choice1} or {choice2}?\n"
