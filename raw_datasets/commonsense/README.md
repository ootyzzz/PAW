# Commonsense训练数据集

## 数据集概述
这是一个包含7个常识推理数据集的统一训练集，经过清理和格式标准化。

## 文件信息
- **主文件**: `commonsense_train.jsonl` (114,170 个样本)
- **样本文件**: `cs_all_final_sample.jsonl` (前20个样本示例)
- **格式**: JSONL (每行一个JSON对象)

## 数据集构成
| 数据集 | 样本数 | 任务类型 | 描述 |
|--------|--------|----------|------|
| arc-challenge | 1,119 | multiple_choice | AI2推理挑战 - 困难版 |
| arc-easy | 2,251 | multiple_choice | AI2推理挑战 - 简单版 |
| boolq | 9,427 | yes_no_question | 布尔问答 |
| hellaswag | 39,905 | sentence_completion | 常识推理 - 句子完成 |
| openbookqa | 4,957 | multiple_choice | 开放书本问答 |
| piqa | 16,113 | physical_reasoning | 物理推理 |
| winogrande | 40,398 | pronoun_resolution | 代词消歧 |
| **总计** | **114,170** | - | - |

## 任务类型分布
| 任务类型 | 样本数 | 描述 |
|----------|--------|------|
| multiple_choice | 8,327 | 多选题 (ARC, OpenBookQA) |
| physical_reasoning | 16,113 | 物理推理 (PIQA) |
| pronoun_resolution | 40,398 | 代词消歧 (WinoGrande) |
| sentence_completion | 39,905 | 句子完成 (HellaSwag) |
| yes_no_question | 9,427 | 是否问答 (BoolQ) |

## 数据格式
每个样本包含以下字段：

```json
{
  "id": "数据集名_原始ID",           // 统一ID格式
  "dataset": "数据集名称",          // 原始数据集名
  "task_type": "任务类型",          // 标准化任务类型
  "input": "输入文本",              // 问题或上下文
  "options": ["选项1", "选项2"],    // 选项列表 (如适用)
  "target": "正确答案",             // 答案文本
  "target_idx": 0                   // 答案在选项中的索引 (-1表示无标准答案)
}
```

## 样本示例

### HellaSwag (句子完成)
```json
{
  "id": "hellaswag_679",
  "dataset": "hellaswag",
  "task_type": "sentence_completion",
  "input": "[header] How to feng shui your life [title] Know who you are...",
  "options": ["选项1", "选项2", "选项3", "选项4"],
  "target": "If you do not know yourself you do not know how to balance your life...",
  "target_idx": 2
}
```

### WinoGrande (代词消歧)
```json
{
  "id": "winogrande_82737",
  "dataset": "winogrande", 
  "task_type": "pronoun_resolution",
  "input": "The reason Carrie is wearing a more modern outfit than Victoria is because _ is a very modern person.",
  "options": ["Carrie", "Victoria"],
  "target": "Carrie",
  "target_idx": 0
}
```

### BoolQ (布尔问答)
```json
{
  "id": "boolq_11815",
  "dataset": "boolq",
  "task_type": "yes_no_question", 
  "input": "Question: do ross and rachel end up getting married\nPassage: In the season's...",
  "options": ["False", "True"],
  "target": "False",
  "target_idx": 0
}
```

## 数据质量
- ✅ 格式验证: 100% 通过
- ✅ 字段完整性: 所有必要字段齐全
- ✅ 编码统一: UTF-8编码
- ✅ ID唯一性: 每个样本都有唯一ID
- ✅ 任务类型标准化: 5种标准任务类型

## 使用说明
1. 数据已经过清理和标准化，可直接用于训练
2. PIQA数据集没有标准答案标签 (`target_idx = -1`)
3. 所有文本都已清理，去除了多余的空白字符
4. 输入文本长度限制在1000字符以内
5. 数据已随机打乱，适合直接训练使用

## 生成工具
- `utils/mix_commonsense_datasets.py`: 数据集合并工具
- `utils/final_data_cleaner.py`: 最终清理工具
- `utils/validate_final_data.py`: 数据验证工具

## 创建时间
2024年 (具体日期请查看文件修改时间)
