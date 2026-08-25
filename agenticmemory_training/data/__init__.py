"""agenticmemory_training.data 数据合成与教师标注模块(P1-1~P1-3 准备)

子模块:
- synthesis.py    P1-1: 数据合成(公开集 + GPT-4 合成)
- teacher_labeling.py P1-2: 教师标注(13 字段 schema)
- evaluation.py   P1-3: 标注一致性 / 字段填充率 / 教师偏置分析

所有脚本共用 JSONL 输入输出格式:每行一个 JSON 对象。
"""