"""QA 数据加载器：从 Excel 文件读取 QA 对。"""

from typing import List, Dict

import pandas as pd


def load_qa_pairs(excel_path: str) -> List[Dict]:
    """从 Excel 文件加载 QA 对。

    Excel 需包含列：qa_id, question, answer, category
    """
    df = pd.read_excel(excel_path)
    required = {"qa_id", "question", "answer", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Excel 缺少必需列：{missing}")

    result = []
    for _, row in df.iterrows():
        result.append({
            "qa_id": int(row["qa_id"]),
            "question": str(row["question"]),
            "answer": str(row["answer"]),
            "category": str(row["category"]),
        })
    return result
