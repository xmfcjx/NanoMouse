"""
评测指标计算
"""


def check_recall(docs, expected_keywords):
    """检查检索文档是否包含期望关键词（召回率）"""
    if not docs:
        return 0.0
    all_text = " ".join(docs).lower()
    hit = sum(1 for kw in expected_keywords if kw.lower() in all_text)
    return hit / len(expected_keywords)


def check_answer_quality(answer, expected_keywords):
    """检查生成答案是否包含期望关键词（准确率）"""
    if not answer:
        return 0.0
    answer_lower = answer.lower()
    hit = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hit / len(expected_keywords)


def calculate_accuracy(results, key="correct"):
    """计算准确率"""
    if not results:
        return 0.0
    return sum(1 for r in results if r.get(key)) / len(results)


def calculate_avg_latency(results, key="latency"):
    """计算平均延迟"""
    if not results:
        return 0.0
    return sum(r.get(key, 0) for r in results) / len(results)
