"""
NanoChat-Lab 评测模块
"""
from .test_cases import RAG_TEST_CASES, AGENT_TEST_CASES, ROUTING_TEST_CASES
from .metrics import check_recall, check_answer_quality, calculate_accuracy, calculate_avg_latency
from .report import save_results, generate_final_report

__all__ = [
    "RAG_TEST_CASES",
    "AGENT_TEST_CASES",
    "ROUTING_TEST_CASES",
    "check_recall",
    "check_answer_quality",
    "calculate_accuracy",
    "calculate_avg_latency",
    "save_results",
    "generate_final_report",
]
