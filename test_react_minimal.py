#!/usr/bin/env python3
"""
ReActAgent 最小化测试脚本
只包含 3 个工具：calc / time / weather
不联网，一步步验证核心流程
"""

import re
import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.llm import LLM


# ============================================================
#  工具函数（纯函数，不依赖网络）
# ============================================================


def tool_calc(expr: str) -> str:
    """计算数学表达式"""
    try:
        allowed = set("0123456789+-*/.() =^")
        cleaned = ""
        for c in expr:
            if c in allowed:
                cleaned += c
        cleaned = cleaned.replace("^", "**")
        if not cleaned:
            return "输入为空"
        result = eval(cleaned, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


def tool_time(_: str) -> str:
    """获取当前时间"""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


MOCK_WEATHER = {
    "北京": "晴, 15~25度",
    "上海": "多云, 18~26度",
    "广州": "阵雨, 22~30度",
    "深圳": "多云, 23~31度",
    "成都": "阴, 16~24度",
    "杭州": "晴, 17~27度",
    "武汉": "小雨, 18~28度",
    "南京": "多云, 16~25度",
    "西安": "晴, 14~26度",
    "重庆": "阴, 19~29度",
}


def tool_weather(city: str) -> str:
    """查询城市天气（模拟数据）"""
    city = city.strip()
    for key, val in MOCK_WEATHER.items():
        if key in city or city in key:
            return f"{key} {val}"
    return f"未找到{city}的天气数据，目前支持: {chr(44).join(MOCK_WEATHER.keys())}"


# ============================================================
#  ReAct Agent（极简版）
# ============================================================


TOOLS = {
    "calc": {
        "func": tool_calc,
        "desc": "计算数学表达式，如 3*5、(2+3)*4、2^10",
    },
    "time": {
        "func": tool_time,
        "desc": "获取当前日期和时间",
    },
    "weather": {
        "func": tool_weather,
        "desc": "查询城市天气，如 北京、上海、广州",
    },
}


SYSTEM_PROMPT = """你是一个助手，能用以下工具回答问题。

可用工具：
- calc: 计算数学表达式，如 3*5、(2+3)*4
- time: 获取当前日期和时间
- weather: 查询城市天气，如 北京、上海

严格按格式回复（不要自己写 Observation）：

Thought: 思考
Action: 工具名
Action Input: 输入

示例：
问题: 3乘以5等于多少
Thought: 用计算工具
Action: calc
Action Input: 3*5

问题: 现在几点
Thought: 用时间工具
Action: time
Action Input: now

问题: 北京天气怎么样
Thought: 用天气工具
Action: weather
Action Input: 北京"""


def parse_response(text: str) -> dict:
    """解析模型输出"""
    text = text.strip()
    # 1. Final Answer
    fa = re.search(r"Final Answer:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if fa:
        return {"type": "final_answer", "content": fa.group(1).strip()}
    # 2. Action
    action = re.search(r"Action:\s*(\w+)", text, re.IGNORECASE)
    inp = re.search(r"Action Input:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if action and inp:
        name = action.group(1).strip().lower()
        value = inp.group(1).strip()
        value = value.strip(chr(39)).strip(chr(34)).strip()  # 去掉引号
        return {"type": "action", "name": name, "input": value}
    # 3. 只有 Thought
    return {"type": "retry", "content": text}


def run_agent(llm, question: str, max_steps: int = 5, verbose: bool = True):
    """运行 ReAct Agent"""
    prompt = SYSTEM_PROMPT + "\n\n问题: " + question + "\nThought: "
    if verbose:
        print(f"\n{chr(61)*60}")
        print(f"问题: {question}")
