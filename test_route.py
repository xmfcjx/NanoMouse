"""
Agent 工具全量测试脚本 (v2.2 - 修复用例歧义 & 真实模拟 chat.py 兜底逻辑)
"""

import re
import os
import sys

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from core.ReActAgent import ReActAgent
from core.input_classifier import InputClassifier


def run_tool_tests(agent):
    print("=" * 65)
    print(" PART 1: 工具函数直接测试 (direct_call)")
    print("=" * 65)

    test_cases = [
        ("calc", "print(15 * 8)", r"^120$", "15*8"),
        ("calc", "print(100 / 5)", r"^20(\.0)?$", "100/5"),
        ("calc", "print(10 + 20)", r"^30$", "10+20"),
        ("calc", "print(2 ** 10)", r"^1024$", "2的10次方"),
        ("calc", "print(3.14 * 2)", r"^6\.28$", "小数运算"),
        ("time", "", r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", "获取当前时间"),
        ("date", "2025-01-01 + 100天", r"^2025-04-11$", "2025元旦+100天"),
        ("date", "2025-12-31 - 1天", r"^2025-12-30$", "2025跨年-1天"),
        ("weekday", "2025-10-01", r"Wednesday", "2025国庆节星期几"),
        ("weekday", "2025-01-01", r"Wednesday", "2025元旦星期几"),
        ("weekday", "", r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)", "不传参默认今天"),
        # 🎯 修复：换用例，用跨年计算，消除闭区间歧义
        ("days_between", "2025-01-01, 2026-01-01", r"^365$", "2025全年天数(跨年)"),
        ("days_between", "2024-01-01, 2024-12-31", r"^365$", "2024闰年天数(头尾算差)"),
        ("convert", "100km to miles", r"^62\.13\d*", "公里转英里"),
        ("convert", "1kg to pounds", r"^2\.204\d*", "千克转磅"),
        ("convert", "36c to f", r"^96\.8°F$", "摄氏转华氏"),
        # 不截断精度，保留实际输出 100.00°C
        ("convert", "212f to c", r"^100\.00°C$", "华氏转摄氏(保留精度)"),
        ("base_convert", "255 to hex", r"^FF$", "255转十六进制"),
        ("base_convert", "1010 to dec", r"^10$", "二进制转十进制"),
        ("base_convert", "255 to bin", r"^11111111$", "255转二进制"),
        ("base_convert", "255 to oct", r"^377$", "255转八进制"),
        ("str_tools", "hello, len", r"^5$", "字符串长度"),
        ("str_tools", "hello, reverse", r"^olleh$", "字符串反转"),
        ("str_tools", "hello, upper", r"^HELLO$", "转大写"),
        ("str_tools", "HELLO, lower", r"^hello$", "转小写"),
        ("random", "dice 3", r"^[\d], [\d], [\d]$", "掷3个骰子(格式)"),
        ("random", "coin 2", r"^(Head|Tail)", "掷2个硬币(格式)"),
        # 不截断精度，保留实际输出 88.0
        ("statistics", "10, 55, 23, 88, 41, max", r"^88\.0$", "最大值(保留精度)"),
        ("statistics", "10, 55, 23, 88, 41, min", r"^10\.0$", "最小值(保留精度)"),
        ("statistics", "10, 55, 23, 88, 41, avg", r"^43\.4$", "平均值"),
        ("statistics", "10, 55, 23, 88, 41, sum", r"^217\.0$", "求和(保留精度)"),
        ("solve", "3*x*x=27", r"x = -3", "3x²=27 (二次)"),
        ("solve", "2*x+5=17", r"x = 6", "2x+5=17 (一次)"),
        ("solve", "x*x=81", r"x = -9", "x²=81 (二次双解)"),
        ("solve", "4*x*x-8*x+4=0", r"x = 1", "4x²-8x+4=0 (重根)"),
        ("solve", "3*x - 9 = 0", r"x = 3", "3x-9=0 (一次)"),
    ]

    pass_count = 0
    fail_count = 0
    fail_details = []

    for tool_name, tool_input, expected_pattern, desc in test_cases:
        result = agent.direct_call(tool_name, tool_input)
        passed = bool(re.search(expected_pattern, result))
        if passed: pass_count += 1; status = "PASS"
        else: fail_count += 1; status = "FAIL"; fail_details.append((tool_name, desc, expected_pattern, result))
        print(f"  [{status}] {tool_name:<14} | {desc:<25} | => {result}")

    print(f"\n  结果: {pass_count} 通过, {fail_count} 失败\n")
    if fail_details:
        print("  " + "-" * 60)
        for tool, desc, pattern, result in fail_details:
            print(f"  {tool} ({desc}) 期望:{pattern} 实际:{result}")
        print()
    return pass_count, fail_count


def run_classifier_tests(classifier):
    print("=" * 65)
    print(" PART 2: 分类器路由测试 (classify)")
    print("=" * 65)

    test_cases = [
        ("what time is it now?", "agent", "time", "时间-几点"),
        ("几点了？", "agent", "time", "时间-中文几点"),
        ("what day is it today?", "agent", "weekday", "时间-星期几"),
        ("今天星期几？", "agent", "weekday", "时间-中文星期"),
        ("今天是几号？", "agent", "date_today", "时间-日期"),
        ("元旦和国庆相差几天", "agent", "days_between", "时间-天数差"),
        ("3*x*x=27", "agent", "equation", "方程-二次"),
        ("15 * 8", "agent", "arithmetic", "算术-乘法"),
        # 🎯 修复：即使没有历史记忆，也必须是 identity
        ("Who am I?", "identity", "unknown", "身份-无记忆查询"),
        ("My name is Alice.", "identity", "set_name", "身份-设置名称"),
        ("What is the capital of France?", "rag", "", "RAG-普通问题"),
    ]

    pass_count = 0
    fail_count = 0
    fail_details = []

    for query, exp_type, exp_subtype, desc in test_cases:
        # 模拟空历史传入
        result = classifier.classify(query, history=[])
        pred_type = result["type"]
        pred_subtype = result.get("subtype", "")
        type_ok = pred_type == exp_type
        subtype_ok = pred_subtype == exp_subtype
        passed = type_ok and subtype_ok

        if passed: pass_count += 1; status = "PASS"
        else: fail_count += 1; status = "FAIL"; fail_details.append((query, desc, exp_type, exp_subtype, pred_type, pred_subtype))

        q_display = f"'{query[:30]}...'" if len(query) > 30 else f"'{query}'"
        print(f"  [{status}] {desc:<22} | {q_display}")
        if not passed:
            print(f"         期望: type={exp_type}, subtype={exp_subtype}")
            print(f"         实际: type={pred_type}, subtype={pred_subtype}")

    print(f"\n  结果: {pass_count} 通过, {fail_count} 失败\n")
    if fail_details:
        print("  " + "-" * 60)
        for query, desc, et, es, pt, ps in fail_details:
            print(f"  {desc}: '{query}' 期望({et},{es}) 实际({pt},{ps})")
        print()
    return pass_count, fail_count


def run_integration_tests(classifier, agent):
    print("=" * 65)
    print(" PART 3: 端到端集成测试 (分类器 → 直通车 → 工具)")
    print("=" * 65)

    test_cases = [
        ("what time is it now?", r"\d{4}-\d{2}-\d{2}", "时间查询"),
        ("今天星期几？", r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)", "星期查询"),
        ("3*x*x=27", r"x = -3.*x = 3|x = 3.*x = -3", "解方程-二次"),
        ("What is 15 multiplied by 4?", r"^60$", "算术-英文"),
    ]

    pass_count = 0
    fail_count = 0
    fail_details = []
    direct_map = {
        "time": ("time", ""), "date_today": ("time", ""),
        "weekday": ("weekday", None), "days_between": ("days_between", None),
        "equation": ("solve", None), "arithmetic": ("calc", None),
    }

    for query, expected_pattern, desc in test_cases:
        result = classifier.classify(query)
        pred_type = result["type"]; subtype = result.get("subtype", "")
        response = ""
        try:
            if pred_type == "agent" and subtype in direct_map:
                tool_name, tool_input = direct_map[subtype]
                if tool_input is None: tool_input = query
                if subtype == "arithmetic":
                    expr = classifier.extract_arithmetic_expr(query)
                    response = agent.direct_call("calc", f"print({expr})") if expr else "ERROR"
                else:
                    response = agent.direct_call(tool_name, tool_input)
            else:
                response = f"UNEXPECTED ROUTE: {pred_type}/{subtype}"
        except Exception as e: response = f"ERROR: {str(e)}"

        passed = bool(re.search(expected_pattern, response))
        if passed: pass_count += 1; status = "PASS"
        else: fail_count += 1; status = "FAIL"; fail_details.append((query, desc, expected_pattern, response))
        print(f"  [{status}] {desc:<18} | '{query}' => {response}")

    print(f"\n  结果: {pass_count} 通过, {fail_count} 失败\n")
    if fail_details:
        print("  " + "-" * 60)
        for query, desc, pattern, result in fail_details:
            print(f"  {desc} 期望:{pattern} 实际:{result}")
        print()
    return pass_count, fail_count


def run_edge_cases(agent):
    print("=" * 65)
    print(" PART 4: 边界和容错测试")
    print("=" * 65)

    test_cases = [
        ("calc", "print(1/0)", "Error", "除零错误"),
        ("solve", "hello world", "格式错误", "非方程输入"),
        ("convert", "abc to xyz", "不支持", "不支持的转换"),
        # 🎯 修复：正则改成匹配"进制转换失败"
        ("base_convert", "zzz to hex", "进制转换失败", "非法数字"),
        ("str_tools", "hello", "格式错误", "缺少操作参数"),
        ("statistics", "abc, max", "无有效数字", "非数字统计"),
        ("TIME_TOOL", "", r"\d{4}-\d{2}-\d{2}", "容错-大写工具名"),
        ("Calc", "print(1+1)", r"^2$", "容错-首字母大写"),
    ]

    pass_count = 0
    fail_count = 0
    fail_details = []

    for tool_name, tool_input, expected_pattern, desc in test_cases:
        result = agent.direct_call(tool_name, tool_input)
        passed = bool(re.search(expected_pattern, result))
        if passed: pass_count += 1; status = "PASS"
        else: fail_count += 1; status = "FAIL"; fail_details.append((tool_name, desc, expected_pattern, result))
        print(f"  [{status}] {desc:<25} | => {result}")

    print(f"\n  结果: {pass_count} 通过, {fail_count} 失败\n")
    if fail_details:
        print("  " + "-" * 60)
        for tool, desc, pattern, result in fail_details:
            print(f"  {tool} ({desc}) 期望:{pattern} 实际:{result}")
        print()
    return pass_count, fail_count


def run_llm_judgment_tests(agent):
    """第五部分：完全模拟 chat.py 的真实兜底逻辑"""
    print("=" * 65)
    print(" PART 5: LLM 无工具判断测试 (100% 还原 chat.py 兜底分支)")
    print("=" * 65)

    test_cases = [
        "What is your name?",
        "帮我写一首关于春天的诗",
        "苹果是什么颜色的？",
        "北京在哪里？",
        "今天天气怎么样？"
    ]

    pass_count = 0
    fail_count = 0
    fail_details = []

    for query in test_cases:
        # 强制走 agent.run()
        agent_response = agent.run(query)
        
        # 🎯 核心修改：100% 抄袭 chat.py 里的 else 分支逻辑
        final_match = re.search(r"Final Answer:\s*(.*)", agent_response, re.IGNORECASE | re.DOTALL)
        if final_match:
            response = final_match.group(1).strip()
        else:
            response = agent_response.strip()

        # 判断标准：最终抛给用户的话，有没有包含 "don't know"
        is_honest = "don't know" in response.lower()
        
        if is_honest:
            status = "PASS"
            grade = "诚实回答"
            pass_count += 1
        else:
            status = "FAIL"
            grade = "瞎编或乱输出"
            fail_count += 1
            fail_details.append((query, response))

        print(f"  [{status}] {grade:<15}")
        print(f"         Q: {query}")
        print(f"         最终抛给用户: {response}")
        print()

    print(f"  结果: {pass_count} 诚实, {fail_count} 瞎编\n")
    if fail_details:
        print("  " + "-" * 60)
        print("  瞎编详情:")
        for query, result in fail_details:
            print(f"  Q: {query}")
            print(f"  A: {result}")
            print()
    return pass_count, fail_count


def main():
    print("\n" + "=" * 65)
    print("  NanoChat Agent 全量测试 (v2.2)")
    print("=" * 65 + "\n")

    print("正在初始化 Agent 和 Classifier...")
    from core.llm import LLM
    llm = LLM()
    agent = ReActAgent(llm, max_steps=3)
    classifier = InputClassifier()
    print("初始化完成。\n")

    scores = {}
    scores["工具函数"], scores["工具失败"] = run_tool_tests(agent)
    scores["分类器"], scores["分类失败"] = run_classifier_tests(classifier)
    scores["集成测试"], scores["集成失败"] = run_integration_tests(classifier, agent)
    scores["边界容错"], scores["边界失败"] = run_edge_cases(agent)
    scores["LLM判断"], scores["LLM失败"] = run_llm_judgment_tests(agent)

    total_pass = sum(scores[k] for k in ["工具函数", "分类器", "集成测试", "边界容错", "LLM判断"])
    total_fail = sum(scores[k] for k in ["工具失败", "分类失败", "集成失败", "边界失败", "LLM失败"])
    total = total_pass + total_fail

    print("=" * 65)
    print(f"  最终结果: {total_pass}/{total} 通过 ({total_fail} 失败)")
    print("=" * 65)
    print(f"  工具函数: {scores['工具函数']}/{scores['工具函数']+scores['工具失败']}")
    print(f"  分类器:   {scores['分类器']}/{scores['分类器']+scores['分类失败']}")
    print(f"  集成测试: {scores['集成测试']}/{scores['集成测试']+scores['集成失败']}")
    print(f"  边界容错: {scores['边界容错']}/{scores['边界容错']+scores['边界失败']}")
    print(f"  LLM判断:  {scores['LLM判断']}/{scores['LLM判断']+scores['LLM失败']}")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
