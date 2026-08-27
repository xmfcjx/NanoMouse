"""
Agent 直通车 vs ReAct 循环对比实验
用法:
  python test_agent_benchmark.py              # 完整测试
  python test_agent_benchmark.py --quick      # 快速测试(默认10条)
  python test_agent_benchmark.py --quick --n 5  # 快速测试(前5条)
"""
import time
import re
import datetime
import os
import argparse
from core.llm import LLM
from core.ReActAgent import ReActAgent
from eval.test_cases import AGENT_TEST_CASES
from config.config import get_config


def check_result(result, case):
    check = case["check_type"]
    if check == "exact":
        return case["expected"].lower() in result.lower()
    elif check == "contains":
        return case["expected"].lower() in result.lower()
    elif check == "time_format":
        try:
            datetime.datetime.strptime(result.strip()[:19], "%Y-%m-%d %H:%M:%S")
            return True
        except:
            return False
    elif check == "weekday":
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        return any(d in result for d in days)
    elif check == "weather":
        return "°C" in result or "°F" in result
    return False


def run_agent_benchmark(quick_test=False, test_n=10):
    test_cases = AGENT_TEST_CASES[:test_n] if quick_test else AGENT_TEST_CASES
    
    print("\n" + "=" * 70)
    print("Agent 直通车 vs ReAct 循环对比实验")
    print("=" * 70)
    if quick_test:
        print(f"[QUICK TEST MODE] Testing {test_n} cases")
    print(f"Total test cases: {len(test_cases)}")
    print("=" * 70)

    print("\n初始化组件...")
    llm = LLM()
    agent = ReActAgent(llm)

    direct_results = []
    react_results = []
    badcases = []
    
    tool_stats_direct = {}
    tool_stats_react = {}
    
    badcases.append("=" * 70)
    badcases.append("Agent Benchmark Badcase Analysis")
    badcases.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if quick_test:
        badcases.append(f"[QUICK TEST MODE] {test_n} cases")
    badcases.append("=" * 70)

    for case in test_cases:
        print(f"\n  [{case['id']}] {case['question']}")

        tool = case["tool"]
        if tool not in tool_stats_direct:
            tool_stats_direct[tool] = {"correct": 0, "total": 0}
            tool_stats_react[tool] = {"correct": 0, "total": 0}
        tool_stats_direct[tool]["total"] += 1
        tool_stats_react[tool]["total"] += 1

        start = time.time()
        direct_answer = agent.direct_call(case["tool"], case["input"])
        direct_latency = time.time() - start
        direct_correct = check_result(direct_answer, case)
        if direct_correct:
            tool_stats_direct[tool]["correct"] += 1
        direct_results.append({
            "id": case["id"],
            "tool": tool,
            "latency": direct_latency,
            "correct": direct_correct,
            "answer": direct_answer[:100],
        })
        print(f"    直通车: {direct_latency:.3f}s | {'✓' if direct_correct else '✗'} | {direct_answer[:60]}")

        start = time.time()
        react_answer = agent.run(case["question"])
        react_latency = time.time() - start
        final_match = re.search(r"Final Answer:\s*(.*)", react_answer, re.IGNORECASE | re.DOTALL)
        react_output = final_match.group(1).strip() if final_match else react_answer.strip()
        react_correct = check_result(react_output, case)
        if react_correct:
            tool_stats_react[tool]["correct"] += 1
        react_results.append({
            "id": case["id"],
            "tool": tool,
            "latency": react_latency,
            "correct": react_correct,
            "answer": react_output[:100],
        })
        print(f"    ReAct:   {react_latency:.3f}s | {'✓' if react_correct else '✗'} | {react_output[:60]}")

        if not direct_correct or not react_correct:
            badcases.append(f"\n--- Case #{case['id']} [{tool}]: {case['question']} ---")
            badcases.append(f"Expected: {case.get('expected', 'N/A')}")
            if not direct_correct:
                badcases.append(f"直通车 FAIL: {direct_answer[:100]}")
            if not react_correct:
                badcases.append(f"ReAct FAIL: {react_output[:100]}")

    direct_avg = sum(r["latency"] for r in direct_results) / len(direct_results)
    react_avg = sum(r["latency"] for r in react_results) / len(react_results)
    direct_acc = sum(1 for r in direct_results if r["correct"]) / len(direct_results)
    react_acc = sum(1 for r in react_results if r["correct"]) / len(react_results)
    speedup = react_avg / direct_avg if direct_avg > 0 else 0

    direct_min = min(r["latency"] for r in direct_results)
    direct_max = max(r["latency"] for r in direct_results)
    react_min = min(r["latency"] for r in react_results)
    react_max = max(r["latency"] for r in react_results)

    results = {
        "total_cases": len(test_cases),
        "direct": {
            "avg_latency": round(direct_avg, 3),
            "min_latency": round(direct_min, 3),
            "max_latency": round(direct_max, 3),
            "accuracy": round(direct_acc, 4),
            "correct": sum(1 for r in direct_results if r["correct"]),
        },
        "react": {
            "avg_latency": round(react_avg, 3),
            "min_latency": round(react_min, 3),
            "max_latency": round(react_max, 3),
            "accuracy": round(react_acc, 4),
            "correct": sum(1 for r in react_results if r["correct"]),
        },
        "speedup": round(speedup, 1),
        "tool_stats": {
            "direct": tool_stats_direct,
            "react": tool_stats_react,
        }
    }

    os.makedirs("eval/results", exist_ok=True)
    
    import json
    with open("eval/results/agent_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open("eval/results/agent_badcase_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(badcases))

    print("\n" + "=" * 70)
    print("Final Results Summary")
    print("=" * 70)
    
    print(f"\n{'Metric':<20} {'Direct':<15} {'ReAct':<15} {'Delta':<15}")
    print("-" * 65)
    print(f"{'Avg Latency':<20} {direct_avg:.3f}s{'':<8} {react_avg:.3f}s{'':<8} {speedup:.1f}x faster")
    print(f"{'Min Latency':<20} {direct_min:.3f}s{'':<8} {react_min:.3f}s{'':<8}")
    print(f"{'Max Latency':<20} {direct_max:.3f}s{'':<8} {react_max:.3f}s{'':<8}")
    print(f"{'Accuracy':<20} {direct_acc*100:.1f}%{'':<9} {react_acc*100:.1f}%{'':<9} {(direct_acc-react_acc)*100:+.1f}%")
    print(f"{'Correct/Total':<20} {results['direct']['correct']}/{len(test_cases)}{'':<10} {results['react']['correct']}/{len(test_cases)}{'':<10}")

    print("\n" + "=" * 70)
    print("Per-Tool Statistics")
    print("=" * 70)
    
    print(f"\n{'Tool':<15} {'Direct Acc':<15} {'ReAct Acc':<15} {'Delta':<15}")
    print("-" * 60)
    for tool in sorted(tool_stats_direct.keys()):
        d_stats = tool_stats_direct[tool]
        r_stats = tool_stats_react[tool]
        d_acc = d_stats["correct"] / d_stats["total"] * 100 if d_stats["total"] > 0 else 0
        r_acc = r_stats["correct"] / r_stats["total"] * 100 if r_stats["total"] > 0 else 0
        print(f"{tool:<15} {d_acc:.1f}% ({d_stats['correct']}/{d_stats['total']}){'':<5} {r_acc:.1f}% ({r_stats['correct']}/{r_stats['total']}){'':<5} {d_acc-r_acc:+.1f}%")

    print("\n" + "=" * 70)
    print("Key Findings")
    print("=" * 70)
    print(f"\n1. 直通车相比 ReAct 加速 {speedup:.1f}x")
    print(f"2. 直通车准确率: {direct_acc*100:.1f}% | ReAct准确率: {react_acc*100:.1f}%")
    print(f"3. 直通车延迟范围: {direct_min:.3f}s - {direct_max:.3f}s")
    print(f"4. ReAct延迟范围: {react_min:.3f}s - {react_max:.3f}s")

    print(f"\n结果已保存到: eval/results/agent_benchmark_results.json")
    print(f"Badcase日志: eval/results/agent_badcase_log.txt")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent 直通车 vs ReAct 对比实验")
    parser.add_argument("--quick", action="store_true", help="快速测试模式")
    parser.add_argument("--n", type=int, default=10, help="快速测试模式下的问题数量")
    args = parser.parse_args()
    
    run_agent_benchmark(quick_test=args.quick, test_n=args.n)
