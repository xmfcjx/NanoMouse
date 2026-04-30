"""
工具数量消融实验
测试不同工具数量对 ReAct 准确率的影响
用法:
  python test_tool_ablation.py              # 完整测试
  python test_tool_ablation.py --quick      # 快速测试(每种配置20条)
"""
import time
import re
import datetime
import os
import argparse
import json
from core.llm import LLM
from core.ReActAgent import ReActAgent
from eval.test_cases import AGENT_TEST_CASES
from config.config import get_config


TOOL_CONFIGS = {
    "3_tools": ["calc", "time", "weather"],
    "5_tools": ["calc", "time", "weather", "solve", "convert"],
    "8_tools": ["calc", "time", "weather", "solve", "convert", "base_convert", "weekday", "date"],
    "10_tools": ["calc", "time", "weather", "solve", "convert", "base_convert", "weekday", "date", "str_tools", "statistics"],
    "12_tools": ["calc", "time", "weather", "solve", "convert", "base_convert", "weekday", "date", "str_tools", "statistics", "random", "days_between"],
}


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


def run_tool_ablation(quick_test=False, test_n=20):
    print("\n" + "=" * 70)
    print("工具数量消融实验")
    print("=" * 70)
    print("测试假设: 工具数量越多，ReAct 准确率越低（选择困难）")
    print("=" * 70)
    
    print("\n初始化模型...")
    llm = LLM()
    
    all_results = {}
    
    for config_name, tools in TOOL_CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"配置: {config_name} ({len(tools)} 个工具)")
        print(f"工具: {', '.join(tools)}")
        print("=" * 70)
        
        filtered_cases = [c for c in AGENT_TEST_CASES if c["tool"] in tools]
        
        if quick_test:
            filtered_cases = filtered_cases[:test_n]
        
        if not filtered_cases:
            print(f"  跳过 {config_name}: 没有匹配的测试用例")
            continue
        
        print(f"  测试用例数: {len(filtered_cases)}")
        
        agent = ReActAgent(llm)
        
        correct_direct = 0
        correct_react = 0
        total = len(filtered_cases)
        
        direct_latencies = []
        react_latencies = []
        
        tool_results = {}
        
        for case in filtered_cases:
            tool = case["tool"]
            if tool not in tool_results:
                tool_results[tool] = {"direct": 0, "react": 0, "total": 0}
            tool_results[tool]["total"] += 1
            
            start = time.time()
            direct_answer = agent.direct_call(case["tool"], case["input"])
            direct_latency = time.time() - start
            direct_correct = check_result(direct_answer, case)
            if direct_correct:
                correct_direct += 1
                tool_results[tool]["direct"] += 1
            direct_latencies.append(direct_latency)
            
            start = time.time()
            react_answer = agent.run(case["question"])
            react_latency = time.time() - start
            final_match = re.search(r"Final Answer:\s*(.*)", react_answer, re.IGNORECASE | re.DOTALL)
            react_output = final_match.group(1).strip() if final_match else react_answer.strip()
            react_correct = check_result(react_output, case)
            if react_correct:
                correct_react += 1
                tool_results[tool]["react"] += 1
            react_latencies.append(react_latency)
            
            print(f"  [{case['id']}] {case['question'][:40]}...")
            print(f"    直通车: {'✓' if direct_correct else '✗'} | ReAct: {'✓' if react_correct else '✗'}")
        
        direct_acc = correct_direct / total if total > 0 else 0
        react_acc = correct_react / total if total > 0 else 0
        avg_direct = sum(direct_latencies) / len(direct_latencies) if direct_latencies else 0
        avg_react = sum(react_latencies) / len(react_latencies) if react_latencies else 0
        
        all_results[config_name] = {
            "num_tools": len(tools),
            "tools": tools,
            "total_cases": total,
            "direct": {
                "correct": correct_direct,
                "accuracy": round(direct_acc, 4),
                "avg_latency": round(avg_direct, 3),
            },
            "react": {
                "correct": correct_react,
                "accuracy": round(react_acc, 4),
                "avg_latency": round(avg_react, 3),
            },
            "per_tool": tool_results,
        }
        
        print(f"\n  结果: 直通车 {direct_acc*100:.1f}% | ReAct {react_acc*100:.1f}%")
    
    os.makedirs("eval/results", exist_ok=True)
    
    output = {
        "experiment": "tool_ablation",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "quick_test": quick_test,
        "results": all_results,
    }
    
    with open("eval/results/tool_ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print("消融实验总结")
    print("=" * 70)
    
    print(f"\n{'配置':<15} {'工具数':<8} {'直通车准确率':<15} {'ReAct准确率':<15} {'准确率差':<12}")
    print("-" * 65)
    
    for config_name, result in all_results.items():
        d_acc = result["direct"]["accuracy"] * 100
        r_acc = result["react"]["accuracy"] * 100
        delta = d_acc - r_acc
        print(f"{config_name:<15} {result['num_tools']:<8} {d_acc:.1f}%{'':<10} {r_acc:.1f}%{'':<10} {delta:+.1f}%")
    
    print("\n" + "=" * 70)
    print("关键发现")
    print("=" * 70)
    
    if len(all_results) >= 2:
        configs = list(all_results.keys())
        first = all_results[configs[0]]
        last = all_results[configs[-1]]
        
        first_react_acc = first["react"]["accuracy"] * 100
        last_react_acc = last["react"]["accuracy"] * 100
        
        if last_react_acc < first_react_acc:
            print(f"\n✓ 假设成立: 工具数量从 {first['num_tools']} 增加到 {last['num_tools']} 时")
            print(f"  ReAct 准确率从 {first_react_acc:.1f}% 下降到 {last_react_acc:.1f}%")
        else:
            print(f"\n✗ 假设不成立: 工具数量增加未导致准确率下降")
            print(f"  {first['num_tools']} 工具: {first_react_acc:.1f}%")
            print(f"  {last['num_tools']} 工具: {last_react_acc:.1f}%")
    
    print(f"\n结果已保存到: eval/results/tool_ablation_results.json")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="工具数量消融实验")
    parser.add_argument("--quick", action="store_true", help="快速测试模式")
    parser.add_argument("--n", type=int, default=20, help="快速测试模式下每种配置的测试数量")
    args = parser.parse_args()
    
    run_tool_ablation(quick_test=args.quick, test_n=args.n)
