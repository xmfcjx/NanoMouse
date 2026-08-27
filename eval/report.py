"""
报告生成模块
"""
import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def ensure_results_dir():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)


def save_results(name, data):
    """保存结果到 JSON 文件"""
    ensure_results_dir()
    path = os.path.join(RESULTS_DIR, f"{name}_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存到 {path}")


def print_rag_summary(results):
    """打印 RAG 消融实验摘要"""
    print("\n" + "=" * 60)
    print("RAG 消融实验摘要")
    print("=" * 60)
    print(f"{'配置':<20} {'召回率':>10} {'准确率':>10} {'延迟(s)':>10}")
    print("-" * 60)
    for name, data in results.items():
        print(f"{name:<20} {data['avg_recall']:>10.2%} {data['avg_quality']:>10.2%} {data['avg_latency']:>10.2f}")

    configs = list(results.values())
    if len(configs) >= 2:
        base = configs[0]["avg_recall"]
        full = configs[-1]["avg_recall"]
        improvement = (full - base) / base * 100 if base > 0 else 0
        print(f"\n结论: 混合检索召回率从 {base:.2%} 提升至 {full:.2%}，提升 {improvement:.1f}%")


def print_agent_summary(results):
    """打印 Agent 对比摘要"""
    print("\n" + "=" * 60)
    print("Agent 对比摘要")
    print("=" * 60)
    print(f"{'模式':<15} {'平均延迟':>12} {'准确率':>10}")
    print("-" * 40)
    print(f"{'直通车':<15} {results['direct_avg_latency']:>10.3f}s {results['direct_accuracy']:>10.1%}")
    print(f"{'ReAct':<15} {results['react_avg_latency']:>10.3f}s {results['react_accuracy']:>10.1%}")
    print(f"\n结论: 直通模式速度提升 {results['speedup']:.1f}x")


def print_routing_summary(results):
    """打印路由测试摘要"""
    print("\n" + "=" * 60)
    print("路由准确率摘要")
    print("=" * 60)
    print(f"类型准确率: {results['type_accuracy']:.1%} ({int(results['type_accuracy'] * results['total'])}/{results['total']})")
    print(f"子类型准确率: {results['subtype_accuracy']:.1%} ({int(results['subtype_accuracy'] * results['total'])}/{results['total']})")

    if results.get("errors"):
        print(f"\n错误用例 ({len(results['errors'])}):")
        for e in results["errors"]:
            print(f"  Q: {e['question'][:30]} | 期望: {e['expected']} | 预测: {e['predicted']}")


def generate_final_report():
    """生成最终 Markdown 报告"""
    ensure_results_dir()

    md = "# NanoChat-Lab Benchmark Report\n\n"

    rag_path = os.path.join(RESULTS_DIR, "rag_ablation_results.json")
    agent_path = os.path.join(RESULTS_DIR, "agent_benchmark_results.json")
    routing_path = os.path.join(RESULTS_DIR, "routing_results.json")

    if os.path.exists(rag_path):
        with open(rag_path, "r", encoding="utf-8") as f:
            rag = json.load(f)
        md += "## 1. RAG 消融实验\n\n"
        md += "| 配置 | 召回率 | 准确率 | 延迟(s) |\n"
        md += "|------|--------|--------|--------|\n"
        for name, data in rag.items():
            md += f"| {name} | {data['avg_recall']:.2%} | {data['avg_quality']:.2%} | {data['avg_latency']:.2f} |\n"
        md += "\n"

    if os.path.exists(agent_path):
        with open(agent_path, "r", encoding="utf-8") as f:
            agent = json.load(f)
        md += "## 2. Agent 对比\n\n"
        md += f"- 直通车延迟: {agent['direct_avg_latency']:.3f}s\n"
        md += f"- ReAct延迟: {agent['react_avg_latency']:.3f}s\n"
        md += f"- 速度提升: {agent['speedup']:.1f}x\n\n"

    if os.path.exists(routing_path):
        with open(routing_path, "r", encoding="utf-8") as f:
            routing = json.load(f)
        md += "## 3. 路由准确率\n\n"
        md += f"- 类型准确率: {routing['type_accuracy']:.1%}\n"
        md += f"- 子类型准确率: {routing['subtype_accuracy']:.1%}\n\n"

    report_path = os.path.join(RESULTS_DIR, "BENCHMARK_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\n最终报告已保存到 {report_path}")
