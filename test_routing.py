"""
路由准确率测试
用法:
  python test_routing.py              # 完整测试
  python test_routing.py --quick      # 快速测试(前20条)
  python test_routing.py --quick --n 10  # 快速测试(前10条)
"""
import os
import json
import argparse
from core.input_classifier import InputClassifier
from eval.test_cases import ROUTING_TEST_CASES


def run_routing_test(quick_test=False, test_n=20):
    test_cases = ROUTING_TEST_CASES[:test_n] if quick_test else ROUTING_TEST_CASES
    
    print("\n" + "=" * 70)
    print("路由准确率测试")
    print("=" * 70)
    if quick_test:
        print(f"[QUICK TEST MODE] Testing {test_n} cases")
    print(f"Total test cases: {len(test_cases)}")
    print("=" * 70)

    print("\n初始化路由分类器...")
    classifier = InputClassifier()

    correct_type = 0
    correct_subtype = 0
    total = len(test_cases)
    errors = []
    
    type_stats = {}
    subtype_stats = {}

    for case in test_cases:
        result = classifier.classify(case["question"])
        pred_type = result["type"]
        pred_subtype = result.get("subtype", "")
        expected_type = case["expected_type"]
        expected_subtype = case["expected_subtype"]

        if expected_type not in type_stats:
            type_stats[expected_type] = {"correct": 0, "total": 0}
        type_stats[expected_type]["total"] += 1
        
        if expected_subtype and expected_subtype not in subtype_stats:
            subtype_stats[expected_subtype] = {"correct": 0, "total": 0}
        if expected_subtype:
            subtype_stats[expected_subtype]["total"] += 1

        type_match = pred_type == expected_type
        subtype_match = pred_subtype == expected_subtype

        if type_match:
            correct_type += 1
            type_stats[expected_type]["correct"] += 1
        if subtype_match:
            correct_subtype += 1
            if expected_subtype:
                subtype_stats[expected_subtype]["correct"] += 1

        if not type_match or not subtype_match:
            errors.append({
                "question": case["question"],
                "expected": f"{expected_type}/{expected_subtype}",
                "predicted": f"{pred_type}/{pred_subtype}",
            })

        mark = "✓" if (type_match and subtype_match) else "✗"
        print(f"  {mark} {case['question'][:45]:<45} → {pred_type}/{pred_subtype:<15} (期望: {expected_type}/{expected_subtype})")

    type_acc = correct_type / total
    subtype_acc = correct_subtype / total

    results = {
        "total": total,
        "type_accuracy": round(type_acc, 4),
        "subtype_accuracy": round(subtype_acc, 4),
        "correct_type": correct_type,
        "correct_subtype": correct_subtype,
        "type_stats": {k: {"accuracy": v["correct"]/v["total"] if v["total"] > 0 else 0, **v} for k, v in type_stats.items()},
        "errors": errors,
    }

    os.makedirs("eval/results", exist_ok=True)
    with open("eval/results/routing_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("Final Results Summary")
    print("=" * 70)
    
    print(f"\n{'Metric':<25} {'Value':<15}")
    print("-" * 40)
    print(f"{'Total Cases':<25} {total}")
    print(f"{'Type Accuracy':<25} {type_acc*100:.2f}% ({correct_type}/{total})")
    print(f"{'Subtype Accuracy':<25} {subtype_acc*100:.2f}% ({correct_subtype}/{total})")
    print(f"{'Errors':<25} {len(errors)}")

    print("\n" + "=" * 70)
    print("Per-Type Statistics")
    print("=" * 70)
    
    print(f"\n{'Type':<15} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-" * 45)
    for t, stats in sorted(type_stats.items()):
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"{t:<15} {stats['correct']:<10} {stats['total']:<10} {acc:.1f}%")

    if errors:
        print("\n" + "=" * 70)
        print("Error Cases")
        print("=" * 70)
        for err in errors[:10]:
            print(f"\n  Q: {err['question']}")
            print(f"  Expected: {err['expected']}")
            print(f"  Predicted: {err['predicted']}")
        if len(errors) > 10:
            print(f"\n  ... and {len(errors) - 10} more errors")

    print(f"\n结果已保存到: eval/results/routing_results.json")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="路由准确率测试")
    parser.add_argument("--quick", action="store_true", help="快速测试模式")
    parser.add_argument("--n", type=int, default=20, help="快速测试模式下的问题数量")
    args = parser.parse_args()
    
    run_routing_test(quick_test=args.quick, test_n=args.n)
