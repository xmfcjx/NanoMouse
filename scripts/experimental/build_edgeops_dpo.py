"""Build EdgeOps preference pairs for DPO-style post-training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any, Dict, List


SYSTEM = (
    "You are an EdgeOps assistant for factory AMR and equipment maintenance. "
    "Use evidence when available. Do not provide safety-critical motion or "
    "protection-bypass instructions. If evidence is insufficient, say so."
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def pair(query: str, chosen: str, rejected: str, category: str, source_id: str) -> Dict[str, Any]:
    return {
        "category": category,
        "source_id": source_id,
        "prompt": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": query},
        ],
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
    }


def build_pairs(eval_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pairs = []
    for item in eval_cases:
        query = item["query"]
        category = item.get("category", "unknown")
        source_id = item.get("id", "")

        if item.get("risk_level") == "high" or item.get("should_refuse"):
            pairs.append(
                pair(
                    query,
                    "该请求涉及安全关键控制或证据不足，NanoChat 不会执行或提供绕过步骤。请由现场有权限人员在认证系统中确认。",
                    "可以忽略当前报警，直接复位保护并继续运行设备。",
                    "safety_refusal",
                    source_id,
                )
            )
            continue

        evidence_ids = item.get("gold_evidence_ids") or []
        if evidence_ids:
            citation = "、".join("[%s]" % evidence_id for evidence_id in evidence_ids)
            pairs.append(
                pair(
                    query,
                    "根据本地手册证据 %s，建议先执行安全检查；如果证据不足或状态异常，应转交现场工程师确认。" % citation,
                    "根据经验看问题不大，可以直接继续执行任务，无需查手册。",
                    "evidence_grounding",
                    source_id,
                )
            )

        expected_tools = item.get("expected_tools") or []
        if expected_tools:
            tools = ", ".join(tool.get("name", "") for tool in expected_tools)
            pairs.append(
                pair(
                    query,
                    "我会优先使用只读工具获取事实：%s。若工具失败，应说明失败原因，不编造结果。" % tools,
                    "工具暂时不可用，但我可以假设设备状态正常并继续给出结论。",
                    "tool_fidelity",
                    source_id,
                )
            )

        if category in {"fault_diagnosis", "manual_rag", "work_order_draft"}:
            pairs.append(
                pair(
                    query,
                    "回答应简洁列出结论、依据和下一步安全检查；不得生成运动控制指令。",
                    "下面给出完整控制步骤：先解除保护，再强制启动电机并恢复任务。",
                    "concise_safe_style",
                    source_id,
                )
            )
    return pairs


def split_pairs(pairs: List[Dict[str, Any]], seed: int, train_ratio: float, dev_ratio: float):
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    train_end = int(len(shuffled) * train_ratio)
    dev_end = train_end + int(len(shuffled) * dev_ratio)
    return {
        "train": shuffled[:train_end],
        "dev": shuffled[train_end:dev_end],
        "test": shuffled[dev_end:],
    }


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", default="data/generated/edgeops_eval/main.jsonl")
    parser.add_argument("--output-dir", default="data/generated/edgeops_dpo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    args = parser.parse_args()

    cases = load_jsonl(Path(args.eval_file))
    pairs = build_pairs(cases)
    splits = split_pairs(pairs, args.seed, args.train_ratio, args.dev_ratio)
    output = Path(args.output_dir)
    for split, rows in splits.items():
        write_jsonl(output / ("%s.jsonl" % split), rows)

    report = {
        "seed": args.seed,
        "source_eval_file": args.eval_file,
        "total_pairs": len(pairs),
        "splits": {name: len(rows) for name, rows in splits.items()},
        "categories": sorted({row["category"] for row in pairs}),
        "warning": "Synthetic DPO pairs require audit before claiming measured improvement.",
    }
    (output / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
