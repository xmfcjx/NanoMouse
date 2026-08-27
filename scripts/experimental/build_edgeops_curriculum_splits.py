"""Build independent EdgeOps route/tool and safety/refusal datasets.

The split is source-group based: cases derived from the same public benchmark
source_id stay in the same split. This avoids leaking paraphrased variants of
one source item across train/dev/test.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List


SYSTEM = (
    "You are an EdgeOps assistant. Return valid JSON only. "
    "Use tools for device facts, use evidence for manuals, and reject unsafe "
    "motion or protection-bypass requests."
)

DPO_SYSTEM = (
    "You are an EdgeOps assistant for factory AMR and equipment maintenance. "
    "Do not provide safety-critical motion commands or protection-bypass steps. "
    "If evidence is insufficient, say so."
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_rows(rows: Iterable[Dict[str, Any]]) -> str:
    text = "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True)
        for row in rows
    )
    return sha256_text(text)


def stable_source_key(row: Dict[str, Any]) -> str:
    source = row.get("source", "unknown")
    source_id = row.get("source_id") or row.get("id", "")
    return "%s:%s" % (source, source_id)


def is_safety_refusal(row: Dict[str, Any]) -> bool:
    return (
        row.get("risk_level") == "high"
        or bool(row.get("should_refuse"))
        or row.get("category") in {"safety_reject", "no_answer"}
        or row.get("expected_route") == "safety_reject"
    )


def split_by_source(
    rows: List[Dict[str, Any]],
    seed: int,
    train_ratio: float,
    dev_ratio: float,
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[stable_source_key(row)].append(row)

    keys = list(grouped)
    rng = random.Random(seed)
    rng.shuffle(keys)
    train_end = int(len(keys) * train_ratio)
    dev_end = train_end + int(len(keys) * dev_ratio)
    split_keys = {
        "train": set(keys[:train_end]),
        "dev": set(keys[train_end:dev_end]),
        "test": set(keys[dev_end:]),
    }
    return {
        split: [row for key in keys if key in selected for row in grouped[key]]
        for split, selected in split_keys.items()
    }


ROUTE_TOOL_HOLDOUT_FAMILIES = {
    "status_en",
    "fault_en",
    "maintenance_en",
    "location_en",
    "tool_en",
    "draft_en",
    "manual_work_order",
    "clarify_missing_device",
}

SAFETY_HOLDOUT_FAMILIES = {
    "safety_bypass",
}


def family_key(row: Dict[str, Any]) -> str:
    return row.get("transform_id") or row.get("category") or "unknown"


def split_family_holdout(
    rows: List[Dict[str, Any]],
    holdout_families: set[str],
    seed: int,
    train_ratio: float,
) -> Dict[str, List[Dict[str, Any]]]:
    test = [row for row in rows if family_key(row) in holdout_families]
    train_dev_candidates = [row for row in rows if family_key(row) not in holdout_families]
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in train_dev_candidates:
        grouped[stable_source_key(row)].append(row)
    keys = list(grouped)
    rng = random.Random(seed)
    rng.shuffle(keys)
    train_end = int(len(keys) * train_ratio)
    train_keys = set(keys[:train_end])
    return {
        "train": [row for key in keys if key in train_keys for row in grouped[key]],
        "dev": [row for key in keys if key not in train_keys for row in grouped[key]],
        "test": test,
    }


def normalize_action(tool: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "action": tool.get("name") or tool.get("tool"),
        "arguments": tool.get("arguments") or {},
    }


def route_tool_sft(row: Dict[str, Any]) -> Dict[str, Any]:
    expected_tools = row.get("expected_tools") or []
    if expected_tools:
        if len(expected_tools) == 1:
            assistant: Dict[str, Any] = normalize_action(expected_tools[0])
        else:
            assistant = {"steps": [normalize_action(tool) for tool in expected_tools]}
    else:
        route = row.get("expected_route", "manual_rag")
        assistant = {
            "route": route,
            "policy": "retrieve_evidence" if route == "manual_rag" else "answer_with_available_context",
        }
    return {
        "task": "route_tool",
        "source_id": row.get("id"),
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": row["query"]},
            {
                "role": "assistant",
                "content": json.dumps(assistant, ensure_ascii=False, sort_keys=True),
            },
        ],
    }


def safety_sft(row: Dict[str, Any]) -> Dict[str, Any]:
    reason = "unsafe_request" if row.get("risk_level") == "high" else "insufficient_evidence"
    assistant = {
        "action": "reject",
        "reason": reason,
        "message": "该请求涉及安全关键控制或证据不足，不能执行。请由现场有权限人员确认。",
    }
    return {
        "task": "safety_refusal",
        "source_id": row.get("id"),
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": row["query"]},
            {
                "role": "assistant",
                "content": json.dumps(assistant, ensure_ascii=False, sort_keys=True),
            },
        ],
    }


def safety_dpo(row: Dict[str, Any]) -> Dict[str, Any]:
    if row.get("risk_level") == "high":
        chosen = "该请求涉及安全关键控制或绕过保护，NanoChat 不会执行或提供操作步骤。请使用认证控制系统并由现场人员确认。"
        rejected = "可以先屏蔽告警并继续运行，等任务结束后再检查。"
        category = "unsafe_control_refusal"
    else:
        chosen = "本地证据不足，NanoChat 不会编造结论。请提供正确设备编号、故障码或转交现场工程师确认。"
        rejected = "虽然没有查到证据，但可以按经验判断设备状态正常。"
        category = "insufficient_evidence_refusal"
    return {
        "category": category,
        "source_id": row.get("id"),
        "prompt": [
            {"role": "system", "content": DPO_SYSTEM},
            {"role": "user", "content": row["query"]},
        ],
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
    }


def dedupe(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    result = []
    for row in rows:
        key = row.get("id") or sha256_text(json.dumps(row, ensure_ascii=False, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def write_track(
    output_root: Path,
    name: str,
    splits: Dict[str, List[Dict[str, Any]]],
    sft_builder,
    dpo_builder=None,
) -> Dict[str, Any]:
    track_root = output_root / name
    report = {
        "splits": {},
        "category_counts": {},
        "hashes": {},
        "source_group_counts": {},
    }
    for split, rows in splits.items():
        write_jsonl(track_root / "eval" / ("%s.jsonl" % split), rows)
        sft_rows = [sft_builder(row) for row in rows]
        write_jsonl(track_root / "sft" / ("%s.jsonl" % split), sft_rows)
        if dpo_builder is not None:
            dpo_rows = [dpo_builder(row) for row in rows]
            write_jsonl(track_root / "dpo" / ("%s.jsonl" % split), dpo_rows)
            report["splits"][split] = {
                "eval": len(rows),
                "sft": len(sft_rows),
                "dpo": len(dpo_rows),
            }
            report["hashes"]["%s_dpo" % split] = sha256_rows(dpo_rows)
        else:
            report["splits"][split] = {"eval": len(rows), "sft": len(sft_rows)}
        report["category_counts"][split] = dict(Counter(row.get("category") for row in rows))
        report["source_group_counts"][split] = len({stable_source_key(row) for row in rows})
        report["hashes"]["%s_eval" % split] = sha256_rows(rows)
        report["hashes"]["%s_sft" % split] = sha256_rows(sft_rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources",
        nargs="+",
        default=[
            "data/generated/edgeops_bfcl_derived/pool.jsonl",
            "data/generated/edgeops_apibank_derived/pool.jsonl",
        ],
    )
    parser.add_argument("--output-dir", default="data/generated/edgeops_curriculum")
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--dev-ratio", type=float, default=0.15)
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for source in args.sources:
        rows.extend(load_jsonl(Path(source)))
    rows = dedupe(rows)

    safety_rows = [row for row in rows if is_safety_refusal(row)]
    route_tool_rows = [row for row in rows if not is_safety_refusal(row)]

    route_splits = split_by_source(route_tool_rows, args.seed, args.train_ratio, args.dev_ratio)
    safety_splits = split_by_source(safety_rows, args.seed + 17, args.train_ratio, args.dev_ratio)
    route_family_splits = split_family_holdout(
        route_tool_rows,
        ROUTE_TOOL_HOLDOUT_FAMILIES,
        args.seed + 101,
        args.train_ratio,
    )
    safety_family_splits = split_family_holdout(
        safety_rows,
        SAFETY_HOLDOUT_FAMILIES,
        args.seed + 117,
        args.train_ratio,
    )

    output = Path(args.output_dir)
    route_report = write_track(output, "route_tool", route_splits, route_tool_sft)
    safety_report = write_track(output, "safety_refusal", safety_splits, safety_sft, safety_dpo)
    route_family_report = write_track(
        output,
        "route_tool_family_holdout",
        route_family_splits,
        route_tool_sft,
    )
    safety_family_report = write_track(
        output,
        "safety_refusal_family_holdout",
        safety_family_splits,
        safety_sft,
        safety_dpo,
    )

    report = {
        "seed": args.seed,
        "sources": args.sources,
        "total_cases": len(rows),
        "route_tool_cases": len(route_tool_rows),
        "safety_refusal_cases": len(safety_rows),
        "split_policy": "source_id grouped split; no source group appears in multiple splits within the same track",
        "family_holdout_policy": {
            "route_tool_test_families": sorted(ROUTE_TOOL_HOLDOUT_FAMILIES),
            "safety_refusal_test_families": sorted(SAFETY_HOLDOUT_FAMILIES),
            "description": "selected transform families are excluded from train/dev and used only as stricter test sets",
        },
        "route_tool": route_report,
        "safety_refusal": safety_report,
        "route_tool_family_holdout": route_family_report,
        "safety_refusal_family_holdout": safety_family_report,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
