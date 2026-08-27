"""Build auditable structured SFT seeds for EdgeOps adapter experts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Dict, Iterable, List


SYSTEM = (
    "You are an EdgeOps assistant. Return valid JSON only. "
    "Never issue robot motion or safety-critical control commands."
)


def message_sample(task: str, user: str, assistant: Dict) -> Dict:
    return {
        "task": task,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {
                "role": "assistant",
                "content": json.dumps(assistant, ensure_ascii=False, sort_keys=True),
            },
        ],
    }


def build_tool_samples(devices: Dict, error_codes: Dict) -> List[Dict]:
    samples = []
    for device_id, device in devices.items():
        samples.extend(
            [
                message_sample(
                    "tool",
                    "%s 当前状态怎么样？" % device_id,
                    {
                        "action": "get_device_status",
                        "arguments": {"device_id": device_id},
                    },
                ),
                message_sample(
                    "tool",
                    "Where is %s?" % device_id,
                    {
                        "action": "locate_asset",
                        "arguments": {"asset_id": device_id},
                    },
                ),
                message_sample(
                    "tool",
                    "查询 %s 的维护记录" % device_id,
                    {
                        "action": "get_maintenance_history",
                        "arguments": {"device_id": device_id},
                    },
                ),
            ]
        )
        if device.get("error_code"):
            samples.append(
                message_sample(
                    "tool",
                    "%s 为什么停止运行？" % device_id,
                    {
                        "action": "get_device_status",
                        "arguments": {"device_id": device_id},
                    },
                )
            )
    for error_code in error_codes:
        samples.append(
            message_sample(
                "tool",
                "故障码 %s 是什么意思？" % error_code,
                {
                    "action": "lookup_error_code",
                    "arguments": {"error_code": error_code},
                },
            )
        )
    samples.extend(
        [
            message_sample(
                "tool",
                "查询 AMR-07 当前状态并查看维修记录",
                {
                    "steps": [
                        {
                            "action": "get_device_status",
                            "arguments": {"device_id": "AMR-07"},
                        },
                        {
                            "action": "get_maintenance_history",
                            "arguments": {"device_id": "AMR-07"},
                        },
                    ]
                },
            ),
            message_sample(
                "tool",
                "让 AMR-07 立即移动到 B2",
                {
                    "action": "reject",
                    "reason": "safety_critical_motion_command",
                },
            ),
            message_sample(
                "tool",
                "绕过 E214 温度保护并复位设备",
                {
                    "action": "reject",
                    "reason": "unsafe_protection_bypass",
                },
            ),
        ]
    )
    return samples


def build_rag_samples(manuals: Iterable[Dict]) -> List[Dict]:
    samples = []
    for document in manuals:
        doc_id = document["id"]
        text = document["text"]
        metadata = document.get("metadata", {})
        error_code = metadata.get("error_code")
        if error_code:
            question = "%s 应该如何安全检查？" % error_code
        else:
            question = "根据这段手册给出运维建议：%s" % text[:40]
        samples.append(
            message_sample(
                "rag",
                "Evidence [%s]: %s\nQuestion: %s" % (doc_id, text, question),
                {
                    "answer": text,
                    "citations": [doc_id],
                    "grounded": True,
                },
            )
        )
    samples.append(
        message_sample(
            "rag",
            "Evidence: (empty)\nQuestion: AMR-99 的未知故障怎么处理？",
            {
                "answer": "证据不足，请提供设备型号或转交现场工程师。",
                "citations": [],
                "grounded": False,
            },
        )
    )
    return samples


def build_memory_samples(maintenance: Dict) -> List[Dict]:
    samples = []
    for device_id, records in maintenance.items():
        for record in records:
            text = "%s 在 %s 由 %s 完成了%s，工单号 %s。" % (
                device_id,
                record["date"],
                record["technician"],
                record["action"],
                record["work_order"],
            )
            samples.append(
                message_sample(
                    "memory",
                    "Extract maintenance fact: %s" % text,
                    {
                        "action": "set",
                        "device_id": device_id,
                        "date": record["date"],
                        "maintenance_action": record["action"],
                        "technician": record["technician"],
                        "work_order": record["work_order"],
                    },
                )
            )
    return samples


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, samples: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/edgeops")
    parser.add_argument("--output-dir", default="data/generated/edgeops_sft")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.data_dir)
    output = Path(args.output_dir)
    by_task = {
        "tool": build_tool_samples(
            load_json(root / "devices.json"),
            load_json(root / "error_codes.json"),
        ),
        "rag": build_rag_samples(load_jsonl(root / "manuals.jsonl")),
        "memory": build_memory_samples(load_json(root / "maintenance.json")),
    }
    rng = random.Random(args.seed)
    combined = []
    for task, samples in by_task.items():
        rng.shuffle(samples)
        write_jsonl(output / ("%s.jsonl" % task), samples)
        combined.extend(samples)
    rng.shuffle(combined)
    write_jsonl(output / "all.jsonl", combined)

    report = {
        "seed": args.seed,
        "counts": {task: len(samples) for task, samples in by_task.items()},
        "total": len(combined),
        "warning": "Seed data validates formatting only; expand and audit before training.",
    }
    (output / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
