"""Convert public BFCL simple split into NanoChat public holdout format."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Dict, List


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def extract_query(row: Dict[str, Any]) -> str:
    return row["question"][0][0]["content"]


def expected_tools(answer: Dict[str, Any]) -> List[Dict[str, Any]]:
    tools = []
    for item in answer.get("ground_truth", []):
        for name, arg_options in item.items():
            args = {
                key: values[0] if isinstance(values, list) and values else values
                for key, values in arg_options.items()
            }
            tools.append({"name": name, "arguments": args})
    return tools


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="data/public/bfcl/BFCL_v3_simple.questions.jsonl")
    parser.add_argument("--answers", default="data/public/bfcl/BFCL_v3_simple.answers.jsonl")
    parser.add_argument("--output", default="data/generated/public_holdout/bfcl_v3_simple_route.jsonl")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()

    questions_path = Path(args.questions)
    answers_path = Path(args.answers)
    questions = load_jsonl(questions_path)
    answers_by_id = {row["id"]: row for row in load_jsonl(answers_path)}
    rows = []
    for question in questions:
        answer = answers_by_id[question["id"]]
        rows.append(
            {
                "id": "bfcl_v3_simple:%s" % question["id"],
                "source": "BFCL_v3_simple",
                "source_id": question["id"],
                "query": extract_query(question),
                "expected_route": "tool_plan",
                "expected_tools": expected_tools(answer),
                "should_refuse": False,
            }
        )

    rng = random.Random(args.seed)
    sampled = rng.sample(rows, min(args.sample_size, len(rows)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in sampled:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    sampled_ids = "\n".join(sorted(row["id"] for row in sampled))
    report = {
        "source": "BFCL_v3_simple",
        "questions_sha256": sha256_file(questions_path),
        "answers_sha256": sha256_file(answers_path),
        "seed": args.seed,
        "source_rows": len(rows),
        "sample_size": len(sampled),
        "sampled_ids_sha256": sha256_text(sampled_ids),
        "output": str(output),
        "note": "Route-only holdout labels all BFCL function-calling queries as tool_plan.",
    }
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
