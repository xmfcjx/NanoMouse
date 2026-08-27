"""Evaluate only route selection on a public holdout JSONL."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edgeops.router import ConfidenceRouter


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: List[float], pct: int) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = min(len(values) - 1, int(round((pct / 100) * (len(values) - 1))))
    return round(values[index], 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--output", default="eval/results/public_holdout/route_eval_summary.json")
    parser.add_argument("--predictions", default="eval/results/public_holdout/route_predictions.jsonl")
    parser.add_argument(
        "--print-errors",
        action="store_true",
        help="Print wrong examples. Do not use on a holdout set before final reporting.",
    )
    args = parser.parse_args()

    rows = load_jsonl(Path(args.eval_file))
    router = ConfidenceRouter()
    records = []
    output = Path(args.output)
    predictions = Path(args.predictions)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.parent.mkdir(parents=True, exist_ok=True)

    with predictions.open("w", encoding="utf-8") as handle:
        for row in rows:
            start = time.perf_counter()
            decision = router.route(row["query"])
            latency_ms = (time.perf_counter() - start) * 1000
            ok = decision.route.value == row["expected_route"]
            record = {
                "id": row.get("id"),
                "source": row.get("source"),
                "expected_route": row["expected_route"],
                "route": decision.route.value,
                "ok": ok,
                "latency_ms": latency_ms,
            }
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "eval_file": args.eval_file,
        "total": len(records),
        "route_accuracy": sum(item["ok"] for item in records) / len(records) if records else 0.0,
        "expected_distribution": dict(Counter(item["expected_route"] for item in records)),
        "route_distribution": dict(Counter(item["route"] for item in records)),
        "p50_latency_ms": percentile([item["latency_ms"] for item in records], 50),
        "p95_latency_ms": percentile([item["latency_ms"] for item in records], 95),
        "note": "Route-only public holdout. No tool-name/argument score without a model or BFCL-compatible tool planner.",
    }
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.print_errors:
        wrong = [item for item in records if not item["ok"]]
        print(json.dumps(wrong[:20], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
