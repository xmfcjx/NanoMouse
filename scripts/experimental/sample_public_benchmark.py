"""Sample a public benchmark JSONL without generating new questions.

The script intentionally prints only aggregate metadata and hashes. Do not
inspect sampled items when the output is used as a holdout set.
"""

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
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def get_nested(row: Dict[str, Any], field: str, default: Any = None) -> Any:
    value: Any = row
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def normalize_row(
    row: Dict[str, Any],
    id_field: str,
    query_field: str,
    route_field: str | None,
    tools_field: str | None,
    default_route: str | None,
    source_name: str,
) -> Dict[str, Any]:
    sample_id = str(get_nested(row, id_field, row["_line_no"]))
    query = get_nested(row, query_field)
    if query is None:
        raise ValueError("Missing query field %s in row %s" % (query_field, sample_id))
    expected_route = (
        get_nested(row, route_field) if route_field else default_route
    )
    if not expected_route:
        raise ValueError("Missing route label for row %s" % sample_id)
    expected_tools = get_nested(row, tools_field, []) if tools_field else []
    return {
        "id": "%s:%s" % (source_name, sample_id),
        "source": source_name,
        "source_id": sample_id,
        "query": query,
        "expected_route": expected_route,
        "expected_tools": expected_tools,
        "should_refuse": bool(get_nested(row, "should_refuse", False)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-jsonl", required=True, help="Downloaded public benchmark JSONL.")
    parser.add_argument("--source-name", required=True, help="Benchmark name/version, e.g. bfcl-v3-live.")
    parser.add_argument("--output", default="data/generated/public_holdout/sample.jsonl")
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--query-field", default="query")
    parser.add_argument("--route-field", default=None)
    parser.add_argument("--tools-field", default=None)
    parser.add_argument("--default-route", default=None)
    args = parser.parse_args()

    source_path = Path(args.source_jsonl)
    rows = load_jsonl(source_path)
    if args.sample_size > len(rows):
        raise ValueError("sample-size %s > source rows %s" % (args.sample_size, len(rows)))

    rng = random.Random(args.seed)
    sampled = rng.sample(rows, args.sample_size)
    normalized = [
        normalize_row(
            row,
            id_field=args.id_field,
            query_field=args.query_field,
            route_field=args.route_field,
            tools_field=args.tools_field,
            default_route=args.default_route,
            source_name=args.source_name,
        )
        for row in sampled
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in normalized:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    sampled_ids = "\n".join(sorted(row["id"] for row in normalized))
    report = {
        "source_name": args.source_name,
        "source_jsonl": str(source_path),
        "source_sha256": sha256_file(source_path),
        "seed": args.seed,
        "sample_size": args.sample_size,
        "sampled_ids_sha256": sha256_text(sampled_ids),
        "output": str(output),
        "policy": "Do not inspect sampled holdout rows before final evaluation.",
    }
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
