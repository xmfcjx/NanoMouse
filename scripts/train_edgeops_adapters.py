"""Train EdgeOps LoRA experts with the existing stable LoRA trainer."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


EXPERTS = {
    "route_tool": {
        "train": "route_tool/sft/train.jsonl",
        "dev": "route_tool/sft/dev.jsonl",
        "max_steps": 600,
    },
    "safety_refusal": {
        "train": "safety_refusal/sft/train.jsonl",
        "dev": "safety_refusal/sft/dev.jsonl",
        "max_steps": 400,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert", choices=["all", *EXPERTS], default="all")
    parser.add_argument("--mode", choices=["local", "gpu11", "cloud"], default="gpu11")
    parser.add_argument("--model-path", default="models/Qwen2.5-1.5B")
    parser.add_argument(
        "--data-root",
        default="data/generated/edgeops_curriculum_v2",
    )
    parser.add_argument("--output-root", default="models/edgeops_adapters_v2")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.expert == "all":
        selected = EXPERTS
    else:
        selected = {args.expert: EXPERTS[args.expert]}
    for expert, config in selected.items():
        train_path = str(Path(args.data_root) / config["train"])
        dev_path = str(Path(args.data_root) / config["dev"])
        for data_path in (train_path, dev_path):
            if not Path(data_path).exists():
                raise FileNotFoundError(
                    "%s does not exist. Run strict-curriculum dataset build first."
                    % data_path
                )
        command = [
            sys.executable,
            "train_lora_fp32_stable.py",
            "--" + args.mode,
            "--model_path",
            args.model_path,
            "--data_path",
            train_path,
            "--eval_data_path",
            dev_path,
            "--output_dir",
            str(Path(args.output_root) / expert),
            "--max_steps",
            str(args.max_steps or config["max_steps"]),
        ]
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
