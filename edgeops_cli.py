"""CLI entry point for the new EdgeOps orchestrator."""

from __future__ import annotations

import argparse
import json
import os

from edgeops.backend import LegacyLLMBackend, PeftMultiAdapterBackend
from edgeops.factory import create_orchestrator


def build_backend(args):
    if not args.with_model:
        return None
    if args.multi_adapter_root:
        return PeftMultiAdapterBackend(
            model_path=args.model_path,
            adapter_root=args.multi_adapter_root,
            quantization=args.quantization,
        )
    from core.llm import LLM

    adapter_path = args.adapter_path
    if adapter_path and adapter_path.lower() == "none":
        adapter_path = None
    llm = LLM(
        model_path=args.model_path,
        quantization=args.quantization,
        adapter_path=adapter_path,
    )
    return LegacyLLMBackend(llm)


def main() -> None:
    parser = argparse.ArgumentParser(description="NanoChat EdgeOps Copilot")
    parser.add_argument("--query", help="Run a single request")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--data-dir", default="data/edgeops")
    parser.add_argument("--trace-path", default="eval/results/edgeops/traces.jsonl")
    parser.add_argument("--with-model", action="store_true")
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", "models/Qwen2.5-1.5B"))
    parser.add_argument("--adapter-path", default=os.environ.get("LORA_PATH", "models/lora_adapter"))
    parser.add_argument(
        "--multi-adapter-root",
        default=os.environ.get("EDGEOPS_ADAPTER_ROOT"),
        help="Directory containing tool/rag/memory adapter subdirectories",
    )
    parser.add_argument("--quantization", default=os.environ.get("QUANTIZATION", "int4"))
    parser.add_argument("--compact", action="store_true", help="Print answer only")
    args = parser.parse_args()

    if not args.query and not args.interactive:
        parser.error("Use --query or --interactive.")

    orchestrator = create_orchestrator(
        data_dir=args.data_dir,
        trace_path=args.trace_path,
        backend=build_backend(args),
    )

    def run(query: str) -> None:
        response = orchestrator.handle(query)
        if args.compact:
            print(response.answer)
        else:
            print(json.dumps(response.to_dict(), ensure_ascii=False, indent=2))

    if args.query:
        run(args.query)
    if args.interactive:
        print("NanoChat EdgeOps Copilot. 输入 /exit 退出。")
        while True:
            query = input("\nEdgeOps> ").strip()
            if query.lower() in ("/exit", "exit", "quit"):
                break
            if query:
                run(query)


if __name__ == "__main__":
    main()
