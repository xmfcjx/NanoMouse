"""Measure INT4 LoRA inference memory and latency without touching sealed test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import threading
import time
from typing import Any, Dict, List


SAFETY_SYSTEM_PROMPT = (
    "You are an EdgeOps assistant. Return valid JSON only. Use tools for "
    "device facts, use evidence for manuals, and reject unsafe motion or "
    "protection-bypass requests."
)

ROUTE_TOOL_SYSTEM_PROMPT = (
    "You are an EdgeOps routing and tool-planning assistant. Return exactly one "
    "valid JSON object. Allowed actions are get_device_status(device_id), "
    "lookup_error_code(error_code), get_maintenance_history(device_id), "
    "locate_asset(asset_id), and create_work_order_draft(device_id). For one "
    "requested operation return {\"action\": \"...\", \"arguments\": {...}}. "
    "For multiple requested operations return {\"steps\": [...]} and include "
    "every requested operation exactly once in the requested order. Never "
    "invent actions. For manual evidence or missing required identifiers, "
    "return a route of manual_rag or clarification."
)


def load_queries(path: Path, count: int) -> List[str]:
    queries = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                queries.append(json.loads(line)["query"])
                if len(queries) >= count:
                    break
    if len(queries) < count:
        raise ValueError(f"{path} contains only {len(queries)} usable queries")
    return queries


def percentile(values: List[float], pct: int) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
    return round(ordered[index], 3)


def gpu_used_mib() -> int | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return int(output.splitlines()[0].strip())
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


class MemoryMonitor:
    def __init__(self, interval_seconds: float = 0.05) -> None:
        self.interval_seconds = interval_seconds
        self.samples: List[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            value = gpu_used_mib()
            if value is not None:
                self.samples.append(value)
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument(
        "--eval-file",
        default="data/generated/edgeops_curriculum_v2/route_tool/eval/dev.jsonl",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--vram-limit-mib", type=int, default=4096)
    args = parser.parse_args()

    if "sealed_test" in Path(args.eval_file).parts:
        parser.error("INT4 acceptance must use dev data, not sealed test")
    if args.samples <= 0 or args.warmup < 0:
        parser.error("--samples must be positive and --warmup must be non-negative")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for INT4 acceptance")

    queries = load_queries(Path(args.eval_file), args.samples + args.warmup)
    system_prompt = (
        SAFETY_SYSTEM_PROMPT
        if "safety_refusal" in Path(args.eval_file).parts
        else ROUTE_TOOL_SYSTEM_PROMPT
    )
    baseline_gpu_mib = gpu_used_mib()
    monitor = MemoryMonitor()
    monitor.start()
    started = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        quantization_config=quantization_config,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    loaded_gpu_mib = gpu_used_mib()

    def generate(query: str) -> int:
        text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_length = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        return int(output.shape[1] - input_length)

    for query in queries[: args.warmup]:
        generate(query)

    torch.cuda.reset_peak_memory_stats()
    latencies_ms: List[float] = []
    generated_tokens = 0
    for query in queries[args.warmup :]:
        start = time.perf_counter()
        generated_tokens += generate(query)
        latencies_ms.append((time.perf_counter() - start) * 1000)

    torch_peak_allocated_mib = torch.cuda.max_memory_allocated() / (1024**2)
    torch_peak_reserved_mib = torch.cuda.max_memory_reserved() / (1024**2)
    monitor.stop()
    nvidia_smi_peak_mib = max(monitor.samples) if monitor.samples else None
    measured_peak_mib = max(
        value
        for value in (
            nvidia_smi_peak_mib,
            int(round(torch_peak_reserved_mib)),
        )
        if value is not None
    )
    total_seconds = sum(latencies_ms) / 1000

    report: Dict[str, Any] = {
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "quantization": "int4_nf4_double_quant",
        "compute_dtype": "float16",
        "gpu": torch.cuda.get_device_name(0),
        "samples": args.samples,
        "warmup_samples": args.warmup,
        "max_new_tokens": args.max_new_tokens,
        "model_load_seconds": round(load_seconds, 3),
        "baseline_gpu_used_mib": baseline_gpu_mib,
        "loaded_gpu_used_mib": loaded_gpu_mib,
        "nvidia_smi_peak_used_mib": nvidia_smi_peak_mib,
        "torch_peak_allocated_mib": round(torch_peak_allocated_mib, 3),
        "torch_peak_reserved_mib": round(torch_peak_reserved_mib, 3),
        "measured_peak_mib": measured_peak_mib,
        "vram_limit_mib": args.vram_limit_mib,
        "vram_pass": measured_peak_mib <= args.vram_limit_mib,
        "mean_latency_ms": round(statistics.fmean(latencies_ms), 3),
        "p50_latency_ms": percentile(latencies_ms, 50),
        "p95_latency_ms": percentile(latencies_ms, 95),
        "generated_tokens": generated_tokens,
        "tokens_per_second": round(generated_tokens / total_seconds, 3),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["vram_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
