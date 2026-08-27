"""DPO + LoRA post-training for EdgeOps preference alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List


def load_pairs(path: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows[:limit] if limit else rows


def messages_to_text(tokenizer, messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    chunks = []
    for message in messages:
        chunks.append("<|im_start|>%s\n%s<|im_end|>" % (message["role"], message["content"]))
    return "\n".join(chunks)


def prepare_rows(tokenizer, rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    prepared = []
    for row in rows:
        prompt = messages_to_text(tokenizer, row["prompt"])
        chosen = messages_to_text(tokenizer, row["chosen"])
        rejected = messages_to_text(tokenizer, row["rejected"])
        prepared.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/Qwen2.5-1.5B")
    parser.add_argument("--train-file", default="data/generated/edgeops_curriculum_v2/safety_refusal/dpo/train.jsonl")
    parser.add_argument("--eval-file", default="data/generated/edgeops_curriculum_v2/safety_refusal/dpo/dev.jsonl")
    parser.add_argument("--output-dir", default="models/edgeops_adapters_v2/safety_refusal_dpo")
    parser.add_argument(
        "--init-adapter-path",
        default="models/edgeops_adapters_v2/safety_refusal",
        help="Safety SFT adapter used as the DPO starting point.",
    )
    parser.add_argument("--quantization", choices=["int4", "int8", "fp16"], default="int4")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=384)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    train_rows = load_pairs(Path(args.train_file), args.limit)
    eval_rows = load_pairs(Path(args.eval_file), args.limit)
    if not args.dry_run and not Path(args.init_adapter_path).exists():
        raise FileNotFoundError(
            "Safety SFT adapter does not exist: %s" % args.init_adapter_path
        )
    print(
        json.dumps(
            {
                "train_file": args.train_file,
                "eval_file": args.eval_file,
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
                "beta": args.beta,
                "init_adapter_path": args.init_adapter_path,
                "output_dir": args.output_dir,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.dry_run:
        print("dry-run ok")
        return

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, PeftModel, TaskType, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import DPOTrainer
        try:
            from trl import DPOConfig
        except ImportError:
            DPOConfig = None
            from transformers import TrainingArguments
    except ImportError as exc:
        print("Missing training dependency: %s" % exc)
        print("Install: pip install -r requirements-edgeops.txt")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token = tokenizer.eos_token
    model_kwargs = {"device_map": "auto", "trust_remote_code": True}
    if args.quantization == "int4":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif args.quantization == "int8":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.config.use_cache = False
    if args.quantization in ("int4", "int8"):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_config = None
    if args.init_adapter_path:
        model = PeftModel.from_pretrained(
            model,
            args.init_adapter_path,
            is_trainable=True,
        )
    else:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
    train_dataset = Dataset.from_list(prepare_rows(tokenizer, train_rows))
    eval_dataset = Dataset.from_list(prepare_rows(tokenizer, eval_rows))

    common_args = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "logging_steps": 5,
        "evaluation_strategy": "steps",
        "eval_steps": 50,
        "save_steps": 50,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "report_to": "none",
        "gradient_checkpointing": True,
        "remove_unused_columns": False,
    }

    if DPOConfig is not None:
        training_args = DPOConfig(
            **common_args,
            beta=args.beta,
            max_length=args.max_length,
            max_prompt_length=args.max_prompt_length,
            padding_value=tokenizer.pad_token_id,
        )
        trainer_kwargs = {
            "model": model,
            "ref_model": None,
            "args": training_args,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "tokenizer": tokenizer,
            "padding_value": tokenizer.pad_token_id,
        }
        if peft_config is not None:
            trainer_kwargs["peft_config"] = peft_config
        trainer = DPOTrainer(**trainer_kwargs)
    else:
        training_args = TrainingArguments(**common_args)
        trainer_kwargs = {
            "model": model,
            "ref_model": None,
            "args": training_args,
            "beta": args.beta,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "tokenizer": tokenizer,
            "max_length": args.max_length,
            "max_prompt_length": args.max_prompt_length,
        }
        if peft_config is not None:
            trainer_kwargs["peft_config"] = peft_config
        trainer = DPOTrainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("saved DPO adapter to %s" % args.output_dir)


if __name__ == "__main__":
    main()
