"""
LoRA 微调训练脚本
支持本地极限模式 (4GB VRAM) 和云端正常模式 (16GB+ VRAM)
用法:
  python train_lora.py --local --max_steps 50     # 本地极限模式
  python train_lora.py --cloud                     # 云端正常模式
  python train_lora.py --gpu11                     # 11GB 显存模式
  python train_lora.py --compare                   # 生成对比报告
"""
import os
import sys
import json
import time
import random
import argparse
import datetime
import torch
from pathlib import Path

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq,
        TrainerCallback,
    )
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel, prepare_model_for_kbit_training
    from datasets import Dataset
except ImportError as e:
    print(f"[Error] 缺少依赖: {e}")
    print("请安装: pip install peft trl datasets")
    sys.exit(1)

from config.config import get_config

RESULTS_DIR = Path("eval/results/lora")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_gpu_capability():
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        return cap[0] * 10 + cap[1]
    return 0


def get_local_config():
    return {
        "mode": "local",
        "quantization": "int4",
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_seq_length": 256,
        "learning_rate": 3e-5,
        "warmup_ratio": 0.1,
        "max_steps": 100,
        "gradient_checkpointing": True,
        "optim": "paged_adamw_8bit",
        "vram_limit_gb": 4,
    }


def get_cloud_config():
    return {
        "mode": "cloud",
        "quantization": "fp16",
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "max_seq_length": 256,
        "learning_rate": 3e-5,
        "warmup_ratio": 0.1,
        "max_steps": 200,
        "gradient_checkpointing": False,
        "optim": "adamw_torch",
        "vram_limit_gb": 16,
    }


def get_11gb_config():
    return {
        "mode": "11gb",
        # RTX 2080Ti / sm75 上，Qwen2.5-1.5B fp16 forward 已确认会产生 NaN。
        # 1.5B 的 fp32 参数约 6GB，11GB 显存通常可以承受 LoRA 训练。
        "quantization": "fp32",
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_seq_length": 256,
        "learning_rate": 1e-5,
        "warmup_ratio": 0.1,
        "max_steps": 300,
        "gradient_checkpointing": True,
        "optim": "adamw_torch",
        "vram_limit_gb": 11,
    }


def load_training_data(data_path="data/sft_train.jsonl", max_samples=None, val_ratio=0.1):
    if not os.path.exists(data_path):
        print(f"[Warning] 训练数据不存在: {data_path}")
        print("请先运行: python build_sft_data.py")
        return None, None

    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    if max_samples:
        samples = samples[:max_samples]

    random.shuffle(samples)
    
    val_count = int(len(samples) * val_ratio)
    val_samples = samples[:val_count]
    train_samples = samples[val_count:]

    print(f"[Data] 总计 {len(samples)} 条数据")
    print(f"[Data] 训练集: {len(train_samples)} 条")
    print(f"[Data] 验证集: {len(val_samples)} 条")
    return train_samples, val_samples


def format_messages(sample):
    messages = sample.get("messages", [])
    formatted = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    return formatted.strip()


def prepare_dataset(samples, tokenizer, max_seq_length):
    texts = []
    for sample in samples:
        messages = sample.get("messages", [])
        if hasattr(tokenizer, 'apply_chat_template'):
            text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
        else:
            text = format_messages(sample)
        texts.append(text)

    if texts:
        print(f"[Debug] 样本格式预览:\n{texts[0][:300]}...")

    assistant_start_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)

    def tokenize(example):
        result = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        input_ids = result["input_ids"]
        labels = [-100] * len(input_ids)
        
        i = 0
        while i <= len(input_ids) - len(assistant_start_ids):
            if input_ids[i:i+len(assistant_start_ids)] == assistant_start_ids:
                start_idx = i + len(assistant_start_ids)
                end_idx = start_idx
                while end_idx <= len(input_ids) - len(im_end_ids):
                    if input_ids[end_idx:end_idx+len(im_end_ids)] == im_end_ids:
                        break
                    end_idx += 1
                for j in range(start_idx, min(end_idx, len(input_ids))):
                    labels[j] = input_ids[j]
                i = end_idx + len(im_end_ids)
            else:
                i += 1
        
        result["labels"] = labels
        return result

    dataset = Dataset.from_dict({"text": texts})
    tokenized = dataset.map(tokenize, remove_columns=["text"])
    
    sample = tokenized[0]
    non_masked = sum(1 for l in sample["labels"] if l != -100)
    total = len(sample["input_ids"])
    print(f"[Debug] 样本 token 数: {total}, assistant tokens: {non_masked} ({non_masked/total*100:.1f}%)")
    
    return tokenized


def get_vram_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return {"allocated_gb": round(allocated, 2), "reserved_gb": round(reserved, 2)}
    return {"allocated_gb": 0, "reserved_gb": 0}


def cast_trainable_params_to_fp32(model):
    """
    Trainer(fp16=True) 会使用 GradScaler。
    如果 LoRA 可训练参数本身是 fp16，反向传播时容易触发：
      Attempting to unscale FP16 gradients.
    因此保留冻结基座为 fp16/int4，只把 requires_grad=True 的 LoRA 参数转成 fp32。
    """
    converted = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.float()
            converted += param.numel()
    print(f"  已将可训练参数转为 fp32: {converted:,} params")


def get_model_input_device(model):
    """device_map='auto' 时，优先把输入放到 embedding 所在设备。"""
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


class VRAMCallback(TrainerCallback):
    def __init__(self, log_file):
        self.log_file = log_file
        self.vram_records = []
        self.train_losses = []
        self.eval_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        vram = get_vram_usage()
        record = {
            "step": state.global_step,
            "vram_allocated_gb": vram["allocated_gb"],
            "vram_reserved_gb": vram["reserved_gb"],
            "loss": logs.get("loss", None) if logs else None,
            "eval_loss": logs.get("eval_loss", None) if logs else None,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        self.vram_records.append(record)

        if logs:
            if "loss" in logs:
                self.train_losses.append((state.global_step, logs["loss"]))
            if "eval_loss" in logs:
                self.eval_losses.append((state.global_step, logs["eval_loss"]))

        if state.global_step % 10 == 0:
            loss_str = f" | Loss: {logs.get('loss', 'N/A'):.4f}" if logs and "loss" in logs else ""
            print(f"[Step {state.global_step}] VRAM: {vram['allocated_gb']:.2f}GB / {vram['reserved_gb']:.2f}GB{loss_str}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            print(f"[Eval Step {state.global_step}] Eval Loss: {metrics['eval_loss']:.4f}")

    def save(self):
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump({
                "vram_records": self.vram_records,
                "train_losses": self.train_losses,
                "eval_losses": self.eval_losses,
            }, f, indent=2)


def train_lora(config, model_path=None, data_path="data/sft_train.jsonl", output_dir=None):
    print("\n" + "=" * 70)
    print(f"LoRA 训练 - {config['mode'].upper()} 模式")
    print("=" * 70)

    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 70)

    if model_path is None:
        model_path = get_config("paths.models", "models/Qwen2.5-1.5B")

    if output_dir is None:
        output_dir = f"models/lora_adapter/{config['mode']}"

    gpu_cap = get_gpu_capability()
    print(f"\n[GPU] 检测到 GPU 计算能力: sm{gpu_cap}")

    if gpu_cap < 80:
        print(f"[GPU] sm{gpu_cap} < sm80，自动调整配置:")
        print("  - 将使用 eager attention 实现")
        if config["quantization"] == "fp32":
            print("  - 11GB/fp32 稳定模式：不启用 fp16 AMP")
        else:
            print("  - k-bit/fp16 模式：使用 fp16 AMP 混合精度 (GradScaler)")
    else:
        print(f"[GPU] sm{gpu_cap} >= sm80，使用 bf16 混合精度")

    start_time = time.time()

    print("\n[1/5] 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("\n[2/5] 加载基座模型...")

    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }

    if gpu_cap < 80:
        model_kwargs["attn_implementation"] = "eager"

    if config["quantization"] == "int4":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quantization_config
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    elif config["quantization"] == "int8":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs["quantization_config"] = quantization_config
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    elif config["quantization"] == "fp16":
        model_kwargs["torch_dtype"] = torch.float16
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    elif config["quantization"] == "fp32":
        model_kwargs["torch_dtype"] = torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    else:
        raise ValueError(f"未知 quantization: {config['quantization']}")

    # gradient checkpointing 与 use_cache 不兼容；提前关闭，避免训练时警告/隐式改动。
    model.config.use_cache = False

    print(f"  模型加载后 VRAM: {get_vram_usage()}")

    print("\n[3/5] 配置 LoRA...")
    if config["quantization"] in ["int4", "int8"]:
        print("  检测到 k-bit 量化，执行 prepare_model_for_kbit_training...")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config["gradient_checkpointing"],
        )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 关键修复：只把 LoRA 等可训练参数转 fp32，避免 AMP unscale fp16 gradients 报错。
    cast_trainable_params_to_fp32(model)

    model.enable_input_require_grads()
    model.config.use_cache = False

    print(f"  LoRA 配置后 VRAM: {get_vram_usage()}")

    print("\n[4/5] 准备训练数据...")
    train_samples, val_samples = load_training_data(data_path)
    if train_samples is None:
        return None

    train_dataset = prepare_dataset(train_samples, tokenizer, config["max_seq_length"])
    val_dataset = prepare_dataset(val_samples, tokenizer, config["max_seq_length"]) if val_samples else None

    print("\n[5/5] 开始训练...")

    use_fp16_amp = False
    use_bf16 = False
    if config["quantization"] == "fp32":
        # fp32 稳定模式：关闭 AMP，避免 sm75 上 fp16 forward NaN。
        use_fp16_amp = False
        use_bf16 = False
    elif gpu_cap >= 80:
        use_bf16 = True
        use_fp16_amp = False
    else:
        use_bf16 = False
        use_fp16_amp = True

    training_args_kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": config["per_device_train_batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "learning_rate": config["learning_rate"],
        "gradient_checkpointing": config["gradient_checkpointing"],
        "optim": config["optim"],
        "logging_steps": 5,
        "save_steps": 50,
        "save_total_limit": 2,
        "report_to": "none",
        "remove_unused_columns": False,
        "fp16": use_fp16_amp,
        "bf16": use_bf16,
        "fp16_full_eval": False,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "max_grad_norm": 1.0,
        "logging_nan_inf_filter": False,
        "evaluation_strategy": "steps" if val_dataset else "no",
        "eval_steps": 50 if val_dataset else None,
        "load_best_model_at_end": True if val_dataset else False,
        "metric_for_best_model": "eval_loss",
    }

    if config.get("warmup_ratio"):
        training_args_kwargs["warmup_ratio"] = config["warmup_ratio"]

    if config["gradient_checkpointing"]:
        training_args_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    training_args = TrainingArguments(**training_args_kwargs)

    if "max_steps" in config:
        training_args.max_steps = config["max_steps"]
    elif "num_train_epochs" in config:
        training_args.num_train_epochs = config["num_train_epochs"]

    vram_log_file = RESULTS_DIR / f"vram_log_{config['mode']}.json"
    vram_callback = VRAMCallback(vram_log_file)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )
    
    print("\n[Debug] 测试前向传播...")
    model.eval()
    with torch.no_grad():
        sample_batch = data_collator([train_dataset[0]])
        input_device = get_model_input_device(model)
        sample_batch = {k: v.to(input_device) for k, v in sample_batch.items()}

        # fp32 稳定模式下不启用 autocast；k-bit/fp16 才启用 fp16/bf16 autocast。
        if torch.cuda.is_available() and (use_fp16_amp or use_bf16):
            autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
            with torch.cuda.amp.autocast(enabled=True, dtype=autocast_dtype):
                outputs = model(**sample_batch)
        else:
            outputs = model(**sample_batch)

        loss_finite = torch.isfinite(outputs.loss).item()
        logits_finite = torch.isfinite(outputs.logits).all().item()
        logits_max_abs = outputs.logits.detach().abs().max().item()

        print(f"  Sample loss: {outputs.loss.item():.4f}")
        print(f"  Loss finite: {loss_finite}")
        print(f"  Logits finite: {logits_finite}")
        print(f"  Logits max abs: {logits_max_abs:.4f}")
        print(f"  Logits shape: {outputs.logits.shape}")
        print(f"  Labels shape: {sample_batch['labels'].shape}")
        print(f"  Non-masked labels: {(sample_batch['labels'] != -100).sum().item()}")

        if not loss_finite or not logits_finite:
            print("[Error] 前向传播已出现 NaN/Inf，终止训练；不会保存 adapter。")
            final_vram = get_vram_usage()
            result = {
                "mode": config["mode"],
                "config": config,
                "training_success": False,
                "failure_reason": "non_finite_debug_forward",
                "final_vram_gb": final_vram,
                "gpu_capability": f"sm{gpu_cap}",
                "output_dir": output_dir,
                "timestamp": datetime.datetime.now().isoformat(),
            }
            result_file = RESULTS_DIR / f"training_result_{config['mode']}.json"
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"  结果保存: {result_file}")
            return result
    model.train()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[vram_callback],
    )

    try:
        trainer.train()
        training_success = True
    except Exception as e:
        print(f"\n[Error] 训练失败: {e}")
        training_success = False

    end_time = time.time()
    training_time = end_time - start_time

    if training_success:
        print(f"\n[保存] 保存 adapter 到 {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    final_vram = get_vram_usage()

    result = {
        "mode": config["mode"],
        "config": config,
        "training_success": training_success,
        "training_time_seconds": round(training_time, 2),
        "training_time_formatted": str(datetime.timedelta(seconds=int(training_time))),
        "final_vram_gb": final_vram,
        "gpu_capability": f"sm{gpu_cap}",
        "output_dir": output_dir,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    result_file = RESULTS_DIR / f"training_result_{config['mode']}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    vram_callback.save()

    print("\n" + "=" * 70)
    print("训练完成")
    print("=" * 70)
    print(f"  模式: {config['mode']}")
    print(f"  成功: {training_success}")
    print(f"  耗时: {result['training_time_formatted']}")
    print(f"  最终显存: {final_vram['allocated_gb']:.2f}GB / {final_vram['reserved_gb']:.2f}GB")
    print(f"  结果保存: {result_file}")

    return result


def compare_results():
    print("\n" + "=" * 70)
    print("本地 vs 云端 LoRA 训练对比")
    print("=" * 70)

    local_result_file = RESULTS_DIR / "training_result_local.json"
    cloud_result_file = RESULTS_DIR / "training_result_cloud.json"

    results = {}

    if local_result_file.exists():
        with open(local_result_file, "r", encoding="utf-8") as f:
            results["local"] = json.load(f)

    if cloud_result_file.exists():
        with open(cloud_result_file, "r", encoding="utf-8") as f:
            results["cloud"] = json.load(f)

    if not results:
        print("未找到训练结果，请先运行训练")
        return

    print(f"\n{'指标':<25} {'本地 (4GB)':<20} {'云端 (16GB)':<20}")
    print("-" * 65)

    if "local" in results and "cloud" in results:
        local = results["local"]
        cloud = results["cloud"]

        print(f"{'训练成功':<25} {local.get('training_success', 'N/A'):<20} {cloud.get('training_success', 'N/A'):<20}")
        print(f"{'训练时间':<25} {local.get('training_time_formatted', 'N/A'):<20} {cloud.get('training_time_formatted', 'N/A'):<20}")

        local_vram = local.get("final_vram_gb", {})
        cloud_vram = cloud.get("final_vram_gb", {})
        print(f"{'显存占用 (Allocated)':<25} {local_vram.get('allocated_gb', 'N/A')}GB{'':<15} {cloud_vram.get('allocated_gb', 'N/A')}GB")
        print(f"{'显存占用 (Reserved)':<25} {local_vram.get('reserved_gb', 'N/A')}GB{'':<15} {cloud_vram.get('reserved_gb', 'N/A')}GB")

        local_config = local.get("config", {})
        cloud_config = cloud.get("config", {})

        print(f"\n{'LoRA Rank':<25} {local_config.get('lora_rank', 'N/A'):<20} {cloud_config.get('lora_rank', 'N/A'):<20}")
        print(f"{'Batch Size':<25} {local_config.get('per_device_train_batch_size', 'N/A'):<20} {cloud_config.get('per_device_train_batch_size', 'N/A'):<20}")
        print(f"{'序列长度':<25} {local_config.get('max_seq_length', 'N/A'):<20} {cloud_config.get('max_seq_length', 'N/A'):<20}")
        print(f"{'量化方式':<25} {local_config.get('quantization', 'N/A'):<20} {cloud_config.get('quantization', 'N/A'):<20}")

    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)

    if "local" in results and "cloud" in results:
        local_success = results["local"].get("training_success", False)
        cloud_success = results["cloud"].get("training_success", False)

        if local_success and cloud_success:
            print("✓ 两种模式都成功完成训练")
            print("✓ 云端训练更快、配置更宽松")
            print("✓ 本地训练可行但需要极限优化")
            print("→ 推荐: 云端训练 + 本地部署")
        elif cloud_success and not local_success:
            print("✗ 本地训练失败（显存不足）")
            print("✓ 云端训练成功")
            print("→ 结论: 云端训练是必要的选择")
        elif local_success and not cloud_success:
            print("✓ 本地训练成功")
            print("✗ 云端训练失败（请检查配置）")
        else:
            print("✗ 两种模式都失败")

    compare_file = RESULTS_DIR / "compare_results.json"
    with open(compare_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n对比结果保存: {compare_file}")


def main():
    parser = argparse.ArgumentParser(description="LoRA 微调训练")
    parser.add_argument("--local", action="store_true", help="本地极限模式 (4GB VRAM)")
    parser.add_argument("--cloud", action="store_true", help="云端正常模式 (16GB+ VRAM)")
    parser.add_argument("--gpu11", action="store_true", help="11GB 显存模式 (RTX 2080Ti)")
    parser.add_argument("--compare", action="store_true", help="生成对比报告")
    parser.add_argument("--max_steps", type=int, default=None, help="最大训练步数")
    parser.add_argument("--model_path", type=str, default=None, help="模型路径")
    parser.add_argument("--data_path", type=str, default="data/sft_train.jsonl", help="训练数据路径")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")

    args = parser.parse_args()

    if args.compare:
        compare_results()
        return

    if not args.local and not args.cloud and not args.gpu11:
        print("请指定训练模式: --local / --gpu11 / --cloud")
        print("示例:")
        print("  python train_lora.py --local --max_steps 50    # 4GB 显存")
        print("  python train_lora.py --gpu11                   # 11GB 显存 (RTX 2080Ti)")
        print("  python train_lora.py --cloud                   # 16GB+ 显存")
        return

    if args.local:
        config = get_local_config()
        if args.max_steps:
            config["max_steps"] = args.max_steps
        train_lora(config, args.model_path, args.data_path, args.output_dir)

    if args.gpu11:
        config = get_11gb_config()
        if args.max_steps:
            config["max_steps"] = args.max_steps
        train_lora(config, args.model_path, args.data_path, args.output_dir)

    if args.cloud:
        config = get_cloud_config()
        train_lora(config, args.model_path, args.data_path, args.output_dir)


if __name__ == "__main__":
    main()
