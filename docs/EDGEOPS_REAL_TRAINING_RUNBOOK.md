# EdgeOps Training and Evaluation Runbook

This document records the reproducible v2 training flow. V2 is frozen as the
accepted release; rerunning these commands creates a new experiment rather than
changing the published result.

## Environment Boundary

- Training and generation evaluation require a CUDA host with the base model.
- The 4 GB constraint applies to the final INT4 inference configuration, not to
  SFT or DPO training.
- Model weights, adapters, generated datasets, and predictions are local
  artifacts and are not committed to Git.

Install dependencies:

```bash
python -m pip install -r requirements-ml.txt
```

## Main Entrypoints

```text
scripts/build_edgeops_eval.py
scripts/train_edgeops_adapters.py
scripts/train_edgeops_dpo.py
scripts/eval_edgeops_lora.py
scripts/eval_edgeops.py
```

`eval_edgeops_lora.py` measures free model generations.
`eval_edgeops.py` exercises the deterministic orchestration path; its metrics
must not be reported as trained-model quality.

## Build the Strict Curriculum

The source pools must already exist under `data/generated/`. They are generated
artifacts and are excluded from Git.

```bash
python scripts/build_edgeops_eval.py \
  --mode strict-curriculum \
  --output-dir data/generated/edgeops_curriculum_v2 \
  --seed 42 \
  --route-train-size 10000 \
  --route-dev-size 2000 \
  --route-test-size 2000 \
  --safety-train-size 6000 \
  --safety-dev-size 1500 \
  --safety-test-size 2000
```

Before training, inspect `build_report.json` and require zero cross-split
overlap for exact query, normalized query, template family, and source ID.

## Train the V2 SFT Adapters

```bash
python scripts/train_edgeops_adapters.py \
  --expert route_tool \
  --mode gpu11 \
  --model-path /data/models/Qwen2.5-1.5B \
  --data-root data/generated/edgeops_curriculum_v2 \
  --output-root /data/models/edgeops_adapters_v2 \
  --max-steps 600

python scripts/train_edgeops_adapters.py \
  --expert safety_refusal \
  --mode gpu11 \
  --model-path /data/models/Qwen2.5-1.5B \
  --data-root data/generated/edgeops_curriculum_v2 \
  --output-root /data/models/edgeops_adapters_v2 \
  --max-steps 400
```

The base model is frozen; only LoRA parameters are trained.

## Development Evaluation

```bash
python scripts/eval_edgeops_lora.py \
  --model-path /data/models/Qwen2.5-1.5B \
  --adapter-path /data/models/edgeops_adapters_v2/route_tool \
  --eval-file data/generated/edgeops_curriculum_v2/route_tool/eval/dev.jsonl \
  --output-dir /data/eval/results/edgeops_v2/route_tool_dev \
  --dtype fp32 \
  --batch-size 8

python scripts/eval_edgeops_lora.py \
  --model-path /data/models/Qwen2.5-1.5B \
  --adapter-path /data/models/edgeops_adapters_v2/safety_refusal \
  --eval-file data/generated/edgeops_curriculum_v2/safety_refusal/eval/dev.jsonl \
  --output-dir /data/eval/results/edgeops_v2/safety_refusal_dev \
  --dtype fp32 \
  --batch-size 8
```

Inspect dev failures for tuning. Do not inspect sealed-test examples.

## DPO Boundary

The optional safety DPO entrypoint initializes from the accepted safety SFT
adapter:

```bash
python scripts/train_edgeops_dpo.py \
  --model-path /data/models/Qwen2.5-1.5B \
  --train-file data/generated/edgeops_curriculum_v2/safety_refusal/dpo/train.jsonl \
  --eval-file data/generated/edgeops_curriculum_v2/safety_refusal/dpo/dev.jsonl \
  --init-adapter-path /data/models/edgeops_adapters_v2/safety_refusal \
  --output-dir /data/models/edgeops_adapters_v2/safety_refusal_dpo \
  --max-steps 200
```

The recorded run stopped at checkpoint 50 after preference saturation. Because
no post-DPO free-generation evaluation was completed, DPO is not part of the
accepted release.

## One-Time Sealed Acceptance

Run only after freezing code, datasets, and adapters:

```bash
ALLOW_SEALED_TEST=YES \
PROJECT_ROOT=/data/NanoChat-Lab \
ADAPTER_ROOT=/data/models/edgeops_adapters_v2 \
RESULT_ROOT=/data/eval/results/edgeops_v2_acceptance \
bash scripts/experimental/run_edgeops_v2_acceptance.sh
```

The acceptance script:

1. verifies the sealed dataset hash;
2. refuses to overwrite an existing result;
3. writes aggregate-only generation metrics;
4. benchmarks each adapter using INT4 NF4 with double quantization;
5. writes input and result SHA-256 manifests.

Opening sealed examples retires that set for future model selection. A changed
model requires a new external holdout with unseen source IDs.

## Published Outcome

The accepted metrics and the rejected v3 decision are recorded in
[`V2_TRAINING_STATUS.md`](V2_TRAINING_STATUS.md). Aggregate evidence is stored
under `eval/results/release/`.
