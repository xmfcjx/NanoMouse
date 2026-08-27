#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/NanoChat-Lab}"
PYTHON="${PYTHON:-/data/venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/models/Qwen2.5-1.5B}"
ADAPTER_ROOT="${ADAPTER_ROOT:-/data/models/edgeops_adapters_v2}"
RESULT_ROOT="${RESULT_ROOT:-/data/eval/results/edgeops_v2}"

cd "$PROJECT_ROOT"
mkdir -p "$ADAPTER_ROOT" "$RESULT_ROOT"

echo "PHASE_ROUTE_SFT"
"$PYTHON" scripts/train_edgeops_adapters.py \
  --expert route_tool \
  --mode gpu11 \
  --model-path "$MODEL_PATH" \
  --output-root "$ADAPTER_ROOT" \
  --max-steps 600

echo "PHASE_ROUTE_DEV"
"$PYTHON" scripts/eval_edgeops_lora.py \
  --model-path "$MODEL_PATH" \
  --adapter-path "$ADAPTER_ROOT/route_tool" \
  --eval-file data/generated/edgeops_curriculum_v2/route_tool/eval/dev.jsonl \
  --output-dir "$RESULT_ROOT/route_tool_dev" \
  --dtype fp32 \
  --batch-size 8

echo "PHASE_SAFETY_SFT"
"$PYTHON" scripts/train_edgeops_adapters.py \
  --expert safety_refusal \
  --mode gpu11 \
  --model-path "$MODEL_PATH" \
  --output-root "$ADAPTER_ROOT" \
  --max-steps 400

echo "PHASE_SAFETY_DEV"
"$PYTHON" scripts/eval_edgeops_lora.py \
  --model-path "$MODEL_PATH" \
  --adapter-path "$ADAPTER_ROOT/safety_refusal" \
  --eval-file data/generated/edgeops_curriculum_v2/safety_refusal/eval/dev.jsonl \
  --output-dir "$RESULT_ROOT/safety_refusal_dev" \
  --dtype fp32 \
  --batch-size 8

echo "PHASE_DPO"
"$PYTHON" scripts/train_edgeops_dpo.py \
  --model-path "$MODEL_PATH" \
  --train-file data/generated/edgeops_curriculum_v2/safety_refusal/dpo/train.jsonl \
  --eval-file data/generated/edgeops_curriculum_v2/safety_refusal/dpo/dev.jsonl \
  --init-adapter-path "$ADAPTER_ROOT/safety_refusal" \
  --output-dir "$ADAPTER_ROOT/safety_refusal_dpo" \
  --max-steps 200

echo "PHASE_DPO_DEV"
"$PYTHON" scripts/eval_edgeops_lora.py \
  --model-path "$MODEL_PATH" \
  --adapter-path "$ADAPTER_ROOT/safety_refusal_dpo" \
  --eval-file data/generated/edgeops_curriculum_v2/safety_refusal/eval/dev.jsonl \
  --output-dir "$RESULT_ROOT/safety_refusal_dpo_dev" \
  --dtype fp32 \
  --batch-size 8

echo "EDGEOPS_V2_DEV_PIPELINE_DONE"
