#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/NanoChat-Lab}"
PYTHON="${PYTHON:-/data/venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/models/Qwen2.5-1.5B}"
ADAPTER_ROOT="${ADAPTER_ROOT:-/data/models/edgeops_adapters_v2}"
RESULT_ROOT="${RESULT_ROOT:-/data/eval/results/edgeops_v2_acceptance}"
VRAM_LIMIT_MIB="${VRAM_LIMIT_MIB:-4096}"

if [[ "${ALLOW_SEALED_TEST:-}" != "YES" ]]; then
  echo "Refusing to open sealed test. Set ALLOW_SEALED_TEST=YES explicitly." >&2
  exit 2
fi

cd "$PROJECT_ROOT"

ROUTE_TEST="data/generated/edgeops_curriculum_v2/sealed_test/route_tool.jsonl"
SAFETY_TEST="data/generated/edgeops_curriculum_v2/sealed_test/safety_refusal.jsonl"
ROUTE_ADAPTER="$ADAPTER_ROOT/route_tool"
SAFETY_ADAPTER="$ADAPTER_ROOT/safety_refusal"

for path in \
  "$MODEL_PATH/config.json" \
  "$ROUTE_TEST" \
  "$SAFETY_TEST" \
  "$ROUTE_ADAPTER/adapter_model.safetensors" \
  "$SAFETY_ADAPTER/adapter_model.safetensors"; do
  if [[ ! -f "$path" ]]; then
    echo "Required file is missing: $path" >&2
    exit 2
  fi
done

(
  cd data/generated/edgeops_curriculum_v2
  sha256sum --check SEALED_TEST.sha256
)

if [[ -e "$RESULT_ROOT" ]]; then
  echo "Acceptance output already exists; refusing to rerun sealed test: $RESULT_ROOT" >&2
  exit 3
fi

mkdir -p "$RESULT_ROOT"
{
  echo "opened_at=$(date --iso-8601=seconds)"
  echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
  sha256sum "$ROUTE_TEST" "$SAFETY_TEST"
  sha256sum \
    "$ROUTE_ADAPTER/adapter_model.safetensors" \
    "$SAFETY_ADAPTER/adapter_model.safetensors"
} > "$RESULT_ROOT/FROZEN_INPUTS.sha256"

echo "PHASE_SEALED_ROUTE"
"$PYTHON" scripts/eval_edgeops_lora.py \
  --model-path "$MODEL_PATH" \
  --adapter-path "$ROUTE_ADAPTER" \
  --eval-file "$ROUTE_TEST" \
  --output-dir "$RESULT_ROOT/sealed_route_tool" \
  --dtype fp32 \
  --batch-size 8 \
  --aggregate-only

echo "PHASE_SEALED_SAFETY"
"$PYTHON" scripts/eval_edgeops_lora.py \
  --model-path "$MODEL_PATH" \
  --adapter-path "$SAFETY_ADAPTER" \
  --eval-file "$SAFETY_TEST" \
  --output-dir "$RESULT_ROOT/sealed_safety_refusal" \
  --dtype fp32 \
  --batch-size 8 \
  --aggregate-only

for output_dir in sealed_route_tool sealed_safety_refusal; do
  if find "$RESULT_ROOT/$output_dir" -type f ! -name edgeops_lora_eval_summary.json | grep -q .; then
    echo "Sealed evaluation wrote non-aggregate output under $output_dir" >&2
    exit 4
  fi
done

echo "PHASE_INT4_ROUTE"
"$PYTHON" scripts/experimental/benchmark_edgeops_int4.py \
  --model-path "$MODEL_PATH" \
  --adapter-path "$ROUTE_ADAPTER" \
  --eval-file data/generated/edgeops_curriculum_v2/route_tool/eval/dev.jsonl \
  --output "$RESULT_ROOT/int4_route_tool.json" \
  --vram-limit-mib "$VRAM_LIMIT_MIB"

echo "PHASE_INT4_SAFETY"
"$PYTHON" scripts/experimental/benchmark_edgeops_int4.py \
  --model-path "$MODEL_PATH" \
  --adapter-path "$SAFETY_ADAPTER" \
  --eval-file data/generated/edgeops_curriculum_v2/safety_refusal/eval/dev.jsonl \
  --output "$RESULT_ROOT/int4_safety_refusal.json" \
  --vram-limit-mib "$VRAM_LIMIT_MIB"

date --iso-8601=seconds > "$RESULT_ROOT/ACCEPTANCE_DONE"
checksum_tmp="$(mktemp)"
find "$RESULT_ROOT" -type f ! -name RESULTS.sha256 -print0 \
  | sort -z \
  | xargs -0 sha256sum > "$checksum_tmp"
mv "$checksum_tmp" "$RESULT_ROOT/RESULTS.sha256"
echo "EDGEOPS_V2_ACCEPTANCE_DONE"
