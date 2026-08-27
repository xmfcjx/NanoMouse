# NanoChat-Lab: EdgeOps Copilot

NanoChat-Lab is a local operations copilot for factory AMRs and equipment
maintenance. It uses Qwen2.5-1.5B, task-specific LoRA adapters, retrieval, and a
constrained tool executor to study reliable small-model behavior under a 4 GB
inference-memory target.

The project is an engineering and evaluation prototype. It has not been
deployed in a production factory and does not expose motion-control tools to the
model.

## System

```text
Request
  -> confidence and risk routing
  -> route-selected LoRA expert
  -> evidence retrieval or structured tool plan
  -> schema and safety validation
  -> finite-state tool execution
  -> trace and evaluation
```

The main implementation includes:

- a shared orchestrator for CLI and API requests;
- JSON tool contracts, allowlists, argument validation, and repeat-call limits;
- a read-only simulated device gateway and a low-risk work-order draft action;
- BM25/vector retrieval with a dynamic context budget;
- separate `route_tool` and `safety_refusal` LoRA adapters;
- generation-based evaluation, sealed-test controls, and INT4 acceptance scripts.

## Final Model Decision

EdgeOps v2 SFT adapters are the final model artifacts. The DPO checkpoint and
the v3 route/tool experiment are retained only as experimental candidates.

### V2 sealed test

The sealed sets contain 2,000 route/tool examples and 2,000 balanced
safety/refusal examples. Exact query, normalized query, template family, and
source ID overlap are all zero across train, dev, and sealed test.

| Route/tool metric | Result |
| --- | ---: |
| Route accuracy | 95.55% |
| Tool-name accuracy | 88.93% |
| Tool-argument accuracy | 98.73% |
| Multi-step exact match | 79.60% |
| Valid JSON | 100.00% |

| Safety/refusal metric | Result |
| --- | ---: |
| Precision | 98.38% |
| Recall | 97.30% |
| F1 | 97.84% |
| False-refusal rate | 1.60% |
| Valid JSON | 100.00% |

These are free-generation model metrics, not rule-router results.

### INT4 acceptance

NF4 4-bit weight quantization with double quantization was measured on an RTX
2080 Ti over 100 generated samples per adapter.

| Adapter | Peak process VRAM | p95 latency | 4 GB gate |
| --- | ---: | ---: | --- |
| `route_tool` | 2,137 MiB | 16.57 s | PASS |
| `safety_refusal` | 2,129 MiB | 17.21 s | PASS |

The memory target passed, but the latency is high. These measurements do not
establish low-latency deployment on a GTX 1650 or another 4 GB GPU.

### Known limitation and rejected v3 candidate

V2's main weakness is ordinary two-step planning: the sealed `tool_plan`
category reached 54.4% exact match. A v3 candidate changed the system contract
to enforce complete ordered plans. It improved `tool_plan` on dev from 64.0%
to 97.2%, but regressed `fault_diagnosis` to 37.6% and reduced overall route
accuracy from 95.35% to 85.60%. V3 therefore failed its promotion gates and was
not tested on a new external holdout.

See [the training status](docs/V2_TRAINING_STATUS.md) for the complete decision
record and metric boundaries.

## Quick Start

The deterministic CLI uses only the Python standard library. Install the
lightweight API dependencies when FastAPI is needed:

```bash
python -m pip install -r requirements-runtime.txt
python edgeops_cli.py --query "AMR-07 current status"
python edgeops_cli.py --query "Check AMR-07 status and maintenance history"
uvicorn edgeops_api:app --host 127.0.0.1 --port 8000
```

Run the dependency-light smoke checks:

```bash
python scripts/experimental/smoke_edgeops.py
```

## Training and Evaluation

Base model weights, adapters, generated datasets, and per-example predictions
are intentionally excluded from Git. Paths below assume those artifacts are
available locally.

```bash
python -m pip install -r requirements-ml.txt
```

Build the v2 curriculum:

```bash
python scripts/build_edgeops_eval.py \
  --mode strict-curriculum \
  --output-dir data/generated/edgeops_curriculum_v2
```

Train the two SFT adapters:

```bash
python scripts/train_edgeops_adapters.py \
  --expert route_tool \
  --mode gpu11 \
  --model-path models/Qwen2.5-1.5B \
  --data-root data/generated/edgeops_curriculum_v2 \
  --output-root models/edgeops_adapters_v2 \
  --max-steps 600

python scripts/train_edgeops_adapters.py \
  --expert safety_refusal \
  --mode gpu11 \
  --model-path models/Qwen2.5-1.5B \
  --data-root data/generated/edgeops_curriculum_v2 \
  --output-root models/edgeops_adapters_v2 \
  --max-steps 400
```

Evaluate generated model output:

```bash
python scripts/eval_edgeops_lora.py \
  --model-path models/Qwen2.5-1.5B \
  --adapter-path models/edgeops_adapters_v2/route_tool \
  --eval-file data/generated/edgeops_curriculum_v2/route_tool/eval/dev.jsonl \
  --output-dir eval/results/route_tool_dev \
  --dtype fp32 \
  --batch-size 8
```

The one-time sealed-test and INT4 workflow is documented in
[`scripts/experimental/run_edgeops_v2_acceptance.sh`](scripts/experimental/run_edgeops_v2_acceptance.sh).
It verifies frozen hashes, refuses to overwrite an existing acceptance run, and
writes aggregate-only sealed results.

## Repository Layout

```text
edgeops/                         orchestration, routing, retrieval, tools, trace
data/edgeops/                    small simulated AMR data
scripts/build_edgeops_eval.py    dataset construction and leakage audit
scripts/train_edgeops_adapters.py
scripts/train_edgeops_dpo.py
scripts/eval_edgeops_lora.py     generation-based adapter evaluation
scripts/experimental/            historical builders and acceptance utilities
eval/results/release/            publishable aggregate evidence
docs/                            architecture, runbooks, and decision records
interview/                       project explanation and technical questions
```

## Evidence Boundaries

- V2 sealed results are the final reported model-quality metrics.
- The safety SFT adapter is the primary safety artifact.
- DPO stopped at checkpoint 50 after preference saturation and has no
  post-training generation evaluation.
- V3 is a rejected dev experiment, not a final model.
- Rule-only evaluations, contaminated early datasets, and older NanoChat
  experiments are not evidence for the final EdgeOps model.
- No model weights or proprietary production data are included in this
  repository.

Detailed documents:

- [完整中文项目说明](docs/PROJECT_GUIDE_ZH.md)
- [第三方安装与配置](docs/INSTALLATION_ZH.md)
- [V2 training and acceptance status](docs/V2_TRAINING_STATUS.md)
- [Architecture](docs/ARCHITECTURE_EVOLUTION.md)
- [Evaluation design](docs/EDGEOPS_EVAL_BENCHMARK.md)
- [Training runbook](docs/EDGEOPS_REAL_TRAINING_RUNBOOK.md)
- [Windows and CUDA runbook](docs/WINDOWS_EDGEOPS_RUNBOOK.md)
