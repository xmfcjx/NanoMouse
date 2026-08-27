# EdgeOps Model and Acceptance Status

Updated: 2026-08-26

## Final Decision

EdgeOps v2 is the final accepted model version:

- `route_tool`: v2 SFT adapter;
- `safety_refusal`: v2 SFT adapter;
- DPO checkpoint 50: experimental only;
- v3 `route_tool`: rejected tuning candidate.

The remote GPU instance is powered off. No further training is required for the
current release.

## Data Gate

The v2 curriculum contains:

| Expert | Train | Dev | Sealed test |
| --- | ---: | ---: | ---: |
| `route_tool` | 10,000 | 2,000 | 2,000 |
| `safety_refusal` | 6,000 | 1,500 | 2,000 |

Safety splits are balanced between refusal-positive and normal requests. The
builder reported zero cross-split overlap for:

- exact query;
- normalized query;
- semantic template family;
- source ID.

## V2 Generation Results

### Development sets

| Route/tool metric | Value |
| --- | ---: |
| Route accuracy | 0.9535 |
| Tool-name accuracy | 0.8980 |
| Tool-argument accuracy | 0.9787 |
| Multi-step exact match | 0.8240 |
| Valid JSON | 1.0000 |

| Safety/refusal metric | Value |
| --- | ---: |
| Precision | 0.9840 |
| Recall | 0.9813 |
| F1 | 0.9826 |
| False-refusal rate | 0.0160 |
| Valid JSON | 1.0000 |

### Sealed test

| Route/tool metric | Value |
| --- | ---: |
| Route accuracy | 0.9555 |
| Tool-name accuracy | 0.8893 |
| Tool-argument accuracy | 0.9873 |
| Multi-step exact match | 0.7960 |
| Valid JSON | 1.0000 |

| Safety/refusal metric | Value |
| --- | ---: |
| Precision | 0.9838 |
| Recall | 0.9730 |
| F1 | 0.9784 |
| False-refusal rate | 0.0160 |
| Valid JSON | 1.0000 |

These are free-generation adapter metrics. They are not deterministic
rule-router metrics.

The sealed route/tool category breakdown exposes the main limitation:

| Category | Route accuracy | Multi-step exact match |
| --- | ---: | ---: |
| `fault_diagnosis` | 0.892 | 0.892 |
| `tool_plan` | 0.848 | 0.544 |
| `work_order_draft` | 0.968 | 0.952 |

## INT4 Acceptance

NF4 4-bit weight quantization with double quantization was measured on an RTX
2080 Ti. Each adapter benchmark used 100 samples after five warm-up samples.

| Adapter | Peak process VRAM | p95 latency | Throughput | 4 GB gate |
| --- | ---: | ---: | ---: | --- |
| `route_tool` | 2,137 MiB | 16,565.6 ms | 6.564 token/s | PASS |
| `safety_refusal` | 2,129 MiB | 17,214.0 ms | 6.381 token/s | PASS |

The result demonstrates memory feasibility under the configured workload, not
low-latency deployment. It must not be presented as a GTX 1650 performance
measurement.

## DPO Decision

Safety SFT remains the primary safety artifact. DPO stopped at checkpoint 50
after preference accuracy reached 1.0 and eval loss reached 0.000179. No
post-DPO free-generation evaluation was completed, so the checkpoint is not an
accepted result.

## V3 Tuning Outcome

V3 attempted to improve the weak two-step `tool_plan` category. It kept the v2
route/tool train and dev rows unchanged and replaced only the system contract:

- enumerate the five allowed actions;
- require each requested operation exactly once and in order;
- prohibit invented actions;
- define single-action and multi-step JSON shapes.

Training completed for 600 steps. Final train loss was 0.12444 and eval loss
was 0.007599.

### Direct dev comparison

Both versions were evaluated on the same 2,000 route/tool dev rows.

| Metric | V2 | V3 | Change |
| --- | ---: | ---: | ---: |
| Route accuracy | 0.9535 | 0.8560 | -0.0975 |
| Tool-name accuracy | 0.8980 | 0.8120 | -0.0860 |
| Tool-argument accuracy | 0.9787 | 0.8253 | -0.1534 |
| Multi-step exact match | 0.8240 | 0.7667 | -0.0573 |
| Valid JSON | 1.0000 | 0.9995 | -0.0005 |
| `tool_plan` exact match | 0.640 | 0.972 | +0.332 |

V3 category results:

| Category | Route accuracy | Multi-step exact match |
| --- | ---: | ---: |
| `asset_location` | 0.820 | n/a |
| `clarification` | 1.000 | n/a |
| `device_status` | 0.864 | n/a |
| `fault_diagnosis` | 0.376 | 0.376 |
| `maintenance_history` | 0.848 | n/a |
| `manual_rag` | 0.996 | n/a |
| `tool_plan` | 0.972 | 0.972 |
| `work_order_draft` | 0.972 | 0.952 |

The candidate failed every aggregate promotion gate except the local
`tool_plan` improvement. It was not promoted and was not evaluated on a new
external holdout.

### Failure analysis

- The contract referred to operations explicitly requested by the user, while
  `fault_diagnosis` labels implicitly decompose one diagnosis request into
  status and error-code calls.
- V3 trained a fresh adapter from the base model instead of continuing from the
  accepted v2 adapter.
- Prompt-ablation runs showed strong distribution dependence between each
  adapter and its training contract.
- Token-level eval loss did not predict strict plan-sequence correctness.
- The longer contract increased truncation risk under the 256-token training
  limit; this remains a hypothesis because a full token-length audit was not
  completed.

Step 550 and step 600 produced identical metrics on the same stratified
400-example subset, so additional late-stage training was not justified.

## Evidence and Boundaries

Release evidence is stored under `eval/results/release/`. The archived adapter
bundle remains outside Git:

```text
../training_results/edgeops_v2_results_20260820_1254
```

Verified archive SHA-256:

```text
57586fd922d89254627d3f2168e9a7044c95ecfd15fc02e6bd900e516ea58d29
```

Do not report:

- contaminated or rule-only early evaluations as model generalization;
- v3 as the final model;
- DPO as an accepted safety improvement;
- the 4 GB memory gate as proof of low latency or GTX 1650 validation.
