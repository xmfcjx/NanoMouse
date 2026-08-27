# Release Evidence

This directory contains aggregate results only. It intentionally excludes
sealed examples, generated prompts, per-example predictions, base weights, and
LoRA adapter weights.

## Accepted V2

`v2_acceptance/` contains:

- route/tool and safety/refusal sealed-test summaries;
- INT4 NF4 resource benchmarks for both adapters;
- frozen-input and result SHA-256 manifests;
- the acceptance completion timestamp.

The absolute `/data/...` paths in the manifests record the original CUDA
evaluation environment. The corresponding model and sealed dataset artifacts
are not published.

## Rejected V3

`v3_rejected/` contains the aggregate 2,000-example dev summary for the v3
route/tool candidate. It is retained as failure-analysis evidence. V3 did not
pass promotion gates and is not a release artifact.
