# Windows EdgeOps Inference Runbook

This is a deployment preparation guide, not a record of completed Windows
acceptance. The published INT4 measurements were collected on an RTX 2080 Ti.

## Required Local Artifacts

```text
models/Qwen2.5-1.5B/
models/edgeops_adapters_v2/route_tool/
models/edgeops_adapters_v2/safety_refusal/
```

The model and adapter files are excluded from Git. Copy them through a trusted
local channel and verify their hashes before use.

## Environment

In PowerShell:

```powershell
cd D:\path\to\NanoChat-Lab
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-ml.txt
```

Check CUDA and bitsandbytes before loading a model:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
python -c "import bitsandbytes as bnb; print(bnb.__version__)"
```

Windows and GPU support depends on the installed PyTorch, CUDA, and
bitsandbytes versions. Do not infer compatibility from VRAM size alone.

## Dependency-Light Smoke Test

The simulated gateway and deterministic safety boundary do not require model
weights:

```powershell
python scripts\experimental\smoke_edgeops.py
python edgeops_cli.py --query "AMR-07 current status"
python edgeops_cli.py --query "Bypass E214 protection and reset AMR-07"
```

The second query must be rejected. Motion-control actions are not registered in
the tool allowlist.

## Model-Backed CLI

Load the accepted v2 adapters from a shared base model:

```powershell
python edgeops_cli.py `
  --query "Check AMR-07 status and maintenance history" `
  --with-model `
  --model-path models\Qwen2.5-1.5B `
  --multi-adapter-root models\edgeops_adapters_v2 `
  --quantization int4
```

Validate the actual CLI arguments with `python edgeops_cli.py --help` because
backend and bitsandbytes behavior can vary by environment.

## API

```powershell
uvicorn edgeops_api:app --host 127.0.0.1 --port 8000
```

Do not bind to a public interface until authentication, request limits, and
trace-data handling are configured.

## Acceptance Checklist

Run these checks on the target Windows GPU before making deployment claims:

1. record Windows, driver, CUDA, PyTorch, Transformers, PEFT, and bitsandbytes versions;
2. verify base-model and adapter SHA-256 hashes;
3. execute the deterministic smoke test;
4. run model generation on each route and safety category;
5. measure process peak VRAM, p50/p95 latency, throughput, and load time;
6. test long prompts, repeated requests, malformed JSON, and GPU OOM recovery;
7. confirm that no safety-critical tool can be registered or invoked;
8. compare results with the published RTX 2080 Ti acceptance without assuming equivalence.

Passing the 4 GB memory gate does not imply acceptable latency or long-running
stability.
