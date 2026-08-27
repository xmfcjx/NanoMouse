"""FastAPI entry point backed by the unified EdgeOps orchestrator."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from edgeops.backend import LegacyLLMBackend, PeftMultiAdapterBackend
from edgeops.factory import create_orchestrator


def _build_backend():
    if os.environ.get("EDGEOPS_WITH_MODEL", "0") != "1":
        return None
    adapter_root = os.environ.get("EDGEOPS_ADAPTER_ROOT")
    if adapter_root:
        return PeftMultiAdapterBackend(
            model_path=os.environ.get("MODEL_PATH", "models/Qwen2.5-1.5B"),
            adapter_root=adapter_root,
            quantization=os.environ.get("QUANTIZATION", "int4"),
        )
    from core.llm import LLM

    adapter_path = os.environ.get("LORA_PATH", "models/lora_adapter")
    if adapter_path.lower() in ("", "none"):
        adapter_path = None
    return LegacyLLMBackend(
        LLM(
            model_path=os.environ.get("MODEL_PATH", "models/Qwen2.5-1.5B"),
            quantization=os.environ.get("QUANTIZATION", "int4"),
            adapter_path=adapter_path,
        )
    )


orchestrator = create_orchestrator(
    data_dir=os.environ.get("EDGEOPS_DATA_DIR", "data/edgeops"),
    trace_path=os.environ.get(
        "EDGEOPS_TRACE_PATH", "eval/results/edgeops/traces.jsonl"
    ),
    backend=_build_backend(),
)

app = FastAPI(
    title="NanoChat EdgeOps Copilot",
    version="0.1.0",
    description="Resource-aware local operations copilot prototype.",
)


class EdgeOpsRequest(BaseModel):
    query: str = Field(..., min_length=1)
    history: List[Dict[str, str]] = Field(default_factory=list)
    confirmed: bool = False


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "architecture": "edgeops",
        "model_enabled": orchestrator.backend.available,
        "tools": orchestrator.tools.specs(),
    }


@app.post("/v1/edgeops/chat")
def chat(request: EdgeOpsRequest) -> Dict[str, Any]:
    return orchestrator.handle(
        request.query,
        history=request.history,
        confirmed=request.confirmed,
    ).to_dict()
