"""Composition root for the EdgeOps architecture."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from edgeops.backend import GenerationBackend, NullBackend
from edgeops.gateway import EquipmentGateway
from edgeops.orchestrator import EdgeOpsOrchestrator
from edgeops.retrieval import ContextPacker, LexicalRetriever
from edgeops.router import ConfidenceRouter
from edgeops.tools import build_tool_registry


def create_orchestrator(
    data_dir: str = "data/edgeops",
    trace_path: str = "eval/results/edgeops/traces.jsonl",
    backend: Optional[GenerationBackend] = None,
) -> EdgeOpsOrchestrator:
    root = Path(data_dir)
    gateway = EquipmentGateway(str(root))
    tools = build_tool_registry(gateway)
    retriever = LexicalRetriever.from_jsonl(str(root / "manuals.jsonl"))
    return EdgeOpsOrchestrator(
        router=ConfidenceRouter(),
        tools=tools,
        retriever=retriever,
        context_packer=ContextPacker(token_budget=384),
        backend=backend or NullBackend(),
        trace_path=trace_path,
    )
