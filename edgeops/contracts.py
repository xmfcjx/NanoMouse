"""Shared contracts for routing, tools, retrieval, and responses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Route(str, Enum):
    SAFETY_REJECT = "safety_reject"
    DEVICE_STATUS = "device_status"
    ERROR_DIAGNOSIS = "error_diagnosis"
    MAINTENANCE_HISTORY = "maintenance_history"
    ASSET_LOCATION = "asset_location"
    TOOL_PLAN = "tool_plan"
    MANUAL_RAG = "manual_rag"
    GENERAL = "general"


class RiskLevel(str, Enum):
    READ_ONLY = "read_only"
    CONFIRMATION = "confirmation"
    FORBIDDEN = "forbidden"


@dataclass
class RouteDecision:
    route: Route
    confidence: float
    reason: str
    candidates: Dict[str, float] = field(default_factory=dict)
    entities: Dict[str, str] = field(default_factory=dict)
    estimated_latency_ms: float = 0.0
    estimated_memory_mb: float = 0.0


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    confirmed: bool = False


@dataclass
class ToolResult:
    ok: bool
    tool: str
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    requires_confirmation: bool = False


@dataclass
class Evidence:
    document_id: str
    text: str
    source: str
    score: float
    token_estimate: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeResponse:
    answer: str
    route: str
    confidence: float
    trace_id: str
    adapter: Optional[str] = None
    tool_results: List[ToolResult] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    requires_confirmation: bool = False
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
