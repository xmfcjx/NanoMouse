"""Structured execution traces for evaluation and failure analysis."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import uuid


class TraceRecorder:
    def __init__(self, query: str, sink_path: Optional[str] = None) -> None:
        self.trace_id = uuid.uuid4().hex
        self.query = query
        self.sink_path = Path(sink_path) if sink_path else None
        self.started_at = time.time()
        self.events: List[Dict[str, Any]] = []

    def add(self, event: str, **payload: Any) -> None:
        self.events.append(
            {
                "event": event,
                "elapsed_ms": round((time.time() - self.started_at) * 1000, 3),
                **payload,
            }
        )

    def finish(self, outcome: str, **payload: Any) -> Dict[str, Any]:
        record = {
            "trace_id": self.trace_id,
            "query": self.query,
            "outcome": outcome,
            "total_latency_ms": round((time.time() - self.started_at) * 1000, 3),
            "events": self.events,
            **payload,
        }
        if self.sink_path:
            self.sink_path.parent.mkdir(parents=True, exist_ok=True)
            with self.sink_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record
