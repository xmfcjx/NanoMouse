"""Finite-state execution for structured tool plans."""

from __future__ import annotations

from enum import Enum
import json
from typing import Callable, List, Optional, Set

from edgeops.contracts import ToolCall, ToolResult
from edgeops.tools import ToolRegistry


class ExecutionState(str, Enum):
    PLAN = "plan"
    TOOL_CALL = "tool_call"
    OBSERVATION = "observation"
    FINAL = "final"
    REJECTED = "rejected"


StateCallback = Callable[[ExecutionState, dict], None]


class StructuredToolExecutor:
    def __init__(self, registry: ToolRegistry, max_steps: int = 4) -> None:
        self.registry = registry
        self.max_steps = max_steps

    def execute(
        self,
        calls: List[ToolCall],
        callback: Optional[StateCallback] = None,
    ) -> List[ToolResult]:
        self._emit(callback, ExecutionState.PLAN, {"steps": len(calls)})
        if len(calls) > self.max_steps:
            result = ToolResult(
                ok=False,
                tool="plan",
                error="Tool plan exceeds maximum step count.",
            )
            self._emit(callback, ExecutionState.REJECTED, {"reason": result.error})
            return [result]

        seen: Set[str] = set()
        results: List[ToolResult] = []
        for index, call in enumerate(calls):
            signature = json.dumps(
                {"name": call.name, "arguments": call.arguments},
                ensure_ascii=False,
                sort_keys=True,
            )
            if signature in seen:
                result = ToolResult(
                    ok=False,
                    tool=call.name,
                    error="Repeated tool call detected.",
                )
                results.append(result)
                self._emit(
                    callback,
                    ExecutionState.REJECTED,
                    {"step": index, "reason": result.error},
                )
                return results
            seen.add(signature)

            self._emit(
                callback,
                ExecutionState.TOOL_CALL,
                {"step": index, "name": call.name, "arguments": call.arguments},
            )
            result = self.registry.execute(call)
            results.append(result)
            self._emit(
                callback,
                ExecutionState.OBSERVATION,
                {
                    "step": index,
                    "name": call.name,
                    "ok": result.ok,
                    "error": result.error,
                },
            )
            if not result.ok:
                self._emit(
                    callback,
                    ExecutionState.REJECTED,
                    {"step": index, "reason": result.error},
                )
                return results

        self._emit(callback, ExecutionState.FINAL, {"completed_steps": len(results)})
        return results

    @staticmethod
    def _emit(
        callback: Optional[StateCallback],
        state: ExecutionState,
        payload: dict,
    ) -> None:
        if callback:
            callback(state, payload)
