"""Parse model-generated JSON tool plans into executable calls."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Dict, List, Optional

from edgeops.contracts import ToolCall


@dataclass
class ParsedPlan:
    calls: List[ToolCall] = field(default_factory=list)
    final_answer: Optional[str] = None
    rejected_reason: Optional[str] = None


class StructuredActionParser:
    def parse(self, text: str, confirmed: bool = False) -> ParsedPlan:
        payload = self._load_payload(text)
        if isinstance(payload, list):
            steps = payload
        elif isinstance(payload, dict) and "steps" in payload:
            steps = payload["steps"]
        elif isinstance(payload, dict):
            steps = [payload]
        else:
            raise ValueError("Tool plan must be a JSON object or array.")

        plan = ParsedPlan()
        for step in steps:
            if not isinstance(step, dict):
                raise ValueError("Each tool step must be a JSON object.")
            action = step.get("action")
            if action == "final":
                plan.final_answer = str(step.get("answer", ""))
                continue
            if action == "reject":
                plan.rejected_reason = str(step.get("reason", "policy_rejection"))
                continue
            arguments = step.get("arguments", {})
            if not isinstance(action, str) or not isinstance(arguments, dict):
                raise ValueError("Each step requires string action and object arguments.")
            plan.calls.append(ToolCall(action, arguments, confirmed=confirmed))
        return plan

    @staticmethod
    def _load_payload(text: str) -> Any:
        cleaned = text.strip()
        fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
        if fence:
            cleaned = fence.group(1).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            object_match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
            if not object_match:
                raise ValueError("Model output does not contain a JSON plan.")
            return json.loads(object_match.group(1))
