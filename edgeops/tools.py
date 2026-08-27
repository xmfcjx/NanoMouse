"""Schema-validated, risk-gated tools for equipment operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from edgeops.contracts import RiskLevel, ToolCall, ToolResult
from edgeops.gateway import EquipmentGateway


@dataclass
class ToolSpec:
    name: str
    description: str
    schema: Dict[str, Any]
    risk: RiskLevel
    handler: Callable[..., Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError("Duplicate tool: %s" % spec.name)
        self._tools[spec.name] = spec

    def specs(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "schema": spec.schema,
                "risk": spec.risk.value,
            }
            for spec in self._tools.values()
        ]

    def execute(self, call: ToolCall) -> ToolResult:
        spec = self._tools.get(call.name)
        if not spec:
            return ToolResult(ok=False, tool=call.name, error="Unknown tool.")
        if spec.risk == RiskLevel.FORBIDDEN:
            return ToolResult(
                ok=False,
                tool=call.name,
                data={"_arguments": dict(call.arguments)},
                error="This safety-critical action is not available to the language model.",
            )
        if spec.risk == RiskLevel.CONFIRMATION and not call.confirmed:
            return ToolResult(
                ok=False,
                tool=call.name,
                data={"_arguments": dict(call.arguments)},
                error="Explicit operator confirmation is required.",
                requires_confirmation=True,
            )

        validation_error = self._validate(spec.schema, call.arguments)
        if validation_error:
            return ToolResult(
                ok=False,
                tool=call.name,
                data={"_arguments": dict(call.arguments)},
                error=validation_error,
            )
        try:
            data = spec.handler(**call.arguments)
            if data is None or data == []:
                return ToolResult(ok=False, tool=call.name, error="No matching record.")
            if isinstance(data, dict):
                payload = data
            else:
                payload = {"items": data}
            payload = dict(payload)
            payload["_arguments"] = dict(call.arguments)
            return ToolResult(ok=True, tool=call.name, data=payload)
        except Exception as exc:
            return ToolResult(ok=False, tool=call.name, error=str(exc))

    @staticmethod
    def _validate(schema: Dict[str, Any], arguments: Dict[str, Any]) -> str:
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in arguments or arguments[key] in (None, ""):
                return "Missing required argument: %s" % key
        for key, value in arguments.items():
            expected = properties.get(key, {}).get("type")
            if not expected:
                return "Unexpected argument: %s" % key
            if expected == "string" and not isinstance(value, str):
                return "Argument %s must be a string." % key
            if expected == "number" and not isinstance(value, (int, float)):
                return "Argument %s must be a number." % key
        return ""


def build_tool_registry(gateway: EquipmentGateway) -> ToolRegistry:
    registry = ToolRegistry()
    string_arg = lambda name: {
        "type": "object",
        "properties": {name: {"type": "string"}},
        "required": [name],
    }
    registry.register(
        ToolSpec(
            "get_device_status",
            "Read the latest status and telemetry for a device.",
            string_arg("device_id"),
            RiskLevel.READ_ONLY,
            gateway.get_device_status,
        )
    )
    registry.register(
        ToolSpec(
            "lookup_error_code",
            "Look up the meaning, severity, and safe checks for an error code.",
            string_arg("error_code"),
            RiskLevel.READ_ONLY,
            gateway.lookup_error_code,
        )
    )
    registry.register(
        ToolSpec(
            "get_maintenance_history",
            "Read maintenance records for a device.",
            string_arg("device_id"),
            RiskLevel.READ_ONLY,
            gateway.get_maintenance_history,
        )
    )
    registry.register(
        ToolSpec(
            "locate_asset",
            "Read the last known location of an asset.",
            string_arg("asset_id"),
            RiskLevel.READ_ONLY,
            gateway.locate_asset,
        )
    )
    registry.register(
        ToolSpec(
            "create_work_order_draft",
            "Create a draft work order without submitting it.",
            {
                "type": "object",
                "properties": {
                    "device_id": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["device_id", "title", "description"],
            },
            RiskLevel.CONFIRMATION,
            gateway.create_work_order_draft,
        )
    )
    registry.register(
        ToolSpec(
            "move_robot",
            "Safety-critical robot motion command.",
            {
                "type": "object",
                "properties": {
                    "device_id": {"type": "string"},
                    "destination": {"type": "string"},
                },
                "required": ["device_id", "destination"],
            },
            RiskLevel.FORBIDDEN,
            lambda **_: None,
        )
    )
    return registry
