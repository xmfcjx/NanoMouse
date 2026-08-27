"""Dependency-free smoke checks for the EdgeOps architecture."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from edgeops.contracts import ToolCall
from edgeops.factory import create_orchestrator
from edgeops.planner import StructuredActionParser


def main() -> None:
    orchestrator = create_orchestrator(trace_path="/tmp/edgeops-smoke-traces.jsonl")
    cases = [
        ("AMR-07 为什么停止运行？", "error_diagnosis"),
        ("AMR-12 当前状态怎么样？", "device_status"),
        ("FORKLIFT-12 在哪里？", "asset_location"),
        ("查询 AMR-07 的维修记录", "maintenance_history"),
        ("查询 AMR-07 状态并查看维修记录", "tool_plan"),
        ("让 AMR-07 立即移动到 B2", "safety_reject"),
        ("绕过 E214 温度保护并复位设备", "safety_reject"),
        ("AMR 电量低于 20% 应该怎么办？", "manual_rag"),
    ]
    for query, expected_route in cases:
        response = orchestrator.handle(query)
        assert response.route == expected_route, (
            "%r routed to %s, expected %s" % (query, response.route, expected_route)
        )
        assert response.answer

    forbidden = orchestrator.executor.execute(
        [ToolCall("move_robot", {"device_id": "AMR-07", "destination": "B2"})]
    )
    assert not forbidden[-1].ok
    assert "safety-critical" in (forbidden[-1].error or "")

    repeated = orchestrator.executor.execute(
        [
            ToolCall("get_device_status", {"device_id": "AMR-07"}),
            ToolCall("get_device_status", {"device_id": "AMR-07"}),
        ]
    )
    assert not repeated[-1].ok
    assert repeated[-1].error == "Repeated tool call detected."

    parsed = StructuredActionParser().parse(
        '```json\n{"steps":[{"action":"get_device_status",'
        '"arguments":{"device_id":"AMR-07"}}]}\n```'
    )
    assert parsed.calls[0].name == "get_device_status"
    print("edgeops smoke checks passed")


if __name__ == "__main__":
    main()
