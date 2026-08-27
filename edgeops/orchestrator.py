"""Unified EdgeOps request orchestration."""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from edgeops.backend import GenerationBackend
from edgeops.contracts import EdgeResponse, Evidence, Route, ToolCall, ToolResult
from edgeops.executor import StructuredToolExecutor
from edgeops.planner import ParsedPlan, StructuredActionParser
from edgeops.retrieval import ContextPacker, LexicalRetriever
from edgeops.router import ConfidenceRouter
from edgeops.tools import ToolRegistry
from edgeops.trace import TraceRecorder


class EdgeOpsOrchestrator:
    ROUTE_ADAPTERS = {
        Route.ERROR_DIAGNOSIS: "tool_adapter",
        Route.DEVICE_STATUS: "tool_adapter",
        Route.MAINTENANCE_HISTORY: "tool_adapter",
        Route.ASSET_LOCATION: "tool_adapter",
        Route.TOOL_PLAN: "tool_adapter",
        Route.MANUAL_RAG: "rag_adapter",
        Route.GENERAL: "general_adapter",
    }

    def __init__(
        self,
        router: ConfidenceRouter,
        tools: ToolRegistry,
        retriever: LexicalRetriever,
        context_packer: ContextPacker,
        backend: GenerationBackend,
        trace_path: Optional[str] = None,
    ) -> None:
        self.router = router
        self.tools = tools
        self.executor = StructuredToolExecutor(tools)
        self.action_parser = StructuredActionParser()
        self.retriever = retriever
        self.context_packer = context_packer
        self.backend = backend
        self.trace_path = trace_path

    def handle(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        confirmed: bool = False,
    ) -> EdgeResponse:
        trace = TraceRecorder(query, self.trace_path)
        decision = self.router.route(query)
        adapter = self.ROUTE_ADAPTERS.get(decision.route)
        trace.add(
            "route",
            route=decision.route.value,
            confidence=decision.confidence,
            reason=decision.reason,
            candidates=decision.candidates,
            entities=decision.entities,
            adapter=adapter,
        )

        if decision.route == Route.SAFETY_REJECT:
            answer = (
                "该请求涉及设备运动或安全关键控制，NanoChat 不会直接执行。"
                "请使用经过认证的控制系统，并由现场操作员确认。"
            )
            return self._finish(trace, answer, decision, adapter, warnings=["safety_gate"])

        if decision.route == Route.DEVICE_STATUS:
            return self._device_status(trace, decision, adapter)
        if decision.route == Route.ERROR_DIAGNOSIS:
            return self._error_diagnosis(trace, decision, adapter)
        if decision.route == Route.MAINTENANCE_HISTORY:
            return self._maintenance(trace, decision, adapter)
        if decision.route == Route.ASSET_LOCATION:
            return self._location(trace, decision, adapter)
        if decision.route == Route.TOOL_PLAN:
            return self._tool_plan(trace, decision, adapter, confirmed)
        if decision.route == Route.MANUAL_RAG:
            return self._manual_rag(trace, decision, adapter, history or [])

        if not self.backend.available:
            answer = "该请求不属于当前 EdgeOps 运维能力范围，且未配置通用模型后端。"
            return self._finish(trace, answer, decision, adapter)
        answer = self.backend.generate(query, adapter=adapter)
        trace.add("generation", adapter=adapter, output_chars=len(answer))
        return self._finish(trace, answer, decision, adapter)

    def _device_status(self, trace, decision, adapter) -> EdgeResponse:
        device_id = decision.entities.get("device_id")
        if not device_id:
            return self._finish(trace, "请提供设备编号，例如 AMR-07。", decision, adapter)
        result = self._execute(
            trace, ToolCall("get_device_status", {"device_id": device_id})
        )
        if not result.ok:
            return self._tool_failure(trace, decision, adapter, result)
        data = result.data
        answer = (
            "{device} 当前状态为 {status}，位置 {location}，电量 {battery}%，"
            "电机温度 {temperature}°C，当前故障码 {error_code}。"
        ).format(
            device=device_id,
            status=data.get("status", "unknown"),
            location=data.get("location", "unknown"),
            battery=data.get("battery_percent", "unknown"),
            temperature=data.get("motor_temperature_c", "unknown"),
            error_code=data.get("error_code") or "无",
        )
        return self._finish(trace, answer, decision, adapter, tool_results=[result])

    def _error_diagnosis(self, trace, decision, adapter) -> EdgeResponse:
        device_id = decision.entities.get("device_id")
        error_code = decision.entities.get("error_code")
        tool_results = []
        status = None
        if device_id:
            status_result = self._execute(
                trace, ToolCall("get_device_status", {"device_id": device_id})
            )
            tool_results.append(status_result)
            if status_result.ok:
                status = status_result.data
                error_code = error_code or status.get("error_code")
        if not error_code:
            return self._finish(
                trace,
                "请提供故障码，或提供包含当前故障状态的设备编号。",
                decision,
                adapter,
                tool_results=tool_results,
            )

        error_result = self._execute(
            trace, ToolCall("lookup_error_code", {"error_code": error_code})
        )
        tool_results.append(error_result)
        if not error_result.ok:
            return self._tool_failure(
                trace, decision, adapter, error_result, tool_results=tool_results
            )
        error = error_result.data
        checks = "；".join(error.get("safe_checks", [])) or "查阅设备手册"
        status_context = ""
        if status:
            status_context = "当前电量 {0}%，电机温度 {1}°C。".format(
                status.get("battery_percent", "unknown"),
                status.get("motor_temperature_c", "unknown"),
            )
        answer = (
            "{code}：{meaning}。严重级别：{severity}。{status}"
            "建议先执行只读/停机安全检查：{checks}。"
            "NanoChat 不会自动复位设备或触发运动。"
        ).format(
            code=error_code,
            meaning=error.get("meaning", "unknown"),
            severity=error.get("severity", "unknown"),
            status=status_context,
            checks=checks,
        )
        return self._finish(trace, answer, decision, adapter, tool_results=tool_results)

    def _maintenance(self, trace, decision, adapter) -> EdgeResponse:
        device_id = decision.entities.get("device_id")
        if not device_id:
            return self._finish(trace, "请提供需要查询的设备编号。", decision, adapter)
        result = self._execute(
            trace, ToolCall("get_maintenance_history", {"device_id": device_id})
        )
        if not result.ok:
            return self._tool_failure(trace, decision, adapter, result)
        rows = result.data.get("items", [])
        answer = "%s 最近的维护记录：%s" % (
            device_id,
            "；".join(
                "%s %s（%s，%s）"
                % (
                    row.get("date"),
                    row.get("action"),
                    row.get("technician"),
                    row.get("work_order", "无工单号"),
                )
                for row in rows[:3]
            ),
        )
        return self._finish(trace, answer, decision, adapter, tool_results=[result])

    def _location(self, trace, decision, adapter) -> EdgeResponse:
        asset_id = decision.entities.get("asset_id")
        if not asset_id:
            return self._finish(trace, "请提供资产编号，例如 FORKLIFT-12。", decision, adapter)
        result = self._execute(trace, ToolCall("locate_asset", {"asset_id": asset_id}))
        if not result.ok:
            return self._tool_failure(trace, decision, adapter, result)
        answer = "{asset} 最后位于 {location}，最后上报时间 {last_seen}，状态 {status}。".format(
            asset=asset_id,
            location=result.data.get("location", "unknown"),
            last_seen=result.data.get("last_seen", "unknown"),
            status=result.data.get("status", "unknown"),
        )
        return self._finish(trace, answer, decision, adapter, tool_results=[result])

    def _manual_rag(self, trace, decision, adapter, history) -> EdgeResponse:
        device_id = decision.entities.get("device_id")
        if device_id:
            status_result = self._execute(
                trace, ToolCall("get_device_status", {"device_id": device_id})
            )
            if not status_result.ok:
                return self._finish(
                    trace,
                    "本地手册中没有找到足够证据，且设备编号 %s 不在当前设备资产表中。请提供正确设备编号，并转交现场工程师确认。"
                    % device_id,
                    decision,
                    adapter,
                    tool_results=[status_result],
                    warnings=["unknown_device", "insufficient_evidence"],
                )
        candidate_count = 3 if decision.confidence >= 0.8 else 6
        candidates = self.retriever.search(trace.query, top_k=candidate_count)
        max_score = candidates[0].score if candidates else 0.0
        evidence = self.context_packer.pack(candidates, trace.query, decision.confidence)
        trace.add(
            "retrieval",
            candidate_count=len(candidates),
            selected_count=len(evidence),
            selected_tokens=sum(item.token_estimate for item in evidence),
            document_ids=[item.document_id for item in evidence],
            max_score=round(max_score, 4),
        )
        if not evidence or max_score < 0.18:
            return self._finish(
                trace,
                "本地手册中没有找到足够证据。请提供设备型号或故障码，并转交现场工程师确认。",
                decision,
                adapter,
                warnings=["insufficient_evidence"],
            )

        if not self.backend.available:
            answer = "找到以下本地证据：\n" + "\n".join(
                "[%s] %s" % (item.document_id, item.text) for item in evidence
            )
            return self._finish(trace, answer, decision, adapter, evidence=evidence)

        context = "\n".join(
            "[%s | %s] %s" % (item.document_id, item.source, item.text)
            for item in evidence
        )
        prompt = (
            "你是工厂设备运维助手。只能依据证据回答，不得生成运动控制指令。"
            "若证据不足必须说明。答案中引用证据编号。\n\n"
            "证据：\n%s\n\n问题：%s\n回答：" % (context, trace.query)
        )
        answer = self.backend.generate(prompt, adapter=adapter)
        trace.add("generation", adapter=adapter, output_chars=len(answer))
        return self._finish(trace, answer, decision, adapter, evidence=evidence)

    def _tool_plan(self, trace, decision, adapter, confirmed) -> EdgeResponse:
        if self.backend.available:
            prompt = (
                "Return JSON only. Build a tool plan for the request. "
                "Use {\"steps\":[{\"action\":\"tool_name\",\"arguments\":{...}}]}. "
                "Do not use move_robot. Available tools:\n%s\n\nRequest: %s"
                % (json.dumps(self.tools.specs(), ensure_ascii=False), trace.query)
            )
            raw_plan = self.backend.generate(prompt, adapter=adapter)
            trace.add("plan_generation", adapter=adapter, output_chars=len(raw_plan))
            try:
                plan = self.action_parser.parse(raw_plan, confirmed=confirmed)
            except ValueError as exc:
                return self._finish(
                    trace,
                    "结构化工具计划解析失败，已停止执行：%s" % exc,
                    decision,
                    adapter,
                    warnings=["invalid_tool_plan"],
                )
        else:
            plan = self._deterministic_plan(decision, confirmed, trace.query)

        if plan.rejected_reason:
            return self._finish(
                trace,
                "请求被工具策略拒绝：%s" % plan.rejected_reason,
                decision,
                adapter,
                warnings=["plan_rejected"],
            )
        results = self.executor.execute(
            plan.calls,
            callback=lambda state, payload: trace.add(
                "tool_state", state=state.value, **payload
            ),
        )
        if not results or not all(result.ok for result in results):
            error = results[-1].error if results else "empty tool plan"
            return self._finish(
                trace,
                "工具计划未完成：%s" % error,
                decision,
                adapter,
                tool_results=results,
                requires_confirmation=bool(
                    results and results[-1].requires_confirmation
                ),
            )
        answer = "结构化工具计划执行完成：\n" + "\n".join(
            "- %s: %s" % (result.tool, json.dumps(result.data, ensure_ascii=False))
            for result in results
        )
        return self._finish(
            trace,
            answer,
            decision,
            adapter,
            tool_results=results,
        )

    @staticmethod
    def _deterministic_plan(decision, confirmed, query):
        calls = []
        device_id = decision.entities.get("device_id")
        asset_id = decision.entities.get("asset_id")
        error_code = decision.entities.get("error_code")
        candidates = decision.candidates
        if device_id and candidates.get(Route.DEVICE_STATUS.value, 0) > 0:
            calls.append(
                ToolCall("get_device_status", {"device_id": device_id}, confirmed)
            )
        if error_code and candidates.get(Route.ERROR_DIAGNOSIS.value, 0) > 0:
            calls.append(
                ToolCall("lookup_error_code", {"error_code": error_code}, confirmed)
            )
        if device_id and candidates.get(Route.MAINTENANCE_HISTORY.value, 0) > 0:
            calls.append(
                ToolCall(
                    "get_maintenance_history",
                    {"device_id": device_id},
                    confirmed,
                )
            )
        if asset_id and candidates.get(Route.ASSET_LOCATION.value, 0) > 0:
            calls.append(ToolCall("locate_asset", {"asset_id": asset_id}, confirmed))
        q = query.lower()
        if (
            device_id
            and candidates.get(Route.TOOL_PLAN.value, 0) > 0
            and any(word in q for word in ["工单", "草稿", "work order", "draft"])
        ):
            status_signature = json.dumps(
                {
                    "name": "get_device_status",
                    "arguments": {"device_id": device_id},
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            existing_signatures = {
                json.dumps(
                    {"name": call.name, "arguments": call.arguments},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                for call in calls
            }
            if status_signature not in existing_signatures:
                calls.insert(
                    0,
                    ToolCall("get_device_status", {"device_id": device_id}, confirmed),
                )
            calls.append(
                ToolCall(
                    "create_work_order_draft",
                    {
                        "device_id": device_id,
                        "title": "维修工单草稿",
                        "description": "根据设备状态、故障信息和手册证据创建草稿，提交前需现场人员确认。",
                    },
                    confirmed,
                )
            )
        return ParsedPlan(calls=calls)

    def _execute(self, trace: TraceRecorder, call: ToolCall) -> ToolResult:
        results = self.executor.execute(
            [call],
            callback=lambda state, payload: trace.add(
                "tool_state", state=state.value, **payload
            ),
        )
        return results[-1]

    def _tool_failure(
        self,
        trace,
        decision,
        adapter,
        result,
        tool_results=None,
    ) -> EdgeResponse:
        results = tool_results or [result]
        return self._finish(
            trace,
            "工具调用失败：%s" % (result.error or "unknown error"),
            decision,
            adapter,
            tool_results=results,
            requires_confirmation=result.requires_confirmation,
        )

    @staticmethod
    def _finish(
        trace,
        answer,
        decision,
        adapter,
        tool_results=None,
        evidence=None,
        requires_confirmation=False,
        warnings=None,
    ) -> EdgeResponse:
        trace.finish(
            "ok",
            route=decision.route.value,
            confidence=decision.confidence,
            adapter=adapter,
            warnings=warnings or [],
        )
        return EdgeResponse(
            answer=answer,
            route=decision.route.value,
            confidence=decision.confidence,
            trace_id=trace.trace_id,
            adapter=adapter,
            tool_results=tool_results or [],
            evidence=evidence or [],
            requires_confirmation=requires_confirmation,
            warnings=warnings or [],
        )
