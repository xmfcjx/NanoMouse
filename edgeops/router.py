"""Confidence- and cost-aware routing for EdgeOps requests."""

from __future__ import annotations

import re
from typing import Callable, Dict, Optional

from edgeops.contracts import Route, RouteDecision


LearnedScorer = Callable[[str], Dict[str, float]]


class ConfidenceRouter:
    ROUTE_COSTS = {
        Route.SAFETY_REJECT: (1.0, 0.0),
        Route.DEVICE_STATUS: (8.0, 5.0),
        Route.ERROR_DIAGNOSIS: (12.0, 5.0),
        Route.MAINTENANCE_HISTORY: (10.0, 5.0),
        Route.ASSET_LOCATION: (8.0, 5.0),
        Route.TOOL_PLAN: (900.0, 900.0),
        Route.MANUAL_RAG: (1200.0, 900.0),
        Route.GENERAL: (1800.0, 1400.0),
    }

    def __init__(
        self,
        learned_scorer: Optional[LearnedScorer] = None,
        confidence_threshold: float = 0.58,
        latency_weight: float = 0.00003,
        memory_weight: float = 0.00001,
        known_error_codes: Optional[set[str]] = None,
    ) -> None:
        self.learned_scorer = learned_scorer
        self.confidence_threshold = confidence_threshold
        self.latency_weight = latency_weight
        self.memory_weight = memory_weight
        self.known_error_codes = known_error_codes or {"E105", "E214"}

    def route(self, query: str) -> RouteDecision:
        rule_scores = self._rule_scores(query)
        learned_scores = self.learned_scorer(query) if self.learned_scorer else {}
        candidates: Dict[Route, float] = {}
        for route in Route:
            rule_score = rule_scores.get(route, 0.0)
            learned_score = learned_scores.get(route.value, 0.0)
            confidence = 0.7 * rule_score + 0.3 * learned_score if learned_scores else rule_score
            candidates[route] = confidence

        if max(candidates.values()) <= 0:
            candidates[Route.MANUAL_RAG] = 0.60

        scored = {}
        for route, confidence in candidates.items():
            latency, memory = self.ROUTE_COSTS[route]
            scored[route] = (
                confidence
                - self.latency_weight * latency
                - self.memory_weight * memory
            )
        selected = max(scored, key=scored.get)
        confidence = candidates[selected]

        if (
            selected not in (Route.SAFETY_REJECT, Route.MANUAL_RAG)
            and confidence < self.confidence_threshold
        ):
            selected = Route.MANUAL_RAG
            confidence = max(candidates[Route.MANUAL_RAG], 0.58)

        latency, memory = self.ROUTE_COSTS[selected]
        entities = self._extract_entities(query)
        return RouteDecision(
            route=selected,
            confidence=round(confidence, 4),
            reason=self._reason(selected, query),
            candidates={route.value: round(value, 4) for route, value in candidates.items()},
            entities=entities,
            estimated_latency_ms=latency,
            estimated_memory_mb=memory,
        )

    def _rule_scores(self, query: str) -> Dict[Route, float]:
        q = query.lower().strip()
        scores: Dict[Route, float] = {}
        entities = self._extract_entities(query)
        has_explicit_id = bool(entities.get("device_id") or entities.get("asset_id"))
        error_code = entities.get("error_code")
        known_error_code = bool(error_code and error_code in self.known_error_codes)

        has_equipment = bool(
            re.search(r"\b(?:amr|robot|vehicle|forklift)-?\d+\b", q)
            or any(word in q for word in ["机器人", "车辆", "叉车", "设备"])
        )
        control_actions = [
            "移动", "前进", "后退", "转向", "急停", "启动", "复位",
            "直接让", "开到", "继续跑", "马上启动",
            "move", "drive", "start", "release brake", "reset", "resume navigation",
        ]
        protection_bypass = [
            "绕过", "解除保护", "关闭安全", "屏蔽", "告警屏蔽", "不用人工确认",
            "disable safety", "bypass protection", "ignore alarm", "force",
        ]
        if (
            has_equipment and any(action in q for action in control_actions)
        ) or any(action in q for action in protection_bypass):
            scores[Route.SAFETY_REJECT] = 0.99
            return scores

        asks_status = any(word in q for word in ["状态", "status", "电量", "battery", "温度", "temperature", "telemetry", "异常", "healthy"])
        asks_alarm_presence = any(word in q for word in ["有没有报警", "是否报警", "has alarm", "any alarm"])
        if asks_status or asks_alarm_presence:
            scores[Route.DEVICE_STATUS] = 0.88
        if any(word in q for word in ["错误码", "故障码", "error code", "报警", "alarm", "为什么停止"]):
            scores[Route.ERROR_DIAGNOSIS] = 0.94
        if asks_alarm_presence and has_explicit_id:
            scores[Route.DEVICE_STATUS] = 0.95
            scores[Route.ERROR_DIAGNOSIS] = min(scores.get(Route.ERROR_DIAGNOSIS, 0), 0.75)
        if error_code and known_error_code:
            scores[Route.ERROR_DIAGNOSIS] = max(scores.get(Route.ERROR_DIAGNOSIS, 0), 0.98)
        elif error_code and not known_error_code:
            scores[Route.MANUAL_RAG] = 0.96
            scores[Route.ERROR_DIAGNOSIS] = min(scores.get(Route.ERROR_DIAGNOSIS, 0), 0.55)
        if any(word in q for word in ["维修记录", "维护记录", "维护历史", "上次维修", "最近维修", "保养", "service records", "maintenance records", "maintenance history", "maintained", "replaced"]):
            scores[Route.MAINTENANCE_HISTORY] = 0.92
        if any(word in q for word in ["在哪里", "位置", "locate", "where is", "last seen", "last location"]):
            scores[Route.ASSET_LOCATION] = 0.91
            if not has_explicit_id:
                scores[Route.ASSET_LOCATION] = 0.45
                scores[Route.MANUAL_RAG] = max(scores.get(Route.MANUAL_RAG, 0), 0.72)
        tool_intents = sum(
            bool(any(word in q for word in group))
            for group in (
                ["状态", "status", "电量", "battery", "温度", "异常", "healthy"],
                ["telemetry"],
                ["故障", "错误码", "error", "alarm"],
                ["维修记录", "维护记录", "维护历史", "service records", "maintenance records", "maintenance history"],
                ["位置", "在哪里", "locate", "where is", "last location"],
                ["工单", "草稿", "work order", "ticket draft", "draft"],
            )
        )
        if tool_intents >= 2 and any(word in q for word in ["并", "然后", "同时", "后", "再", " and ", " then ", "fetch"]):
            scores[Route.TOOL_PLAN] = 0.96
        has_work_order = any(word in q for word in ["工单", "草稿", "work order", "ticket draft", "draft"])
        asks_policy = any(word in q for word in ["什么时候", "何时", "when", "policy", "可以"])
        if has_work_order and has_explicit_id and not asks_policy:
            scores[Route.TOOL_PLAN] = max(scores.get(Route.TOOL_PLAN, 0), 0.99)
            scores[Route.ERROR_DIAGNOSIS] = min(scores.get(Route.ERROR_DIAGNOSIS, 0), 0.70)
        elif has_work_order and asks_policy:
            scores[Route.MANUAL_RAG] = max(scores.get(Route.MANUAL_RAG, 0), 0.97)
        procedural_intent = (
            any(word in q for word in ["手册", "sop", "如何", "怎么办", "说明", "manual", "procedure"])
            or bool(re.search(r"怎么(?:处理|操作|检查|解决|维护)", q))
        )
        if procedural_intent:
            scores[Route.MANUAL_RAG] = 0.95
            if not has_explicit_id or any(word in q for word in ["手册", "manual"]):
                scores[Route.MANUAL_RAG] = 0.99
                scores[Route.ERROR_DIAGNOSIS] = min(scores.get(Route.ERROR_DIAGNOSIS, 0), 0.70)
        if any(word in q for word in ["写诗", "讲笑话", "poem", "joke", "闲聊"]):
            scores[Route.GENERAL] = 0.65
        return scores

    @staticmethod
    def _extract_entities(query: str) -> Dict[str, str]:
        entities: Dict[str, str] = {}
        device = re.search(r"\b(?:AMR|FORKLIFT|ROBOT|CNC)-?\d+\b", query, re.IGNORECASE)
        if device:
            normalized = device.group(0).upper()
            if "-" not in normalized:
                normalized = re.sub(r"([A-Z]+)(\d+)", r"\1-\2", normalized)
            entities["device_id"] = normalized
            entities["asset_id"] = normalized
        error = re.search(r"\bE\d{3,5}\b", query, re.IGNORECASE)
        if error:
            entities["error_code"] = error.group(0).upper()
        return entities

    @staticmethod
    def _reason(route: Route, query: str) -> str:
        reasons = {
            Route.SAFETY_REJECT: "The request appears to control safety-critical equipment.",
            Route.DEVICE_STATUS: "The request asks for current device status or telemetry.",
            Route.ERROR_DIAGNOSIS: "The request contains a fault or error-code intent.",
            Route.MAINTENANCE_HISTORY: "The request asks about maintenance history.",
            Route.ASSET_LOCATION: "The request asks for the location of an asset.",
            Route.TOOL_PLAN: "The request requires multiple structured tool calls.",
            Route.MANUAL_RAG: "The request requires evidence from manuals or SOP documents.",
            Route.GENERAL: "The request is outside the supported operational domain.",
        }
        return reasons[route]
