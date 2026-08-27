"""Lightweight retrieval and token-aware context packing for EdgeOps."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

from edgeops.contracts import Evidence


@dataclass
class Document:
    document_id: str
    text: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredDocument:
    document: Document
    score: float


class LexicalRetriever:
    """Dependency-free retrieval used for smoke tests and offline fallback."""

    def __init__(self, documents: Iterable[Document]) -> None:
        self.documents = list(documents)
        self.doc_tokens = [self._tokens(doc.text) for doc in self.documents]

    @classmethod
    def from_jsonl(cls, path: str) -> "LexicalRetriever":
        documents = []
        source_path = Path(path)
        if source_path.exists():
            for index, line in enumerate(source_path.read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                item = json.loads(line)
                documents.append(
                    Document(
                        document_id=item.get("id", "doc-%04d" % index),
                        text=item["text"],
                        source=item.get("source", source_path.name),
                        metadata=item.get("metadata", {}),
                    )
                )
        return cls(documents)

    def search(self, query: str, top_k: int = 6) -> List[ScoredDocument]:
        query_tokens = self._tokens(query)
        results = []
        for document, tokens in zip(self.documents, self.doc_tokens):
            if not tokens or not query_tokens:
                continue
            overlap = len(query_tokens & tokens)
            union = len(query_tokens | tokens)
            score = overlap / max(union, 1)
            error_code = document.metadata.get("error_code", "").lower()
            if error_code and error_code in query.lower():
                score += 0.8
            device_type = document.metadata.get("device_type", "").lower()
            if device_type and device_type in query.lower():
                score += 0.2
            if score > 0:
                results.append(ScoredDocument(document=document, score=score))
        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]

    @staticmethod
    def _tokens(text: str) -> Set[str]:
        lowered = text.lower()
        words = set(re.findall(r"[a-z0-9_-]+", lowered))
        chinese = re.findall(r"[\u4e00-\u9fff]", lowered)
        if len(chinese) == 1:
            words.update(chinese)
        else:
            words.update(
                "".join(chinese[index : index + 2])
                for index in range(len(chinese) - 1)
            )
        return words


class ContextPacker:
    def __init__(
        self,
        token_budget: int = 384,
        token_counter: Optional[Callable[[str], int]] = None,
        redundancy_weight: float = 0.25,
    ) -> None:
        self.token_budget = token_budget
        self.token_counter = token_counter or self._estimate_tokens
        self.redundancy_weight = redundancy_weight

    def dynamic_budget(self, query: str, route_confidence: float) -> int:
        complexity_bonus = min(len(query) // 8, 96)
        uncertainty_bonus = int(max(0.0, 0.8 - route_confidence) * 200)
        return min(self.token_budget + complexity_bonus + uncertainty_bonus, 640)

    def pack(
        self,
        candidates: List[ScoredDocument],
        query: str,
        route_confidence: float,
    ) -> List[Evidence]:
        budget = self.dynamic_budget(query, route_confidence)
        selected: List[Evidence] = []
        selected_tokens: Set[str] = set()
        remaining = list(candidates)

        while remaining:
            best_index = -1
            best_utility = float("-inf")
            for index, candidate in enumerate(remaining):
                tokens = LexicalRetriever._tokens(candidate.document.text)
                redundancy = (
                    len(tokens & selected_tokens) / max(len(tokens | selected_tokens), 1)
                    if selected_tokens
                    else 0.0
                )
                utility = candidate.score - self.redundancy_weight * redundancy
                if utility > best_utility:
                    best_utility = utility
                    best_index = index

            candidate = remaining.pop(best_index)
            count = self.token_counter(candidate.document.text)
            used = sum(item.token_estimate for item in selected)
            if used + count > budget:
                continue
            selected.append(
                Evidence(
                    document_id=candidate.document.document_id,
                    text=candidate.document.text,
                    source=candidate.document.source,
                    score=round(candidate.score, 4),
                    token_estimate=count,
                    metadata=candidate.document.metadata,
                )
            )
            selected_tokens.update(LexicalRetriever._tokens(candidate.document.text))
        return selected

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        chinese_count = len(re.findall(r"[\u4e00-\u9fff]", text))
        other = re.sub(r"[\u4e00-\u9fff]", " ", text)
        word_count = len(re.findall(r"\w+|[^\w\s]", other))
        return max(1, chinese_count + word_count)
