"""Shared types and helpers for perf suite case handlers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def tail_text(text: str, max_lines: int = 60, max_chars: int = 4000) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    tail = "\n".join(lines[-max_lines:])
    if len(tail) > max_chars:
        tail = tail[-max_chars:]
    return tail


@dataclass
class RunContext:
    """Shared execution context for all case handlers."""

    repo_root: str
    container_runner: str
    env: Dict[str, str]
    previous: Optional[Dict[str, Any]] = None


@dataclass
class CaseResult:
    """Normalized per-case result emitted by every handler."""

    id: str
    type: str
    status: str = "ok"
    name: str = ""
    entry: str = ""
    invalid_reason: str = ""
    failure_detail: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    # point_id -> {metric_name: number}
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # "point.metric" -> pct
    delta_vs_prev_pct: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "entry": self.entry,
            "status": self.status,
            "invalid_reason": self.invalid_reason,
            "failure_detail": self.failure_detail,
            "config": self.config,
            "metrics": self.metrics,
            "delta_vs_prev_pct": self.delta_vs_prev_pct,
        }


class CaseHandler:
    """Base class for typed case runners/parsers."""

    type_name = "base"

    def run(self, case: Dict[str, Any], ctx: RunContext) -> CaseResult:
        raise NotImplementedError
