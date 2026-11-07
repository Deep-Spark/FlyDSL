"""Registry of perf case handlers."""

from __future__ import annotations

from typing import Dict

from .base import CaseHandler
from .bench_tflops import BenchTflopsHandler
from .pytest_perf import PytestPerfHandler

PARSERS: Dict[str, CaseHandler] = {
    BenchTflopsHandler.type_name: BenchTflopsHandler(),
    PytestPerfHandler.type_name: PytestPerfHandler(),
}


def get_handler(case_type: str) -> CaseHandler:
    handler = PARSERS.get(case_type)
    if handler is None:
        known = ", ".join(sorted(PARSERS))
        raise KeyError(f"unknown case type {case_type!r}; known: {known}")
    return handler
