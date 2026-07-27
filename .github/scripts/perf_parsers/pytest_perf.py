"""Handler for pytest-based perf cases.

Contract (printed by the test or a pytest plugin):
  PERF_CASE_JSON={"metrics": {"default": {"latency_us": 12.3}}, ...}

Phase 1 ships the runner contract; suite cases of this type can be added later
without changing the workflow YAML.
"""

from __future__ import annotations

import json
import re
import subprocess
from typing import Any, Dict, List, Optional

from .base import CaseHandler, CaseResult, RunContext, tail_text

_MARKER_RE = re.compile(r"^PERF_CASE_JSON=(\{.*\})$")
_TS_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T[0-9:.]+Z\s+")


def parse_perf_case_json(output: str) -> Optional[Dict[str, Any]]:
    found = None
    for raw in output.splitlines():
        text = _TS_PREFIX_RE.sub("", raw.strip(), count=1)
        m = _MARKER_RE.match(text)
        if not m:
            continue
        try:
            found = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
    return found if isinstance(found, dict) else None


class PytestPerfHandler(CaseHandler):
    type_name = "pytest_perf"

    def run(self, case: Dict[str, Any], ctx: RunContext) -> CaseResult:
        case_id = str(case["id"])
        entry = str(case.get("entry", ""))
        name = str(case.get("name", case_id))
        params = case.get("params") or {}
        if not isinstance(params, dict):
            params = {}

        result = CaseResult(
            id=case_id,
            type=self.type_name,
            name=name,
            entry=entry,
            config={
                "markers": params.get("markers", ""),
                "extra_args": params.get("extra_args", []),
            },
        )
        if not entry:
            result.status = "invalid_sample"
            result.invalid_reason = "missing entry"
            return result

        cmd: List[str] = ["python3", "-m", "pytest", entry, "-q", "-s"]
        markers = str(params.get("markers", "")).strip()
        if markers:
            cmd.extend(["-m", markers])
        extra = params.get("extra_args") or []
        if isinstance(extra, list):
            cmd.extend(str(x) for x in extra)

        in_container_cmd = ["bash", ctx.container_runner, "--", *cmd]
        print(f"$ {' '.join(in_container_cmd)}")
        proc = subprocess.run(
            in_container_cmd,
            text=True,
            capture_output=True,
            env=ctx.env,
            cwd=ctx.repo_root,
        )
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="")

        if proc.returncode != 0:
            result.status = "invalid_sample"
            result.invalid_reason = "pytest_perf failed"
            result.failure_detail = {
                "returncode": proc.returncode,
                "stdout_tail": tail_text(proc.stdout or ""),
                "stderr_tail": tail_text(proc.stderr or ""),
            }
            return result

        merged = (proc.stdout or "") + "\n" + (proc.stderr or "")
        payload = parse_perf_case_json(merged)
        if payload is None:
            result.status = "invalid_sample"
            result.invalid_reason = "unable to parse PERF_CASE_JSON marker"
            result.failure_detail = {"merged_tail": tail_text(merged)}
            return result

        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict) or not metrics:
            result.status = "invalid_sample"
            result.invalid_reason = "PERF_CASE_JSON missing metrics"
            result.failure_detail = {"payload": payload}
            return result

        # Normalize: each point maps to a metric dict of floats.
        for point, values in metrics.items():
            if isinstance(values, dict):
                point_metrics: Dict[str, float] = {}
                for metric_name, value in values.items():
                    if isinstance(value, (int, float)):
                        point_metrics[str(metric_name)] = float(value)
                if point_metrics:
                    result.metrics[str(point)] = point_metrics
            elif isinstance(values, (int, float)):
                # Convenience: bare number -> default metric name from gate or "value"
                gate = case.get("gate") or {}
                metric_name = str(gate.get("metric", "value"))
                result.metrics[str(point)] = {metric_name: float(values)}

        if not result.metrics:
            result.status = "invalid_sample"
            result.invalid_reason = "PERF_CASE_JSON metrics contained no numeric values"
        return result
