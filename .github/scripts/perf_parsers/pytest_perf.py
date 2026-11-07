"""Handler for pytest-based perf cases.

Contract (printed by the test or a pytest plugin):
  PERF_CASE_JSON={"metrics": {"default": {"latency_us": 12.3}}, ...}

Optional ``params.perf_config`` (mapping) is written to a temp json under the
repo ``.perf/`` directory and exposed to the pytest process as
``FLYDSL_PERF_CONFIG_PATH``. Tests that understand the file can drive their
matrix from the suite json; others ignore the env var.

Phase 1 ships the runner contract; suite cases of this type can be added later
without changing the workflow YAML.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import CaseHandler, CaseResult, RunContext, tail_text

_MARKER_RE = re.compile(r"^PERF_CASE_JSON=(\{.*\})$")
_TS_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T[0-9:.]+Z\s+")

PERF_CONFIG_ENV = "FLYDSL_PERF_CONFIG_PATH"


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


def write_perf_config_file(repo_root: str | Path, case_id: str, perf_config: Dict[str, Any]) -> Path:
    """Serialize ``perf_config`` under ``<repo>/.perf/<case_id>.perf_config.json``.

    Returns the absolute path of the written file (host-side). Callers that pass
    the path into a container must use a repo-relative form instead -- see
    ``env_with_perf_config``.
    """
    root = Path(repo_root)
    out_dir = root / ".perf"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Keep the filename stable and filesystem-safe.
    safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in case_id) or "case"
    path = out_dir / f"{safe_id}.perf_config.json"
    path.write_text(json.dumps(perf_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path.resolve()


def repo_relative_perf_config_path(repo_root: str | Path, absolute_path: Path) -> str:
    """Map a host absolute config path to a repo-relative path for container runs.

    ``run_perf_trend_in_container.sh`` mounts the repo at ``/workspace`` and sets
    ``-w /workspace``. An absolute host path (e.g. under the actions-runner work
    dir) is invisible inside the container; the same relative path resolves on
    both sides.
    """
    root = Path(repo_root).resolve()
    try:
        return str(absolute_path.resolve().relative_to(root))
    except ValueError:
        return str(absolute_path)


def env_with_perf_config(
    base_env: Dict[str, str],
    *,
    repo_root: str | Path,
    case_id: str,
    params: Dict[str, Any],
) -> Dict[str, str]:
    """Copy ``base_env`` and, if ``params.perf_config`` is a mapping, materialize it."""
    env = dict(base_env)
    perf_config = params.get("perf_config")
    if isinstance(perf_config, dict):
        path = write_perf_config_file(repo_root, case_id, perf_config)
        env[PERF_CONFIG_ENV] = repo_relative_perf_config_path(repo_root, path)
    else:
        env.pop(PERF_CONFIG_ENV, None)
    return env


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
                "has_perf_config": isinstance(params.get("perf_config"), dict),
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

        run_env = env_with_perf_config(
            ctx.env,
            repo_root=ctx.repo_root,
            case_id=case_id,
            params=params,
        )
        if PERF_CONFIG_ENV in run_env:
            print(f"# {PERF_CONFIG_ENV}={run_env[PERF_CONFIG_ENV]}")

        in_container_cmd = ["bash", ctx.container_runner, "--", *cmd]
        print(f"$ {' '.join(in_container_cmd)}")
        proc = subprocess.run(
            in_container_cmd,
            text=True,
            capture_output=True,
            env=run_env,
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
