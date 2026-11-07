#!/usr/bin/env python3
"""Unified perf suite runner.

Reads a suite YAML, dispatches typed case handlers, computes deltas against
previous metrics, and writes .perf/result.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow `from perf_parsers import ...` when invoked as a script.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from perf_parsers import get_handler  # noqa: E402
from perf_parsers.base import RunContext  # noqa: E402


def git_short(path: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=path,
        text=True,
        capture_output=True,
    )
    if proc.returncode == 0:
        return proc.stdout.strip()
    return "unknown"


def _legacy_defaults() -> Dict[str, Any]:
    return {
        "delta_flat_pct": 3,
        "env": {
            "FLYDSL_COMPILE_BACKEND": "iluvatar",
            "FLYDSL_RUNTIME_KIND": "iluvatar",
            "ARCH": "ivcore11",
            "FLYDSL_RUNTIME_ENABLE_CACHE": "0",
            "PYTHONUNBUFFERED": "1",
        },
    }


def _normalize_legacy_case(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    name = str(item.get("name", f"case_{idx}")).strip()
    entry = str(item.get("entry", "")).strip()
    case_id = str(item.get("id", name)).strip() or f"case_{idx}"
    case_type = str(item.get("type", "bench_tflops")).strip() or "bench_tflops"
    params = {
        "major_pattern": str(item.get("major_pattern", "nt")),
        "epilogue": str(item.get("epilogue", "no_c_read")),
        "epilogue_store": str(item.get("epilogue_store", "shfl")),
        "k_atoms": int(item.get("k_atoms", 2)),
        "warmup": int(item.get("warmup", 5)),
        "iters": int(item.get("iters", 15)),
        "shapes": item.get("shapes", ""),
    }
    gate = {
        "compare": "delta_vs_prev",
        "metric": "tflops",
        "flat_pct": 3,
    }
    return {
        "id": case_id,
        "type": case_type,
        "name": name,
        "entry": entry,
        "params": params,
        "gate": gate,
    }


def _parse_suite_text(text: str, path: Path) -> Any:
    # Prefer JSON to avoid a hard pyyaml dependency on the host runner.
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import yaml  # optional; only required for real YAML suites

        return yaml.safe_load(text)


def load_suite(path: Path) -> Dict[str, Any]:
    data = _parse_suite_text(path.read_text(encoding="utf-8"), path)
    # Backward compatibility: .github/perf-kernels-iluvatar.json legacy array format.
    if isinstance(data, list):
        cases: List[Dict[str, Any]] = []
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"legacy suite case must be a mapping: index={idx}")
            cases.append(_normalize_legacy_case(item, idx))
        if not cases:
            raise ValueError(f"legacy suite file must contain a non-empty case list: {path}")
        return {
            "suite": path.stem,
            "defaults": _legacy_defaults(),
            "cases": cases,
        }

    if not isinstance(data, dict):
        raise ValueError(f"suite file must be a mapping: {path}")
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"suite file must contain a non-empty cases list: {path}")
    return data


def load_previous(prev_file: Path) -> Optional[Dict[str, Any]]:
    if not prev_file.exists():
        return None
    try:
        data = json.loads(prev_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def previous_case_metrics(previous: Optional[Dict[str, Any]], case_id: str) -> Dict[str, Dict[str, float]]:
    """Return point -> metric -> value for a prior case id.

    Supports:
      - new schema: previous["cases"][case_id]["metrics"]
      - legacy schema: previous["kernels"][legacy_id]["metrics"] as flat floats
    """
    if not previous:
        return {}

    cases = previous.get("cases")
    if isinstance(cases, dict) and case_id in cases and isinstance(cases[case_id], dict):
        metrics = cases[case_id].get("metrics", {})
        if isinstance(metrics, dict):
            out: Dict[str, Dict[str, float]] = {}
            for point, values in metrics.items():
                if isinstance(values, dict):
                    out[str(point)] = {str(k): float(v) for k, v in values.items() if isinstance(v, (int, float))}
                elif isinstance(values, (int, float)):
                    # legacy nested under cases somehow
                    out[str(point)] = {"tflops": float(values)}
            return out

    # Legacy kernels map: metrics are flat point -> tflops
    kernels = previous.get("kernels")
    if isinstance(kernels, dict):
        for kernel in kernels.values():
            if not isinstance(kernel, dict):
                continue
            # Best-effort match by name/entry if ids differ
            if str(kernel.get("name", "")) == case_id or str(kernel.get("id", "")) == case_id:
                metrics = kernel.get("metrics", {})
                if isinstance(metrics, dict):
                    return {
                        str(point): {"tflops": float(val)}
                        for point, val in metrics.items()
                        if isinstance(val, (int, float))
                    }
        # Single-kernel legacy fallback when only one case ran historically
        if len(kernels) == 1:
            only = next(iter(kernels.values()))
            if isinstance(only, dict):
                metrics = only.get("metrics", {})
                if isinstance(metrics, dict) and metrics and all(isinstance(v, (int, float)) for v in metrics.values()):
                    return {
                        str(point): {"tflops": float(val)}
                        for point, val in metrics.items()
                        if isinstance(val, (int, float))
                    }

    # Ultra-legacy top-level metrics
    top = previous.get("metrics")
    if isinstance(top, dict) and top and all(isinstance(v, (int, float)) for v in top.values()):
        return {str(point): {"tflops": float(val)} for point, val in top.items()}

    return {}


def apply_deltas(
    case_result: Dict[str, Any],
    prev_metrics: Dict[str, Dict[str, float]],
    gate_metric: str,
) -> None:
    deltas: Dict[str, float] = {}
    metrics = case_result.get("metrics") or {}
    if not isinstance(metrics, dict):
        case_result["delta_vs_prev_pct"] = {}
        return

    for point, values in metrics.items():
        if not isinstance(values, dict):
            continue
        # Prefer configured gate metric; otherwise compute for all numeric metrics.
        metric_names = [gate_metric] if gate_metric in values else list(values.keys())
        for metric_name in metric_names:
            cur = values.get(metric_name)
            prev = (prev_metrics.get(str(point)) or {}).get(metric_name)
            if isinstance(cur, (int, float)) and isinstance(prev, (int, float)) and prev != 0:
                deltas[f"{point}.{metric_name}"] = (float(cur) - float(prev)) / float(prev) * 100.0

    case_result["delta_vs_prev_pct"] = deltas


def build_env(suite_defaults: Dict[str, Any], corex_root: Path) -> Dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = f"{corex_root}/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = f"{corex_root}/lib64:{corex_root}/lib:" + env.get("LD_LIBRARY_PATH", "")
    default_env = suite_defaults.get("env") or {}
    if isinstance(default_env, dict):
        for key, value in default_env.items():
            env[str(key)] = str(value)
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        required=True,
        help="Path to suite YAML (e.g. .ci-utils/.github/perf/suites/iluvatar.yaml)",
    )
    parser.add_argument(
        "--container-runner",
        default=".ci-utils/.github/scripts/run_perf_trend_in_container.sh",
        help="Host script that launches commands inside the perf container",
    )
    parser.add_argument(
        "--perf-dir",
        default=".perf",
        help="Directory for previous_metrics.json / result.json",
    )
    args = parser.parse_args()

    repo_root = Path(".").resolve()
    suite_path = Path(args.suite)
    perf_dir = Path(args.perf_dir)
    perf_dir.mkdir(parents=True, exist_ok=True)
    result_file = perf_dir / "result.json"
    prev_file = perf_dir / "previous_metrics.json"

    suite = load_suite(suite_path)
    defaults = suite.get("defaults") or {}
    if not isinstance(defaults, dict):
        defaults = {}
    suite_name = str(suite.get("suite", suite_path.stem))

    start_ts = int(time.time())
    result: Dict[str, Any] = {
        "status": "ok",
        "invalid_reason": "",
        "failure_detail": {},
        "suite": suite_name,
        "flydsl_commit": git_short(repo_root),
        "ixcc_commit": os.environ.get("IXCC_COMMIT", "unknown"),
        "ixsdk_commit": os.environ.get("IXSDK_COMMIT", "unknown"),
        "config": {
            "branch": os.environ.get("PERF_BRANCH", "iluvatar"),
            "suite_file": str(suite_path),
            "delta_flat_pct": float(defaults.get("delta_flat_pct", os.environ.get("PERF_DELTA_FLAT_PCT", 3))),
        },
        "cases": {},
        "timestamp": start_ts,
    }

    previous = load_previous(prev_file)
    corex_root = Path(os.environ.get("COREX_ROOT", ""))

    def invalidate(reason: str, detail: Optional[Dict[str, Any]] = None) -> None:
        result["status"] = "invalid_sample"
        result["invalid_reason"] = reason
        if detail:
            result["failure_detail"] = detail

    if not corex_root.is_dir():
        invalidate(f"COREX_ROOT not found: {corex_root}")
    if result["status"] == "ok" and os.environ.get("IX_TOOLCHAIN_OUTCOME") != "success":
        invalidate("ix toolchain refresh failed")

    if result["status"] == "ok":
        env = build_env(defaults, corex_root)
        ctx = RunContext(
            repo_root=str(repo_root),
            container_runner=args.container_runner,
            env=env,
            previous=previous,
        )

        for case in suite["cases"]:
            if not isinstance(case, dict):
                invalidate("suite case must be a mapping")
                break
            case_id = str(case.get("id", "")).strip()
            case_type = str(case.get("type", "")).strip()
            if not case_id or not case_type:
                invalidate("each case requires non-empty id and type")
                break
            try:
                handler = get_handler(case_type)
            except KeyError as exc:
                invalidate(str(exc))
                break

            case_result_obj = handler.run(case, ctx)
            case_dict = case_result_obj.to_dict()

            gate = case.get("gate") or {}
            gate_metric = str(gate.get("metric", "tflops"))
            prev_metrics = previous_case_metrics(previous, case_id)
            # Also try matching legacy kernel id "name (entry)" for continuity.
            if not prev_metrics and case_result_obj.name and case_result_obj.entry:
                legacy_id = f"{case_result_obj.name} ({case_result_obj.entry})"
                if previous and isinstance(previous.get("kernels"), dict):
                    legacy = previous["kernels"].get(legacy_id)
                    if isinstance(legacy, dict):
                        metrics = legacy.get("metrics", {})
                        if isinstance(metrics, dict):
                            prev_metrics = {
                                str(point): {"tflops": float(val)}
                                for point, val in metrics.items()
                                if isinstance(val, (int, float))
                            }
            apply_deltas(case_dict, prev_metrics, gate_metric)
            result["cases"][case_id] = case_dict

        invalid_items = [(cid, cdata) for cid, cdata in result["cases"].items() if cdata.get("status") != "ok"]
        if invalid_items and result["status"] == "ok":
            first_id, first = invalid_items[0]
            result["status"] = "invalid_sample"
            result["invalid_reason"] = f"{first_id}: {first.get('invalid_reason', 'unknown')}"
            detail = dict(first.get("failure_detail") or {})
            if detail:
                detail["case_id"] = first_id
            result["failure_detail"] = detail

    result_file.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    # Keep marker for one release as debug aid; artifact is source of truth.
    print("PERF_METRIC_JSON=" + json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
