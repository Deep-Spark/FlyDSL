#!/usr/bin/env python3
"""Render GitHub Step Summary from .perf/result.json (normalized suite schema)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple


def bucket_for_delta(delta: float, flat_threshold_pct: float) -> str:
    if delta <= -flat_threshold_pct:
        return "regressed"
    if delta >= flat_threshold_pct:
        return "improved"
    return "flat"


def case_bucket(case_result: Dict[str, Any], flat_threshold_pct: float) -> str:
    if case_result.get("status") != "ok":
        return "invalid"
    deltas = [float(v) for v in (case_result.get("delta_vs_prev_pct") or {}).values() if isinstance(v, (int, float))]
    if not deltas:
        return "flat"
    if any(d <= -flat_threshold_pct for d in deltas):
        return "regressed"
    if any(d >= flat_threshold_pct for d in deltas):
        return "improved"
    return "flat"


def normalize_cases(result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    cases = result.get("cases")
    if isinstance(cases, dict) and cases:
        return {str(k): v for k, v in cases.items() if isinstance(v, dict)}

    # Legacy kernels -> cases adapter for one-release continuity.
    kernels = result.get("kernels")
    if not isinstance(kernels, dict) or not kernels:
        return {}

    adapted: Dict[str, Dict[str, Any]] = {}
    for kernel_id, kernel in kernels.items():
        if not isinstance(kernel, dict):
            continue
        metrics_in = kernel.get("metrics") or {}
        torch_metrics = kernel.get("torch_metrics") or {}
        speedups = kernel.get("speedup_vs_torch") or {}
        metrics_out: Dict[str, Dict[str, float]] = {}
        if isinstance(metrics_in, dict):
            for point, value in metrics_in.items():
                if isinstance(value, dict):
                    metrics_out[str(point)] = {
                        str(mk): float(mv) for mk, mv in value.items() if isinstance(mv, (int, float))
                    }
                elif isinstance(value, (int, float)):
                    point_metrics = {"tflops": float(value)}
                    tr = torch_metrics.get(point) if isinstance(torch_metrics, dict) else None
                    su = speedups.get(point) if isinstance(speedups, dict) else None
                    if isinstance(tr, (int, float)):
                        point_metrics["torch_tflops"] = float(tr)
                    if isinstance(su, (int, float)):
                        point_metrics["speedup"] = float(su)
                    metrics_out[str(point)] = point_metrics

        deltas_in = kernel.get("delta_vs_prev_pct") or {}
        deltas_out: Dict[str, float] = {}
        if isinstance(deltas_in, dict):
            for key, value in deltas_in.items():
                if not isinstance(value, (int, float)):
                    continue
                key_s = str(key)
                if "." in key_s:
                    deltas_out[key_s] = float(value)
                else:
                    deltas_out[f"{key_s}.tflops"] = float(value)

        adapted[str(kernel_id)] = {
            "id": str(kernel_id),
            "type": "bench_tflops",
            "name": kernel.get("name", ""),
            "entry": kernel.get("entry", ""),
            "status": kernel.get("status", "unknown"),
            "invalid_reason": kernel.get("invalid_reason", ""),
            "failure_detail": kernel.get("failure_detail", {}),
            "config": kernel.get("config", {}),
            "metrics": metrics_out,
            "delta_vs_prev_pct": deltas_out,
        }
    return adapted


def primary_metric_for_case(case_result: Dict[str, Any]) -> str:
    metrics = case_result.get("metrics") or {}
    if not isinstance(metrics, dict) or not metrics:
        return "tflops"
    first = next(iter(metrics.values()))
    if isinstance(first, dict):
        if "tflops" in first:
            return "tflops"
        if "latency_us" in first:
            return "latency_us"
        if first:
            return str(next(iter(first.keys())))
    return "tflops"


def generic_metric_sort_key(metric_name: str) -> Tuple[int, int, str]:
    raw_metric_order = {
        "latency_us": 0,
        "torch_latency_us": 1,
        "ixdnn_latency_us": 2,
        "ixblas_latency_us": 3,
    }
    derived_metric_order = {
        "speedup": 0,
        "speedup_torch": 1,
        "speedup_ixdnn": 2,
        "speedup_ixblas": 3,
    }
    if metric_name in raw_metric_order:
        return (0, raw_metric_order[metric_name], metric_name)
    if metric_name in derived_metric_order:
        return (1, derived_metric_order[metric_name], metric_name)
    if metric_name.startswith("speedup"):
        return (1, len(derived_metric_order), metric_name)
    return (0, len(raw_metric_order), metric_name)


def render(result: Dict[str, Any], flat_threshold_pct: float, input_reason: str) -> str:
    cases = normalize_cases(result)
    cfg = result.get("config") or {}
    if not isinstance(cfg, dict):
        cfg = {}

    lines: List[str] = []
    lines.append("### Daily performance trend (Iluvatar)")
    lines.append("")
    lines.append("- mode: `trend-only` (non-blocking)")
    lines.append(f"- branch: `{cfg.get('branch', os.environ.get('PERF_BRANCH', 'iluvatar'))}`")
    lines.append(f"- suite: `{result.get('suite', cfg.get('suite_file', 'n/a'))}`")
    if cfg.get("suite_file"):
        lines.append(f"- suite file: `{cfg.get('suite_file')}`")
    lines.append(f"- flydsl_commit: `{result.get('flydsl_commit', 'unknown')}`")
    lines.append(f"- ixcc_commit: `{result.get('ixcc_commit', 'unknown') or '<unknown>'}`")
    lines.append(f"- ixsdk_commit: `{result.get('ixsdk_commit', 'unknown') or '<unknown>'}`")
    if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
        lines.append(f"- manual reason: `{input_reason}`")
    lines.append(f"- delta threshold: `flat if |Δ| < {flat_threshold_pct:.1f}%`")

    regressed = flat = improved = invalid = 0
    for case_result in cases.values():
        bucket = case_bucket(case_result, flat_threshold_pct)
        if bucket == "regressed":
            regressed += 1
        elif bucket == "improved":
            improved += 1
        elif bucket == "invalid":
            invalid += 1
        else:
            flat += 1
    lines.append(
        f"- rollup: RED regressed `{regressed}` | YELLOW flat `{flat}` | "
        f"GREEN improved `{improved}` | INVALID `{invalid}`"
    )
    lines.append("")

    ordered: List[Tuple[str, Dict[str, Any]]] = sorted(
        cases.items(),
        key=lambda kv: (0 if kv[1].get("status") != "ok" else 1, kv[0].lower()),
    )

    for case_id, case_result in ordered:
        title = case_id
        name = case_result.get("name")
        entry = case_result.get("entry")
        if name and entry:
            title = f"{case_id} — {name} ({entry})"
        lines.append(f"#### {title}")
        lines.append("")
        lines.append(f"- status: `{case_result.get('status', 'unknown')}`")
        lines.append(f"- type: `{case_result.get('type', 'n/a')}`")
        kcfg = case_result.get("config") or {}
        if isinstance(kcfg, dict) and kcfg:
            if case_result.get("type") == "bench_tflops":
                lines.append(
                    f"- bench config: pattern=`{kcfg.get('major_pattern', 'n/a')}`, "
                    f"epilogue=`{kcfg.get('epilogue', 'n/a')}`, "
                    f"store=`{kcfg.get('epilogue_store', 'n/a')}`, "
                    f"k_atoms=`{kcfg.get('k_atoms', 'n/a')}`, "
                    f"warmup=`{kcfg.get('warmup', 'n/a')}`, "
                    f"iters=`{kcfg.get('iters', 'n/a')}`"
                )
            else:
                compact = ", ".join(f"{k}=`{v}`" for k, v in kcfg.items())
                lines.append(f"- config: {compact}")

        if case_result.get("status") != "ok":
            lines.append(f"- reason: `{case_result.get('invalid_reason', 'unknown')}`")
            detail = case_result.get("failure_detail") or {}
            if isinstance(detail, dict) and detail:
                shape = detail.get("shape", "")
                returncode = detail.get("returncode", "")
                if shape:
                    lines.append(f"- failed shape: `{shape}`")
                if returncode != "":
                    lines.append(f"- returncode: `{returncode}`")
                for label, key in (
                    ("stderr tail", "stderr_tail"),
                    ("stdout tail", "stdout_tail"),
                    ("output tail", "merged_tail"),
                ):
                    text = detail.get(key, "")
                    if text:
                        lines.append("")
                        lines.append(f"{label}:")
                        lines.append("```")
                        lines.append(str(text))
                        lines.append("```")
            lines.append("")
            continue

        metrics = case_result.get("metrics") or {}
        deltas = case_result.get("delta_vs_prev_pct") or {}
        if not isinstance(metrics, dict):
            metrics = {}
        if not isinstance(deltas, dict):
            deltas = {}

        primary = primary_metric_for_case(case_result)
        has_torch = any(
            isinstance(v, dict) and isinstance(v.get("torch_tflops"), (int, float)) for v in metrics.values()
        )
        has_speedup = any(isinstance(v, dict) and isinstance(v.get("speedup"), (int, float)) for v in metrics.values())
        has_latency = any(
            isinstance(v, dict) and isinstance(v.get("latency_us"), (int, float)) for v in metrics.values()
        )
        has_torch_latency = any(
            isinstance(v, dict) and isinstance(v.get("torch_latency_us"), (int, float)) for v in metrics.values()
        )

        lines.append("")
        if primary == "tflops" and has_torch and has_latency and has_torch_latency:
            header = (
                "| point | FlyDSL us | torch us | FlyDSL TFLOPS | torch TFLOPS " "| speedup (us: torch_us/flydsl_us) |"
            )
            sep = "|---|---:|---:|---:|---:|---:|"
            lines.append(header)
            lines.append(sep)
            for point in sorted(metrics):
                values = metrics[point] if isinstance(metrics[point], dict) else {}
                cur_us = values.get("latency_us")
                torch_us = values.get("torch_latency_us")
                cur = values.get("tflops")
                torch_ref = values.get("torch_tflops")
                speedup = values.get("speedup")
                cur_us_text = "n/a" if not isinstance(cur_us, (int, float)) else f"{cur_us:.1f}"
                torch_us_text = "n/a" if not isinstance(torch_us, (int, float)) else f"{torch_us:.1f}"
                cur_text = "n/a" if not isinstance(cur, (int, float)) else f"{cur:.2f}"
                torch_text = "n/a" if not isinstance(torch_ref, (int, float)) else f"{torch_ref:.2f}"
                speedup_text = "n/a" if not isinstance(speedup, (int, float)) else f"{speedup:.2f}x"
                if not has_speedup:
                    speedup_text = "n/a"
                lines.append(
                    f"| `{point}` | `{cur_us_text}` | `{torch_us_text}` "
                    f"| `{cur_text}` | `{torch_text}` | `{speedup_text}` |"
                )
        elif primary == "tflops" and has_torch:
            header = "| point | TFLOPS | torch TFLOPS | speedup |"
            sep = "|---|---:|---:|---:|"
            lines.append(header)
            lines.append(sep)
            for point in sorted(metrics):
                values = metrics[point] if isinstance(metrics[point], dict) else {}
                cur = values.get("tflops")
                torch_ref = values.get("torch_tflops")
                speedup = values.get("speedup")
                cur_text = "n/a" if not isinstance(cur, (int, float)) else f"{cur:.2f}"
                torch_text = "n/a" if not isinstance(torch_ref, (int, float)) else f"{torch_ref:.2f}"
                speedup_text = "n/a" if not isinstance(speedup, (int, float)) else f"{speedup:.2f}x"
                if not has_speedup:
                    speedup_text = "n/a"
                lines.append(f"| `{point}` | `{cur_text}` | `{torch_text}` | `{speedup_text}` |")
        else:
            # Generic metric table: raw measurements first, then derived metrics.
            metric_names = sorted(
                {
                    str(mk)
                    for values in metrics.values()
                    if isinstance(values, dict)
                    for mk, mv in values.items()
                    if isinstance(mv, (int, float))
                },
                key=generic_metric_sort_key,
            )
            if not metric_names:
                metric_names = [primary]
            header = "| point | " + " | ".join(metric_names) + " |"
            sep = "|---|" + "|".join(["---:"] * len(metric_names)) + "|"
            lines.append(header)
            lines.append(sep)
            for point in sorted(metrics):
                values = metrics[point] if isinstance(metrics[point], dict) else {}
                cells = []
                for mk in metric_names:
                    val = values.get(mk)
                    cells.append("n/a" if not isinstance(val, (int, float)) else f"{val:.2f}")
                lines.append(f"| `{point}` | " + " | ".join(f"`{c}`" for c in cells) + " |")

        lines.append("")
        lines.append("| point | metric | Δ vs prev | bucket |")
        lines.append("|---|---|---:|---|")
        # Prefer explicit delta keys; otherwise synthesize n/a rows for primary metric.
        delta_keys = sorted(str(k) for k in deltas.keys())
        if not delta_keys:
            for point in sorted(metrics):
                delta_keys.append(f"{point}.{primary}")
        for key in delta_keys:
            if "." in key:
                point, metric_name = key.rsplit(".", 1)
            else:
                point, metric_name = key, primary
            delta = deltas.get(key)
            if not isinstance(delta, (int, float)):
                delta_text = "n/a"
                bucket_text = "n/a"
            else:
                delta_text = f"{float(delta):+.2f}%"
                bucket_text = bucket_for_delta(float(delta), flat_threshold_pct)
            lines.append(f"| `{point}` | `{metric_name}` | `{delta_text}` | `{bucket_text}` |")
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    result_path = Path(os.environ.get("PERF_RESULT_FILE", ".perf/result.json"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    flat_threshold_pct = float(
        result.get("config", {}).get("delta_flat_pct")
        if isinstance(result.get("config"), dict) and result.get("config", {}).get("delta_flat_pct") is not None
        else os.environ.get("PERF_DELTA_FLAT_PCT", "3")
    )
    input_reason = os.environ.get("INPUT_REASON", "")
    text = render(result, flat_threshold_pct, input_reason)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        Path(summary_path).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
