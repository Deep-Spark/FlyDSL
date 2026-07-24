"""Handler for example/script style TFLOPS benches (e.g. MR HGEMM)."""

from __future__ import annotations

import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from .base import CaseHandler, CaseResult, RunContext, tail_text

# Example bench output line (see examples/03-tiledMma-iluvatar-mr-pipeline-hgemm.py):
#   [bench] ... 123.4 us/iter  101.23 TFLOPS  (torch 50.7 us, 42.35 TFLOPS)
# Speedup is derived here as torch_us / flydsl_us rather than being emitted by
# the bench script itself so the ratio matches the reported latency numbers.
_BENCH_RE = re.compile(
    r"\[bench\].*?"
    r"([0-9]+(?:\.[0-9]+)?)\s+us/iter\s+"
    r"([0-9]+(?:\.[0-9]+)?)\s+TFLOPS\s+"
    r"\(torch\s+([0-9]+(?:\.[0-9]+)?)\s+us,\s+"
    r"([0-9]+(?:\.[0-9]+)?)\s+TFLOPS"
    r"\)"
)


def parse_bench_metrics(
    output: str,
) -> Optional[Tuple[float, float, float, float, float]]:
    matches = _BENCH_RE.findall(output)
    if not matches:
        return None
    flydsl_us, flydsl_tflops, torch_us, torch_tflops = matches[-1]
    flydsl_us_f = float(flydsl_us)
    torch_us_f = float(torch_us)
    speedup = torch_us_f / flydsl_us_f if flydsl_us_f > 0 else 0.0
    return flydsl_us_f, float(flydsl_tflops), torch_us_f, float(torch_tflops), speedup


class BenchTflopsHandler(CaseHandler):
    type_name = "bench_tflops"

    def run(self, case: Dict[str, Any], ctx: RunContext) -> CaseResult:
        case_id = str(case["id"])
        name = str(case.get("name", case_id))
        entry = str(case.get("entry", ""))
        params = case.get("params") or {}
        if not isinstance(params, dict):
            params = {}

        result = CaseResult(
            id=case_id,
            type=self.type_name,
            name=name,
            entry=entry,
            config={
                "major_pattern": params.get("major_pattern", "nt"),
                "epilogue": params.get("epilogue", "no_c_read"),
                "epilogue_store": params.get("epilogue_store", "shfl"),
                "k_atoms": int(params.get("k_atoms", 2)),
                "warmup": int(params.get("warmup", 5)),
                "iters": int(params.get("iters", 15)),
                "shapes": params.get("shapes", []),
            },
        )

        if not entry:
            result.status = "invalid_sample"
            result.invalid_reason = "missing entry"
            return result

        shapes = params.get("shapes", [])
        if isinstance(shapes, str):
            shape_specs = [s.strip() for s in shapes.split(",") if s.strip()]
        elif isinstance(shapes, list):
            shape_specs = [str(s).strip() for s in shapes if str(s).strip()]
        else:
            shape_specs = []

        if not shape_specs:
            result.status = "invalid_sample"
            result.invalid_reason = "no shapes configured"
            return result

        for spec in shape_specs:
            parts = spec.split("x")
            if len(parts) != 3:
                result.status = "invalid_sample"
                result.invalid_reason = f"invalid shape spec: {spec}"
                result.failure_detail = {"shape": spec}
                break
            m, n, k = parts
            cmd: List[str] = [
                "python3",
                entry,
                "--bench",
                "--epilogue",
                str(result.config["epilogue"]),
                "--epilogue-store",
                str(result.config["epilogue_store"]),
                "--major-pattern",
                str(result.config["major_pattern"]),
                "--k-atoms",
                str(result.config["k_atoms"]),
                "--warmup",
                str(result.config["warmup"]),
                "--iters",
                str(result.config["iters"]),
                "--m",
                m,
                "--n",
                n,
                "--k",
                k,
            ]
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
                result.invalid_reason = f"bench failed for shape {spec}"
                result.failure_detail = {
                    "shape": spec,
                    "returncode": proc.returncode,
                    "stdout_tail": tail_text(proc.stdout or ""),
                    "stderr_tail": tail_text(proc.stderr or ""),
                }
                break

            merged = (proc.stdout or "") + "\n" + (proc.stderr or "")
            parsed = parse_bench_metrics(merged)
            if parsed is None:
                result.status = "invalid_sample"
                result.invalid_reason = f"unable to parse bench metrics for shape {spec}"
                result.failure_detail = {
                    "shape": spec,
                    "returncode": proc.returncode,
                    "stdout_tail": tail_text(proc.stdout or ""),
                    "stderr_tail": tail_text(proc.stderr or ""),
                    "merged_tail": tail_text(merged),
                }
                break

            flydsl_us, tflops, torch_us, torch_tflops, speedup = parsed
            result.metrics[spec] = {
                "tflops": tflops,
                "torch_tflops": torch_tflops,
                "latency_us": flydsl_us,
                "torch_latency_us": torch_us,
                "speedup": speedup,
            }

        return result
