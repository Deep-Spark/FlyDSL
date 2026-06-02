#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""L2 device end-to-end test for the ivcore11 (FlyIXDL) FP16 SME GEMM.

Gating:
  - Module skips unless torch + an Iluvatar GPU are available.
  - The numeric correctness run additionally requires ``FLYDSL_RUN_DEVICE=1``
    (kernel numeric bring-up is finalized via on-device iteration).
  - The compile smoke is marked xfail (non-strict) while the full kernel
    trace/lowering path is being brought up, so the suite stays green while
    surfacing progress (xpass) once it traces cleanly.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l2_device]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_KERNEL_SCRIPT = _REPO_ROOT / "kernels" / "iluvatar_sme_gemm.py"


def _iluvatar_env(base: dict | None = None) -> dict:
    """Return an env dict with the ivcore11 / iluvatar backend pinned.

    flydsl defaults ``FLYDSL_COMPILE_BACKEND`` to ``rocm``; both the compile
    backend and the runtime kind must be ``iluvatar`` for the ixdl lowering +
    device launch path. ``setdefault`` semantics keep explicit overrides.
    """
    env = dict(base if base is not None else os.environ)
    env.setdefault("ARCH", "ivcore11")
    env.setdefault("FLYDSL_COMPILE_BACKEND", "iluvatar")
    env.setdefault("FLYDSL_RUNTIME_KIND", "iluvatar")
    return env


# Pin the backend for the in-process numeric test (and anything importing the
# kernel module) before flydsl resolves the compile/runtime backend.
for _k, _v in (
    ("ARCH", "ivcore11"),
    ("FLYDSL_COMPILE_BACKEND", "iluvatar"),
    ("FLYDSL_RUNTIME_KIND", "iluvatar"),
):
    os.environ.setdefault(_k, _v)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _is_iluvatar_device() -> bool:
    if torch is None or not torch.cuda.is_available():
        return False
    arch = os.environ.get("ARCH", "") or os.environ.get("FLYDSL_GPU_ARCH", "")
    if arch.startswith("ivcore"):
        return True
    try:
        name = torch.cuda.get_device_name(0).lower()
        return "iluvatar" in name or "bi-v" in name or "mr" in name
    except Exception:
        return False


if not _is_iluvatar_device():
    pytest.skip(
        "Iluvatar GPU not available; skipping ivcore11 SME GEMM device tests.",
        allow_module_level=True,
    )


def _device_runtime_healthy() -> bool:
    """Probe (in a subprocess) that the GPU runtime can create a CUDA context.

    A broken/mismatched CoreX driver crashes (SIGSEGV) inside ``libcuda.so.1``
    on the first context init, taking down the whole interpreter. Running the
    probe in a subprocess lets us turn that into a graceful skip instead of a
    misleading xfail/crash.
    """
    import subprocess as _sp
    import sys as _sys

    probe = "import torch; x = torch.randn(8, dtype=torch.float16).cuda(); " "torch.cuda.synchronize(); print('OK', float(x.float().sum()))"
    try:
        proc = _sp.run(
            [_sys.executable, "-c", probe],
            text=True,
            capture_output=True,
            timeout=120,
        )
    except Exception:
        return False
    return proc.returncode == 0 and "OK" in proc.stdout


if not _device_runtime_healthy():
    pytest.skip(
        "Iluvatar CoreX GPU runtime is unhealthy (CUDA context init crashes; "
        "e.g. libcuda.so.1 / kernel-module version skew). Fix the driver/runtime "
        "before running on-device numeric validation.",
        allow_module_level=True,
    )


M, N, K = 16, 16, 32  # Phase A: single warp, single block.


def _build():
    from kernels.iluvatar_sme_gemm import build_sme_gemm

    return build_sme_gemm(M, N, K)


def test_sme_gemm_compile_smoke():
    """The kernel traces + compiles + launches through the iluvatar pipeline.

    Runs in a subprocess so that any hard crash during the kernel run is
    isolated as a non-zero exit code (asserted below) instead of taking down
    the whole pytest process. Numeric correctness is validated separately in
    ``test_sme_gemm_matches_torch`` (gated by ``FLYDSL_RUN_DEVICE``).
    """
    env = _iluvatar_env()
    proc = subprocess.run(
        [sys.executable, str(_KERNEL_SCRIPT)],
        cwd=str(_REPO_ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=600,
    )
    assert proc.returncode == 0, (
        f"kernel script exited {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout[-2000:]}\n"
        f"--- stderr ---\n{proc.stderr[-2000:]}"
    )


@pytest.mark.skipif(
    os.environ.get("FLYDSL_RUN_DEVICE") != "1",
    reason="set FLYDSL_RUN_DEVICE=1 to run on-device numeric validation",
)
def test_sme_gemm_matches_torch():
    """Random-input correctness vs torch reference (atol/rtol 1e-1)."""
    sme_gemm = _build()
    A = torch.randn(M, K, dtype=torch.float16).cuda()
    B = torch.randn(N, K, dtype=torch.float16).cuda()
    C = torch.zeros(M, N, dtype=torch.float32).cuda()

    sme_gemm(A, B, C, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    expected = A.float() @ B.float().T
    max_diff = (C - expected).abs().max().item()
    assert torch.allclose(C, expected, atol=1e-1, rtol=1e-1), f"max_diff={max_diff:.6f}"
