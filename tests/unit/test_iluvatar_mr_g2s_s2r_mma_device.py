# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Opt-in Iluvatar MR G2S -> S2R -> MMA staged device tests.

Chains production ``mr_hgemm_g2s_issue_operands``, ``mr_hgemm_s2r_copy_*``, and a
single ``fx.gemm`` on warp-00 atom (im=0,jn=0). No epilogue, no multi-K-tile loop.

This is the first stage test that exercises ``make_tiled_copy_A/B`` on real G2S
smem tiles. Scalar S2R readback passing does not imply this path is correct.

Set ``FLYDSL_ILUVATAR_RUN_MR_S2R_MMA=1`` to run (needs an Iluvatar device).
"""

import os
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l2_device]

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kernels.iluvatar_mr_common import ATOM_M, ATOM_N  # noqa: E402
from tests.unit.iluvatar_mr_hgemm_test_common import (  # noqa: E402
    STAGED_BRICK_M,
    STAGED_BRICK_N,
    brick_k_from_k_rep,
    expected_warp00_atom_gemm,
    multibrick_position_tensor,
    remap_hgemm_tensors_for_pattern,
)
from tests.unit.iluvatar_mr_staged_kernels import build_mr_g2s_s2r_mma_warp00_launch  # noqa: E402

_G2S_S2R_MMA_PATTERNS = ("nt", "nn", "tn", "tt")
_G2S_S2R_MMA_K_REP = (2,)


def _require_enabled() -> None:
    if os.environ.get("FLYDSL_ILUVATAR_RUN_MR_S2R_MMA", "").lower() not in {"1", "true", "yes", "on"}:
        pytest.skip("set FLYDSL_ILUVATAR_RUN_MR_S2R_MMA=1 to run Iluvatar MR G2S->S2R->MMA tests")


def _require_imports():
    try:
        import flydsl

        generated_pkg = Path(os.environ.get("FLYDSL_PYTHON_PACKAGES", _REPO_ROOT / "build-fly" / "python_packages"))
        generated_flydsl = generated_pkg / "flydsl"
        if generated_flydsl.is_dir() and str(generated_flydsl) not in flydsl.__path__:
            flydsl.__path__.append(str(generated_flydsl))
    except ModuleNotFoundError as exc:
        pytest.fail(f"FlyDSL Python package is not importable: {exc}")


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        pytest.skip(f"torch is required for Iluvatar MR G2S->S2R->MMA device tests: {exc}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA-compatible Iluvatar device is not available")
    return torch


def _configure_iluvatar_env(monkeypatch) -> None:
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)


@pytest.mark.parametrize("k_rep", _G2S_S2R_MMA_K_REP)
@pytest.mark.parametrize("major_pattern", _G2S_S2R_MMA_PATTERNS)
def test_iluvatar_mr_g2s_s2r_mma_warp00_atom_device(major_pattern, k_rep, monkeypatch):
    """G2S -> production S2R copy -> MMA for warp-00 top-left 16x16 atom."""

    _require_enabled()
    _require_imports()
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    brick_k = brick_k_from_k_rep(k_rep)
    launch, _ = build_mr_g2s_s2r_mma_warp00_launch(major_pattern=major_pattern, k_rep=k_rep)

    A_logical = multibrick_position_tensor(torch, (STAGED_BRICK_M, brick_k), torch.float16)
    B_logical = multibrick_position_tensor(torch, (STAGED_BRICK_N, brick_k), torch.float16)
    B_logical = B_logical + torch.tensor(17.0, device="cuda", dtype=torch.float16)
    A_dev, B_dev = remap_hgemm_tensors_for_pattern(A_logical, B_logical, major_pattern)

    C_out = torch.zeros((ATOM_M, ATOM_N), device="cuda", dtype=torch.float32)
    launch(A_dev, B_dev, C_out)
    torch.cuda.synchronize()

    expected = expected_warp00_atom_gemm(A_logical, B_logical, brick_k=brick_k)
    torch.testing.assert_close(
        C_out,
        expected,
        rtol=2e-2,
        atol=2e-2,
        msg=f"{major_pattern} k_rep={k_rep} G2S->S2R->MMA warp-00 atom mismatch",
    )
