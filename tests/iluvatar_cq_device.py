# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Helpers for CQ (ivcore30) device tests.

Iluvatar CI currently runs on MR (ivcore11) cards. CQ HGEMM / SmexMtx device
cases must skip when CUDA is missing or the visible GPU is not CQ.
"""

from __future__ import annotations

import pytest

# FIXME: replace this name/smem heuristic with a proper runtime arch query
# (e.g. ivcore30 vs ivcore11) once the Iluvatar torch/CUDA stack exposes one.
# CQ (ivcore30) SLB is 192 KiB; MR (ivcore11) is 128 KiB.
_CQ_SMEM_BYTES = 196608
# Device-name tokens observed on CQ (e.g. "Iluvatar TG-V300X-72-A2").
_CQ_NAME_TOKENS = ("V300", "TG-V300", "IVCORE30", "CONQUEROR")


def is_iluvatar_cq_cuda_device(torch) -> bool:
    """Return True if ``cuda:0`` looks like an Iluvatar CQ (ivcore30) GPU."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        return False
    try:
        props = torch.cuda.get_device_properties(0)
    except (AssertionError, RuntimeError):
        return False
    name = (getattr(props, "name", None) or "").upper()
    if any(tok in name for tok in _CQ_NAME_TOKENS):
        return True
    smem = int(getattr(props, "shared_memory_per_block", 0) or 0)
    return smem >= _CQ_SMEM_BYTES


def require_iluvatar_cq_torch():
    """Import torch and skip unless a CQ device is visible."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        pytest.skip(f"torch is required for Iluvatar CQ device tests: {exc}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        pytest.skip("CUDA-compatible Iluvatar device is not available")
    if not is_iluvatar_cq_cuda_device(torch):
        try:
            props = torch.cuda.get_device_properties(0)
            name = getattr(props, "name", "<unknown>")
            smem = getattr(props, "shared_memory_per_block", None)
        except (AssertionError, RuntimeError):
            name, smem = "<unavailable>", None
        pytest.skip(
            f"Iluvatar CQ (ivcore30) device is not available "
            f"(cuda:0 name={name!r}, shared_memory_per_block={smem}); "
            f"CQ HGEMM device tests are skipped on MR / non-CQ hosts"
        )
    return torch
