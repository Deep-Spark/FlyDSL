# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar MoE token sorting V1 device tests (fp32 weights / i32 ids).

Reference and expected output layout match the ROCm/CK convention:

- ``sorted_ids``: packed ``(topk_pos << 24) | token_id``; sentinel
  ``(topk << 24) | M`` fills the padding slots inside each expert block.
- ``sorted_weights``: fp32, padding slots filled with ``0.0``.
- ``sorted_expert_ids``: one entry per ``unit_size`` block.
- ``num_valid_ids = [total_padded, M]``.

V1 output is fully deterministic (no atomics inside the kernel) so all four
outputs are compared **bit-exact** against a hand-computed CPU reference.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.iluvatar_lower]

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA-compatible Iluvatar device is not available", allow_module_level=True)

from kernels.moe.iluvatar.moe_sorting_kernel import (  # noqa: E402
    DEFAULT_UNIT_SIZE,
    compile_iluvatar_moe_sorting,
)


def _configure_iluvatar_env(monkeypatch) -> None:
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)


def _reference_moe_sorting(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    unit_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """CPU/torch reference matching the CK packed-ID layout (V1: no EP mode).

    Returns ``(sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids)``
    with the exact same numeric outputs the Iluvatar kernel should produce
    (deterministic order per expert).
    """
    device = topk_ids.device
    M, topk = int(topk_ids.shape[0]), int(topk_ids.shape[1])
    max_padded = M * topk + num_experts * unit_size - topk
    max_blocks = (max_padded + unit_size - 1) // unit_size

    sentinel = (topk << 24) | M
    sorted_ids = torch.full((max_padded,), sentinel, dtype=torch.int32, device=device)
    sorted_weights = torch.zeros((max_padded,), dtype=torch.float32, device=device)
    sorted_expert_ids = torch.full((max_blocks,), -1, dtype=torch.int32, device=device)
    num_valid_ids = torch.zeros(2, dtype=torch.int32, device=device)

    ids_cursor = 0
    expert_ids_cursor = 0
    for eid in range(num_experts):
        token_id, topk_pos = torch.where(topk_ids == eid)
        count = int(token_id.numel())
        if count == 0:
            continue
        num_blocks = (count + unit_size - 1) // unit_size
        padded = num_blocks * unit_size
        packed = (topk_pos.to(torch.int32) << 24) | token_id.to(torch.int32)
        sorted_ids[ids_cursor : ids_cursor + count] = packed
        sorted_weights[ids_cursor : ids_cursor + count] = topk_weights[token_id, topk_pos]
        ids_cursor += padded
        sorted_expert_ids[expert_ids_cursor : expert_ids_cursor + num_blocks] = eid
        expert_ids_cursor += num_blocks

    num_valid_ids[0] = ids_cursor
    num_valid_ids[1] = M
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids


def _generate_inputs(M: int, num_experts: int, topk: int, *, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Random ``topk_ids`` with unique expert per token, plus random ``topk_weights``."""
    assert topk <= num_experts, f"topk={topk} must be <= E={num_experts}"
    g = torch.Generator(device="cuda").manual_seed(seed)
    if M == 0:
        topk_ids = torch.zeros((0, topk), dtype=torch.int32, device="cuda")
    else:
        rows = []
        for _ in range(M):
            perm = torch.randperm(num_experts, generator=g, device="cuda")[:topk]
            rows.append(perm.to(torch.int32))
        topk_ids = torch.stack(rows, dim=0)
    topk_weights = torch.rand((M, topk), generator=g, device="cuda", dtype=torch.float32)
    return topk_ids, topk_weights


def _alloc_outputs(
    M: int, num_experts: int, topk: int, unit_size: int, device: str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_padded = M * topk + num_experts * unit_size - topk
    max_blocks = (max_padded + unit_size - 1) // unit_size
    sorted_ids = torch.empty((max_padded,), dtype=torch.int32, device=device)
    sorted_weights = torch.empty((max_padded,), dtype=torch.float32, device=device)
    sorted_expert_ids = torch.empty((max_blocks,), dtype=torch.int32, device=device)
    num_valid_ids = torch.empty((2,), dtype=torch.int32, device=device)
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids


def _run_kernel(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    topk: int,
    unit_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    launch = compile_iluvatar_moe_sorting(num_experts=num_experts, topk=topk, unit_size=unit_size)
    M = int(topk_ids.shape[0])
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids = _alloc_outputs(M, num_experts, topk, unit_size)
    ret = launch(
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
    )
    torch.cuda.synchronize()
    assert ret == (sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids)
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids


def _assert_bit_exact(
    got: tuple[torch.Tensor, ...],
    ref: tuple[torch.Tensor, ...],
    *,
    num_experts: int,
    topk: int,
    unit_size: int,
) -> None:
    """Compare all four outputs on the *valid* range and *valid* expert blocks.

    Regions past ``num_valid_ids[0]`` in ``sorted_ids`` / ``sorted_weights`` and
    past ``ceil(num_valid_ids[0]/unit_size)`` in ``sorted_expert_ids`` are
    left uninitialised by the kernel (caller-allocated with ``torch.empty``),
    so we cannot assert on them.
    """
    got_ids, got_w, got_eids, got_nv = got
    ref_ids, ref_w, ref_eids, ref_nv = ref

    assert torch.equal(got_nv, ref_nv), f"num_valid_ids mismatch: got={got_nv.tolist()} ref={ref_nv.tolist()}"
    num_padded = int(ref_nv[0].item())
    num_blocks = (num_padded + unit_size - 1) // unit_size

    assert torch.equal(
        got_ids[:num_padded], ref_ids[:num_padded]
    ), f"sorted_ids mismatch on valid range [0:{num_padded}]"
    assert torch.equal(
        got_w[:num_padded], ref_w[:num_padded]
    ), f"sorted_weights mismatch on valid range [0:{num_padded}]"
    assert torch.equal(
        got_eids[:num_blocks], ref_eids[:num_blocks]
    ), f"sorted_expert_ids mismatch on valid blocks [0:{num_blocks}]"


@pytest.mark.parametrize(
    "M,E,topk",
    [
        # Decode-size (small M)
        (1, 8, 2),
        (1, 32, 5),
        (1, 256, 8),
        # Small batch
        (4, 32, 5),
        (8, 128, 4),
        (16, 256, 8),
        # Medium batch
        (128, 256, 8),
    ],
)
def test_iluvatar_moe_sorting_forward(monkeypatch, M, E, topk):
    _configure_iluvatar_env(monkeypatch)
    torch.manual_seed(42)
    topk_ids, topk_weights = _generate_inputs(M, E, topk, seed=42 + M * 1000 + E * 10 + topk)
    got = _run_kernel(topk_ids, topk_weights, E, topk, DEFAULT_UNIT_SIZE)
    ref = _reference_moe_sorting(topk_ids, topk_weights, E, DEFAULT_UNIT_SIZE)
    _assert_bit_exact(got, ref, num_experts=E, topk=topk, unit_size=DEFAULT_UNIT_SIZE)


@pytest.mark.large_shape
def test_iluvatar_moe_sorting_large(monkeypatch):
    _configure_iluvatar_env(monkeypatch)
    M, E, topk = 1024, 256, 8
    topk_ids, topk_weights = _generate_inputs(M, E, topk, seed=1024_256_8)
    got = _run_kernel(topk_ids, topk_weights, E, topk, DEFAULT_UNIT_SIZE)
    ref = _reference_moe_sorting(topk_ids, topk_weights, E, DEFAULT_UNIT_SIZE)
    _assert_bit_exact(got, ref, num_experts=E, topk=topk, unit_size=DEFAULT_UNIT_SIZE)


@pytest.mark.parametrize(
    "M,E,topk",
    [
        (2, 8, 8),  # topk == E: every token routed to every expert
        (2, 8, 1),  # topk == 1: each token single expert
        (0, 32, 5),  # M == 0: no work, but num_valid_ids must still be set
    ],
)
def test_iluvatar_moe_sorting_edge_cases(monkeypatch, M, E, topk):
    _configure_iluvatar_env(monkeypatch)
    topk_ids, topk_weights = _generate_inputs(M, E, topk, seed=99 + M * 100 + E + topk)
    got = _run_kernel(topk_ids, topk_weights, E, topk, DEFAULT_UNIT_SIZE)
    ref = _reference_moe_sorting(topk_ids, topk_weights, E, DEFAULT_UNIT_SIZE)
    _assert_bit_exact(got, ref, num_experts=E, topk=topk, unit_size=DEFAULT_UNIT_SIZE)


def test_iluvatar_moe_sorting_compile_time_guards():
    with pytest.raises(ValueError, match="num_experts must be > 0"):
        compile_iluvatar_moe_sorting(num_experts=0, topk=2)
    with pytest.raises(ValueError, match="topk must be > 0"):
        compile_iluvatar_moe_sorting(num_experts=8, topk=0)
    with pytest.raises(ValueError, match="unit_size must be > 0"):
        compile_iluvatar_moe_sorting(num_experts=8, topk=2, unit_size=0)
    with pytest.raises(ValueError, match="topk must be <= num_experts"):
        compile_iluvatar_moe_sorting(num_experts=4, topk=8)
    with pytest.raises(ValueError, match="topk must be < 128"):
        compile_iluvatar_moe_sorting(num_experts=256, topk=128)


def test_iluvatar_moe_sorting_runtime_guards(monkeypatch):
    _configure_iluvatar_env(monkeypatch)
    E, topk = 32, 4
    launch = compile_iluvatar_moe_sorting(num_experts=E, topk=topk)
    M = 4
    unit_size = DEFAULT_UNIT_SIZE
    max_padded = M * topk + E * unit_size - topk
    max_blocks = (max_padded + unit_size - 1) // unit_size

    topk_ids = torch.zeros((M, topk), dtype=torch.int32, device="cuda").contiguous()
    topk_weights = torch.rand((M, topk), dtype=torch.float32, device="cuda").contiguous()
    sorted_ids = torch.empty((max_padded,), dtype=torch.int32, device="cuda")
    sorted_weights = torch.empty((max_padded,), dtype=torch.float32, device="cuda")
    sorted_expert_ids = torch.empty((max_blocks,), dtype=torch.int32, device="cuda")
    num_valid_ids = torch.empty((2,), dtype=torch.int32, device="cuda")

    # ---- Wrong dtype -------------------------------------------------
    bad_ids_dtype = topk_ids.to(torch.int64)
    with pytest.raises(ValueError, match="topk_ids dtype must be torch.int32"):
        launch(bad_ids_dtype, topk_weights, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids)

    bad_weights_dtype = topk_weights.to(torch.float16)
    with pytest.raises(ValueError, match="topk_weights dtype must be torch.float32"):
        launch(topk_ids, bad_weights_dtype, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids)

    # ---- Shape mismatch ---------------------------------------------
    mismatched_weights = torch.rand((M + 1, topk), dtype=torch.float32, device="cuda").contiguous()
    with pytest.raises(ValueError, match=r"topk_weights shape"):
        launch(topk_ids, mismatched_weights, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids)

    # ---- Wrong compile-time topk vs input ---------------------------
    wrong_topk_ids = torch.zeros((M, topk + 1), dtype=torch.int32, device="cuda").contiguous()
    wrong_topk_weights = torch.rand((M, topk + 1), dtype=torch.float32, device="cuda").contiguous()
    with pytest.raises(ValueError, match=r"topk_ids.shape\[1\] == topk"):
        launch(
            wrong_topk_ids,
            wrong_topk_weights,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
        )

    # ---- Undersized outputs -----------------------------------------
    small_sorted_ids = torch.empty((max_padded - 1,), dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="sorted_ids too small"):
        launch(topk_ids, topk_weights, small_sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids)

    # ---- Non-contiguous input ---------------------------------------
    nc_ids = torch.zeros((topk, M), dtype=torch.int32, device="cuda").t()
    assert tuple(nc_ids.shape) == (M, topk) and not nc_ids.is_contiguous()
    with pytest.raises(ValueError, match="topk_ids must be contiguous"):
        launch(nc_ids, topk_weights, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids)

    # ---- Output overlap with input ----------------------------------
    # Aliasing sorted_ids with topk_ids: share the same storage. Use a large
    # enough buffer so the size check does not trip first.
    big_ids_buf = torch.zeros((max_padded,), dtype=torch.int32, device="cuda")
    alias_topk_ids = big_ids_buf[: M * topk].view(M, topk)
    assert alias_topk_ids.is_contiguous() and alias_topk_ids.data_ptr() == big_ids_buf.data_ptr()
    with pytest.raises(ValueError, match="must not overlap with topk_ids"):
        launch(
            alias_topk_ids,
            topk_weights,
            big_ids_buf,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
        )
