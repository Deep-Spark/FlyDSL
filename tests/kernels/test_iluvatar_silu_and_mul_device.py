# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar BF16 silu-and-mul device tests."""

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

from kernels.moe.iluvatar.silu_and_mul import (  # noqa: E402
    compile_iluvatar_silu_and_mul,
)


def _configure_iluvatar_env(monkeypatch) -> None:
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)


def _make_sorted_ids(token_num: int, topk: int, padding: int):
    packed = [
        (slot << 24) | token
        for token in range(token_num)
        for slot in range(topk)
    ]
    # Deterministic non-identity order exercises sorted-row to original-row mapping.
    packed = packed[1::2] + packed[::2]
    packed.extend([((topk << 24) | token_num)] * padding)
    return torch.tensor(packed, device="cuda", dtype=torch.int32)


def _to_gui_layout(x_separated, inter_dim: int):
    rows = int(x_separated.shape[0])
    gate = x_separated[:, :inter_dim].reshape(rows, inter_dim // 16, 16)
    up = x_separated[:, inter_dim:].reshape(rows, inter_dim // 16, 16)
    return torch.cat((gate, up), dim=2).reshape(rows, 2 * inter_dim).contiguous()


def _reference(
    x_separated,
    *,
    inter_dim: int,
    topk_ids=None,
    bias=None,
    act: str,
    swiglu_limit: float,
):
    gate = x_separated[:, :inter_dim].to(torch.float32)
    linear = x_separated[:, inter_dim:].to(torch.float32)
    if bias is not None:
        row_bias = bias[topk_ids.reshape(-1).to(torch.int64)]
        gate = gate + row_bias[:, :inter_dim]
        linear = linear + row_bias[:, inter_dim:]

    if act == "swiglu":
        limit = swiglu_limit if swiglu_limit != 0.0 else 7.0
        gate = gate.clamp(max=limit)
        linear = linear.clamp(min=-limit, max=limit)
        result = gate * torch.sigmoid(1.702 * gate) * (linear + 1.0)
    elif swiglu_limit != 0.0:
        gate = gate.clamp(max=swiglu_limit)
        linear = linear.clamp(min=-swiglu_limit, max=swiglu_limit)
        result = gate * torch.sigmoid(1.702 * gate) * linear
    else:
        result = torch.nn.functional.silu(gate) * linear
    return result.to(torch.bfloat16)


@pytest.mark.parametrize(
    "inter_dim,token_num,topk,gui_layout,enable_bias,act,swiglu_limit",
    (
        (256, 5, 2, False, False, "silu", 0.0),
        (512, 3, 4, True, False, "silu", 0.0),
        (256, 4, 2, False, True, "silu", 0.0),
        (256, 3, 2, True, True, "swiglu", 5.0),
    ),
)
def test_iluvatar_silu_and_mul_correctness(
    inter_dim,
    token_num,
    topk,
    gui_layout,
    enable_bias,
    act,
    swiglu_limit,
    monkeypatch,
):
    _configure_iluvatar_env(monkeypatch)
    torch.manual_seed(42)

    rows = token_num * topk
    x_separated = (
        (torch.rand(rows, 2 * inter_dim, device="cuda", dtype=torch.float32) * 6.0) - 3.0
    ).to(torch.bfloat16)
    x = _to_gui_layout(x_separated, inter_dim) if gui_layout else x_separated.contiguous()

    experts = 7
    if enable_bias:
        topk_ids = torch.randint(
            0,
            experts,
            (token_num, topk),
            device="cuda",
            dtype=torch.int32,
        ).contiguous()
        bias = (
            torch.rand(experts, 2 * inter_dim, device="cuda", dtype=torch.float32) - 0.5
        ).contiguous()
    else:
        topk_ids = None
        bias = None

    padding = 3
    sorted_ids = _make_sorted_ids(token_num, topk, padding)
    num_valid_ids = torch.tensor(
        [int(sorted_ids.numel())],
        device="cuda",
        dtype=torch.int32,
    )
    out = torch.full(
        (rows, inter_dim),
        float("nan"),
        device="cuda",
        dtype=torch.bfloat16,
    )
    out_scale = torch.empty(0, device="cuda", dtype=torch.uint8)

    launch = compile_iluvatar_silu_and_mul(
        inter_dim=inter_dim,
        topk=topk,
        quant_mode="none",
        gui_layout=gui_layout,
        act=act,
        enable_bias=enable_bias,
        swiglu_limit=swiglu_limit,
    )
    stream = torch.cuda.Stream()
    ret = launch(
        x,
        out,
        out_scale,
        sorted_ids,
        num_valid_ids,
        topk_ids,
        bias,
        token_num,
        int(sorted_ids.numel()),
        stream=stream,
    )
    assert ret is out
    stream.synchronize()

    expected = _reference(
        x_separated,
        inter_dim=inter_dim,
        topk_ids=topk_ids,
        bias=bias,
        act=act,
        swiglu_limit=swiglu_limit,
    )
    torch.testing.assert_close(
        out.to(torch.float32),
        expected.to(torch.float32),
        rtol=5e-2,
        atol=5e-2,
        msg=(
            f"Iluvatar silu_and_mul mismatch: N={inter_dim}, topk={topk}, "
            f"gui_layout={gui_layout}, bias={enable_bias}, act={act}"
        ),
    )


def test_iluvatar_silu_and_mul_zero_rows(monkeypatch):
    _configure_iluvatar_env(monkeypatch)

    inter_dim = 256
    launch = compile_iluvatar_silu_and_mul(inter_dim=inter_dim, topk=2)
    x = torch.empty((0, 2 * inter_dim), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((0, inter_dim), device="cuda", dtype=torch.bfloat16)
    out_scale = torch.empty(0, device="cuda", dtype=torch.uint8)
    sorted_ids = torch.empty(0, device="cuda", dtype=torch.int32)
    num_valid_ids = torch.zeros(1, device="cuda", dtype=torch.int32)

    ret = launch(
        x,
        out,
        out_scale,
        sorted_ids,
        num_valid_ids,
        None,
        None,
        0,
        0,
    )
    assert ret is out
    assert out.numel() == 0


def test_iluvatar_silu_and_mul_compile_time_guards():
    with pytest.raises(ValueError, match="inter_dim must be positive"):
        compile_iluvatar_silu_and_mul(inter_dim=0, topk=2)
    with pytest.raises(ValueError, match="must be divisible by 32"):
        compile_iluvatar_silu_and_mul(inter_dim=33, topk=2)
    with pytest.raises(ValueError, match=r"topk must be in \[1, 255\]"):
        compile_iluvatar_silu_and_mul(inter_dim=256, topk=0)
    with pytest.raises(ValueError, match="supports only quant_mode='none'"):
        compile_iluvatar_silu_and_mul(
            inter_dim=256,
            topk=2,
            quant_mode="fp4",
        )
    with pytest.raises(ValueError, match="act must be one of"):
        compile_iluvatar_silu_and_mul(
            inter_dim=256,
            topk=2,
            act="relu",
        )


def test_iluvatar_silu_and_mul_runtime_guards(monkeypatch):
    _configure_iluvatar_env(monkeypatch)

    token_num, topk, inter_dim = 2, 2, 256
    rows = token_num * topk
    launch = compile_iluvatar_silu_and_mul(inter_dim=inter_dim, topk=topk)
    x = torch.randn(
        rows,
        2 * inter_dim,
        device="cuda",
        dtype=torch.bfloat16,
    ).contiguous()
    out = torch.empty(rows, inter_dim, device="cuda", dtype=torch.bfloat16)
    out_scale = torch.empty(0, device="cuda", dtype=torch.uint8)
    sorted_ids = _make_sorted_ids(token_num, topk, 0)
    num_valid_ids = torch.tensor([rows], device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError, match="expected x shape"):
        launch(
            x,
            out,
            out_scale,
            sorted_ids,
            num_valid_ids,
            None,
            None,
            token_num + 1,
            rows,
        )

    out_f32 = torch.empty(rows, inter_dim, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match=r"out dtype must be torch\.bfloat16"):
        launch(
            x,
            out_f32,
            out_scale,
            sorted_ids,
            num_valid_ids,
            None,
            None,
            token_num,
            rows,
        )

    out_alias = x.view(-1)[: rows * inter_dim].view(rows, inter_dim)
    assert out_alias.is_contiguous()
    with pytest.raises(ValueError, match="out must not overlap with x"):
        launch(
            x,
            out_alias,
            out_scale,
            sorted_ids,
            num_valid_ids,
            None,
            None,
            token_num,
            rows,
        )
