# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Opt-in MR async copy correctness tests for Iluvatar.

This file is a parameterized scaffold for:
  - data widths: b8 / b16 / b32
  - major order: row / col

At the moment only b32 row-major is enabled for execution. The remaining
combinations are explicit placeholders and are skipped until shared-memory
swizzle and related paths are completed.
"""

import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l2_device]

_REPO_ROOT = Path(__file__).resolve().parents[2]

GMEM_M = 128
GMEM_N = 128
PADDED_PITCH_ELEMS = 192
TILE_M = 16
SENTINEL = 37

SUBTILE_CASES = {
    "three_tiles": ((0, 0), (32, 32), (96, 64)),
}
MAX_TILES = max(len(coords) for coords in SUBTILE_CASES.values())

_PENDING_SWIZZLE = "placeholder: enable after shared-memory swizzle path is implemented"

ATOM_CASES = [
    pytest.param(
        {"elem_bits": 8, "tile_n": 64, "copy_op_ctor": "AsyncCopy16x64B8Row"},
        id="b8_row",
        marks=pytest.mark.skip(reason=_PENDING_SWIZZLE),
    ),
    pytest.param(
        {"elem_bits": 8, "tile_n": 64, "copy_op_ctor": "AsyncCopy16x64B8Col"},
        id="b8_col",
        marks=pytest.mark.skip(reason=_PENDING_SWIZZLE),
    ),
    pytest.param(
        {"elem_bits": 16, "tile_n": 32, "copy_op_ctor": "AsyncCopy16x32B16Row"},
        id="b16_row",
        marks=pytest.mark.skip(reason=_PENDING_SWIZZLE),
    ),
    pytest.param(
        {"elem_bits": 16, "tile_n": 32, "copy_op_ctor": "AsyncCopy16x32B16Col"},
        id="b16_col",
        marks=pytest.mark.skip(reason=_PENDING_SWIZZLE),
    ),
    pytest.param({"elem_bits": 32, "tile_n": 16, "copy_op_ctor": "AsyncCopy16x16B32Row"}, id="b32_row"),
    pytest.param(
        {"elem_bits": 32, "tile_n": 16, "copy_op_ctor": "AsyncCopy16x16B32Col"},
        id="b32_col",
        marks=pytest.mark.skip(reason=_PENDING_SWIZZLE),
    ),
]


def _require_enabled() -> None:
    if os.environ.get("FLYDSL_ILUVATAR_RUN_MR_ASYNC_COPY", "").lower() not in {"1", "true", "yes", "on"}:
        pytest.skip("set FLYDSL_ILUVATAR_RUN_MR_ASYNC_COPY=1 to run Iluvatar MR async copy tests")


def _require_imports():
    try:
        import flydsl

        generated_pkg = Path(os.environ.get("FLYDSL_PYTHON_PACKAGES", _REPO_ROOT / "build-fly" / "python_packages"))
        generated_flydsl = generated_pkg / "flydsl"
        if generated_flydsl.is_dir() and str(generated_flydsl) not in flydsl.__path__:
            flydsl.__path__.append(str(generated_flydsl))

        import flydsl.compiler as flyc
        import flydsl.expr as fx
        import flydsl.expr.iluvatar as ix
    except ModuleNotFoundError as exc:
        pytest.fail(f"FlyDSL Python package is not importable: {exc}")
    return flyc, fx, ix


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        pytest.skip(f"torch is required for Iluvatar MR async copy tests: {exc}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA-compatible device is not available")
    return torch


def _build_expected(a, tile_coords, out_rows, tile_n):
    expected = a.new_full((out_rows, tile_n), fill_value=SENTINEL)
    for tile_idx, (tile_row, tile_col) in enumerate(tile_coords):
        dst_row = tile_idx * TILE_M
        expected[dst_row : dst_row + TILE_M, :] = a[tile_row : tile_row + TILE_M, tile_col : tile_col + tile_n]
    return expected


def _torch_dtype_for_bits(torch, elem_bits):
    if elem_bits == 8:
        return torch.int8
    if elem_bits == 16:
        return torch.float16
    if elem_bits == 32:
        return torch.int32
    raise ValueError(f"unsupported elem_bits={elem_bits}")


def _fx_elem_type_for_bits(fx, elem_bits):
    if elem_bits == 8:
        return fx.Int8
    if elem_bits == 16:
        return fx.Float16
    if elem_bits == 32:
        return fx.Int32
    raise ValueError(f"unsupported elem_bits={elem_bits}")


@pytest.mark.parametrize("atom_case", ATOM_CASES)
def test_iluvatar_mr_async_copy_subtile_gather(monkeypatch, atom_case):
    """Copy selected sub-tiles from a padded source matrix and verify gather output.

    Launch uses one full warp (64 threads). G2S async copy is warp-cooperative,
    and S2G write-back is lane-partitioned so each lane stores a disjoint
    linear segment. Source pitch is passed at runtime via ``stride_byte``.
    """

    _require_enabled()
    flyc, fx, ix = _require_imports()
    torch = _require_torch()
    elem_bits = atom_case["elem_bits"]
    tile_n = atom_case["tile_n"]
    elem_bytes = elem_bits // 8
    tile_bits = TILE_M * tile_n * elem_bits
    smem_bytes = TILE_M * tile_n * elem_bytes
    tile_elems = TILE_M * tile_n
    assert tile_elems % 64 == 0
    elems_per_lane = tile_elems // 64
    copy_op = getattr(ix, atom_case["copy_op_ctor"])
    torch_dtype = _torch_dtype_for_bits(torch, elem_bits)
    fx_elem_type = _fx_elem_type_for_bits(fx, elem_bits)

    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)

    @flyc.kernel
    def _g2s_s2g_kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        src_offset: fx.Int32,
        dst_offset: fx.Int32,
        src_pitch_elems: fx.Int32,
        src_pitch_bytes: fx.Int32,
    ):
        tid = fx.thread_idx.x
        g_src_layout = fx.make_layout((TILE_M, tile_n), (src_pitch_elems, 1))
        s_layout = fx.make_layout((TILE_M, tile_n), (tile_n, 1))
        a_base = fx.get_iter(A)
        b_base = fx.get_iter(B)
        s_base = fx.get_dyn_shared(fx_elem_type)
        sA = fx.make_view(s_base, s_layout)

        g2s = fx.make_copy_atom(copy_op(), tile_bits)
        g2s = g2s.set_value("stride_byte", src_pitch_bytes)

        gA = fx.make_view(fx.add_offset(a_base, fx.make_int_tuple(src_offset)), g_src_layout)
        gB_base = fx.add_offset(b_base, fx.make_int_tuple(dst_offset))
        fx.copy(g2s, gA, sA)
        ix.cp_async_commit_group()
        ix.cp_async_wait_group(0)

        lane_base = tid * elems_per_lane
        for i in fx.range_constexpr(elems_per_lane):
            elem_idx = lane_base + i
            src_ptr = fx.add_offset(s_base, fx.make_int_tuple(elem_idx))
            dst_ptr = fx.add_offset(gB_base, fx.make_int_tuple(elem_idx))
            fx.ptr_store(fx.ptr_load(src_ptr), dst_ptr)

    @flyc.jit
    def _launch(
        A: fx.Tensor,
        B: fx.Tensor,
        src_offset: fx.Int32,
        dst_offset: fx.Int32,
        src_pitch_elems: fx.Int32,
        src_pitch_bytes: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        _g2s_s2g_kernel(A, B, src_offset, dst_offset, src_pitch_elems, src_pitch_bytes).launch(
            grid=(1, 1, 1), block=(64, 1, 1), smem=smem_bytes, stream=stream
        )

    A_storage = torch.arange(GMEM_M * PADDED_PITCH_ELEMS, dtype=torch_dtype, device="cuda").reshape(
        GMEM_M, PADDED_PITCH_ELEMS
    )
    A = A_storage[:, :GMEM_N]
    src_pitch_elems = int(A.stride(0))
    src_pitch_bytes = src_pitch_elems * elem_bytes
    assert src_pitch_elems == PADDED_PITCH_ELEMS
    assert src_pitch_bytes % 64 == 0
    stream = torch.cuda.Stream()

    for case_name, tile_coords in SUBTILE_CASES.items():
        B = torch.full((MAX_TILES * TILE_M, tile_n), fill_value=SENTINEL, dtype=torch_dtype, device="cuda")
        for tile_idx, (tile_row, tile_col) in enumerate(tile_coords):
            assert tile_row % TILE_M == 0 and tile_col % tile_n == 0
            assert tile_row + TILE_M <= GMEM_M and tile_col + tile_n <= GMEM_N
            src_offset = tile_row * src_pitch_elems + tile_col
            dst_offset = tile_idx * TILE_M * tile_n
            _launch(
                A,
                B,
                src_offset=src_offset,
                dst_offset=dst_offset,
                src_pitch_elems=src_pitch_elems,
                src_pitch_bytes=src_pitch_bytes,
                stream=stream,
            )

        torch.cuda.synchronize()

        expected = _build_expected(A, tile_coords, out_rows=MAX_TILES * TILE_M, tile_n=tile_n)
        if not torch.equal(expected, B):
            if expected.is_floating_point():
                max_abs_diff = (expected.to(torch.float64) - B.to(torch.float64)).abs().max().item()
            else:
                max_abs_diff = (expected.to(torch.int64) - B.to(torch.int64)).abs().max().item()
            pytest.fail(f"{case_name} failed: max_abs_diff={max_abs_diff}")
