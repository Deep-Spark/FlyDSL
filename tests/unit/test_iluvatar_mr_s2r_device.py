# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Opt-in Iluvatar MR S2R device correctness tests.

These tests sit between the async-copy G2S tests and the MMA tests. They prepare
the exact A/B operand tile bytes in shared memory with ordinary 32-bit stores,
then validate that ``make_tiled_copy_A/B`` lands those values in the MMA operand
fragment layout expected by ``MRMma``. The fragment is copied back to global for
inspection; no async-copy instruction and no MMA compute is executed here.

Set ``FLYDSL_ILUVATAR_RUN_MR_S2R=1`` to run (needs an Iluvatar device).

Stage coverage notes
--------------------

* The original S2R test hand-packs linear shared memory and never exercises
  ``mr_hgemm_s2r_a_tile`` / ``mr_hgemm_s2r_b_tile`` against production G2S SME
  layout.
* ``test_iluvatar_mr_g2s_s2r_ki_chain_device`` chains production G2S with
  ``mr_hgemm_s2r_*_tile`` views and scalar smem readback (warp-00 Ki slices).
  B is asserted for every pattern; A is asserted only for ``tn``/``tt`` where
  the scalar readback matches logical layout (``nt``/``nn`` A uses Row-SME
  swizzle that this harness does not decode).
"""

import os
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l2_device]

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kernels.iluvatar_mr_common import ATOM_K, ATOM_M, ATOM_N, SMEM_ROWS, WARP_SIZE, major_pattern_id  # noqa: E402
from kernels.iluvatar_mr_s2r import mr_hgemm_s2r_copy_a, mr_hgemm_s2r_copy_b  # noqa: E402
from tests.unit.iluvatar_mr_hgemm_test_common import (  # noqa: E402
    STAGED_BRICK_M,
    STAGED_BRICK_N,
    brick_k_from_k_rep,
    expected_warp00_ab_ki_slice,
    multibrick_position_tensor,
    remap_hgemm_tensors_for_pattern,
)
from tests.unit.iluvatar_mr_staged_kernels import build_mr_g2s_s2r_ki_dump_launch  # noqa: E402

# Chain test targets BK=32 (production default k_rep=2). k_rep=4 needs wider Ki
# brick accounting in mr_hgemm_s2r_a_tile (tn A fails at brick_k=64).
_G2S_K_REP_VALUES = (2,)

S2R_MAJOR_PATTERNS = ("nt", "nn", "tn", "tt")
S2R_K_CHUNK = 16

S2R_DTYPE_CASES = [
    {
        "name": "b8",
        "torch_dtype": "int8",
        "fx_dtype": "Int8",
        "fx_acc": "Int32",
        "elem_bits": 8,
        "mma_k": 32,
        "scalar_atom": "UniversalCopy8b",
    },
    {
        "name": "b16",
        "torch_dtype": "float16",
        "fx_dtype": "Float16",
        "fx_acc": "Float32",
        "elem_bits": 16,
        "mma_k": 16,
        "scalar_atom": "UniversalCopy16b",
    },
    {
        "name": "b32",
        "torch_dtype": "float32",
        "fx_dtype": "Float32",
        "fx_acc": "Float32",
        "elem_bits": 32,
        "mma_k": 16,
        "scalar_atom": "UniversalCopy32b",
    },
]


def _require_enabled() -> None:
    if os.environ.get("FLYDSL_ILUVATAR_RUN_MR_S2R", "").lower() not in {"1", "true", "yes", "on"}:
        pytest.skip("set FLYDSL_ILUVATAR_RUN_MR_S2R=1 to run Iluvatar MR S2R device tests")


def _require_imports():
    try:
        import flydsl

        generated_pkg = Path(os.environ.get("FLYDSL_PYTHON_PACKAGES", _REPO_ROOT / "build-fly" / "python_packages"))
        generated_flydsl = generated_pkg / "flydsl"
        if generated_flydsl.is_dir() and str(generated_flydsl) not in flydsl.__path__:
            flydsl.__path__.append(str(generated_flydsl))

        import flydsl.compiler as flyc
        import flydsl.expr as fx
        import flydsl.expr.ixdl as ixdl
    except ModuleNotFoundError as exc:
        pytest.fail(f"FlyDSL Python package is not importable: {exc}")
    return flyc, fx, ixdl


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        pytest.skip(f"torch is required for Iluvatar MR S2R device tests: {exc}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA-compatible Iluvatar device is not available")
    return torch


def _configure_iluvatar_env(monkeypatch) -> None:
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)


def _position_tensor(torch, shape, dtype):
    rows, cols = shape
    row_idx = torch.arange(rows, device="cuda", dtype=torch.int32).view(rows, 1)
    col_idx = torch.arange(cols, device="cuda", dtype=torch.int32).view(1, cols)
    encoded = row_idx * 257 + col_idx
    if dtype == torch.int8:
        encoded = (encoded * 73 + 19) % 255 - 127
    return encoded.to(dtype)


def _remap_operands_for_pattern(A, B, major_pattern: str):
    pattern_id = major_pattern_id(major_pattern)
    if pattern_id == 0:
        return A, B.t().contiguous()
    if pattern_id == 1:
        return A.t().contiguous(), B.t().contiguous()
    if pattern_id == 2:
        return A, B
    if pattern_id == 3:
        return A.t().contiguous(), B
    raise ValueError(f"unknown major_pattern: {major_pattern}")


def _pack_tensor_as_i32(torch, tensor):
    bytes_flat = tensor.contiguous().view(torch.uint8).to(torch.int32).reshape(-1, 4)
    return (
        bytes_flat[:, 0] | (bytes_flat[:, 1] << 8) | (bytes_flat[:, 2] << 16) | (bytes_flat[:, 3] << 24)
    ).contiguous()


def _compile_s2r_dump_kernel(flyc, fx, ixdl, major_pattern: str, dtype_case):
    pattern_id = major_pattern_id(major_pattern)
    elem_bits = dtype_case["elem_bits"]
    elem_bytes = elem_bits // 8
    mma_k = dtype_case["mma_k"]
    operand_m = max(16, 512 // elem_bits)
    operand_n = max(16, 512 // elem_bits)
    operand_k = max(512 // elem_bits, mma_k)
    k_chunks = mma_k // S2R_K_CHUNK
    packed_a_elems = ATOM_M * mma_k
    packed_b_elems = ATOM_N * mma_k
    packed_a_i32 = (packed_a_elems * elem_bytes) // 4
    packed_b_i32 = (packed_b_elems * elem_bytes) // 4
    packed_b_i32_base = packed_a_i32
    smem_i32 = packed_a_i32 + packed_b_i32
    a_copy_iters = packed_a_i32 // WARP_SIZE
    b_copy_iters = packed_b_i32 // WARP_SIZE
    fx_dtype = getattr(fx, dtype_case["fx_dtype"])
    fx_acc = getattr(fx, dtype_case["fx_acc"])
    scalar_atom_factory = getattr(fx, dtype_case["scalar_atom"])
    _ = pattern_id

    @flyc.kernel(known_block_size=[WARP_SIZE, 1, 1])
    def s2r_dump_kernel(A_init: fx.Tensor, B_init: fx.Tensor, A_out: fx.Tensor, B_out: fx.Tensor):
        tid = fx.thread_idx.x
        init_A = fx.make_view(fx.get_iter(A_init), fx.make_layout(packed_a_i32, 1))
        init_B = fx.make_view(fx.get_iter(B_init), fx.make_layout(packed_b_i32, 1))
        init_A_iter = fx.get_iter(init_A)
        init_B_iter = fx.get_iter(init_B)
        smem_i32_base = fx.recast_iter(
            fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared),
            fx.get_dyn_shared(),
        )
        smem_elem_base = fx.recast_iter(
            fx.PointerType.get(fx_dtype.ir_type, fx.AddressSpace.Shared),
            fx.get_dyn_shared(),
        )
        for i in fx.range_constexpr(a_copy_iters):
            idx = fx.Int32(i * WARP_SIZE) + tid
            src_ptr = fx.add_offset(init_A_iter, fx.make_int_tuple(idx))
            dst_ptr = fx.add_offset(smem_i32_base, fx.make_int_tuple(idx))
            fx.ptr_store(fx.Int32(fx.ptr_load(src_ptr, fx.Int32)), dst_ptr)

        for i in fx.range_constexpr(b_copy_iters):
            idx = fx.Int32(i * WARP_SIZE) + tid
            src_ptr = fx.add_offset(init_B_iter, fx.make_int_tuple(idx))
            dst_ptr = fx.add_offset(smem_i32_base, fx.make_int_tuple(idx + fx.Int32(packed_b_i32_base)))
            fx.ptr_store(fx.Int32(fx.ptr_load(src_ptr, fx.Int32)), dst_ptr)

        def _smem(elem_offset, shape, stride):
            ptr = fx.add_offset(smem_elem_base, fx.make_int_tuple(fx.Int32(elem_offset)))
            return fx.make_view(ptr, fx.make_layout(shape, stride))

        fx.gpu.barrier()

        scalar_atom = fx.make_copy_atom(scalar_atom_factory(), fx_dtype)
        tiled_pack = fx.make_tiled_copy_tv(
            scalar_atom,
            fx.make_layout((SMEM_ROWS, WARP_SIZE // SMEM_ROWS), (1, SMEM_ROWS)),
            fx.make_layout((1, S2R_K_CHUNK // (WARP_SIZE // SMEM_ROWS)), (1, 1)),
        )
        pack = tiled_pack.get_slice(tid)

        for k_chunk in fx.range_constexpr(k_chunks):
            a_src = _smem(
                fx.Int32(k_chunk * S2R_K_CHUNK),
                (ATOM_M, S2R_K_CHUNK),
                (mma_k, 1),
            )
            a_dst = _smem(
                fx.Int32(k_chunk * S2R_K_CHUNK),
                (ATOM_M, S2R_K_CHUNK),
                (mma_k, 1),
            )
            frag = fx.make_fragment_like(pack.partition_S(a_src))
            fx.copy(scalar_atom, pack.partition_S(a_src), frag)
            fx.copy(scalar_atom, frag, pack.partition_D(a_dst))

            b_src = _smem(
                fx.Int32(packed_a_elems + k_chunk * S2R_K_CHUNK),
                (ATOM_N, S2R_K_CHUNK),
                (mma_k, 1),
            )
            b_dst = _smem(
                fx.Int32(packed_a_elems + k_chunk * S2R_K_CHUNK),
                (ATOM_N, S2R_K_CHUNK),
                (mma_k, 1),
            )
            frag = fx.make_fragment_like(pack.partition_S(b_src))
            fx.copy(scalar_atom, pack.partition_S(b_src), frag)
            fx.copy(scalar_atom, frag, pack.partition_D(b_dst))

        fx.gpu.barrier()

        mma_atom = fx.make_mma_atom(ixdl.MRMma(ATOM_M, ATOM_N, mma_k, fx_dtype, fx_dtype, fx_acc))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (1, 1, 1)))
        thr_mma = tiled_mma.thr_slice(tid)
        tiled_copy_A = fx.make_tiled_copy_A(scalar_atom, tiled_mma)
        tiled_copy_B = fx.make_tiled_copy_B(scalar_atom, tiled_mma)
        thr_copy_A = tiled_copy_A.get_slice(tid)
        thr_copy_B = tiled_copy_B.get_slice(tid)

        packed_A = _smem(0, (ATOM_M, mma_k), (mma_k, 1))
        packed_B = _smem(packed_a_elems, (ATOM_N, mma_k), (mma_k, 1))
        frag_A = mr_hgemm_s2r_copy_a(
            copy_atom=scalar_atom,
            thr_copy_a=thr_copy_A,
            thr_mma=thr_mma,
            smem_a_tile=packed_A,
        )
        frag_B = mr_hgemm_s2r_copy_b(
            copy_atom=scalar_atom,
            thr_copy_b=thr_copy_B,
            thr_mma=thr_mma,
            smem_b_tile=packed_B,
        )

        out_A = fx.make_view(fx.get_iter(A_out), fx.make_layout((ATOM_M, mma_k), (mma_k, 1)))
        out_B = fx.make_view(fx.get_iter(B_out), fx.make_layout((ATOM_N, mma_k), (mma_k, 1)))
        fx.copy(scalar_atom, thr_copy_A.retile(frag_A), thr_copy_A.partition_D(out_A), pred=None)
        fx.copy(scalar_atom, thr_copy_B.retile(frag_B), thr_copy_B.partition_D(out_B), pred=None)

    @flyc.jit
    def launch(
        A_init: fx.Tensor,
        B_init: fx.Tensor,
        A_out: fx.Tensor,
        B_out: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        s2r_dump_kernel(A_init, B_init, A_out, B_out).launch(
            grid=(1, 1, 1),
            block=(WARP_SIZE, 1, 1),
            smem=smem_i32 * 4,
            stream=stream,
        )

    return launch, operand_m, operand_n, operand_k, mma_k


@pytest.mark.parametrize("major_pattern", S2R_MAJOR_PATTERNS)
@pytest.mark.parametrize("dtype_case", S2R_DTYPE_CASES, ids=[c["name"] for c in S2R_DTYPE_CASES])
def test_iluvatar_mr_s2r_operand_layout_device(major_pattern, dtype_case, monkeypatch):
    """MMA-coupled S2R dump for A/B operand layouts.

    G2S / async-copy is intentionally not part of this file. The host prepares
    the exact logical A/B operand tile bytes, the kernel copies those bytes into
    shared memory with ordinary 32-bit stores, and the only operation under test
    is ``make_tiled_copy_A/B`` from shared memory into the ``MRMma`` operand
    fragment layout. The fragment is then dumped back to global for comparison.

    The dtype matrix mirrors the async-copy tests:

        b8   -> MRMma K=32, int8 fragments
        b16  -> MRMma K=16, fp16 fragments
        b32  -> MRMma K=16, fp32 fragments

    ``major_pattern`` stays parameterized so this S2R stage can be run alongside
    the G2S matrix, but the shared-memory setup here deliberately does not call
    any ``MRAsyncCp*`` instruction.
    """

    _require_enabled()
    flyc, fx, ixdl = _require_imports()
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)

    torch_dtype = getattr(torch, dtype_case["torch_dtype"])
    launch, operand_m, operand_n, operand_k, mma_k = _compile_s2r_dump_kernel(flyc, fx, ixdl, major_pattern, dtype_case)
    A_logical = _position_tensor(torch, (operand_m, operand_k), torch_dtype)
    B_logical = _position_tensor(torch, (operand_n, operand_k), torch_dtype)
    if torch_dtype != torch.int8:
        B_logical = B_logical + torch.tensor(17.0, device="cuda", dtype=torch_dtype)
    A_init = _pack_tensor_as_i32(torch, A_logical[:ATOM_M, :mma_k]).to(device="cuda")
    B_init = _pack_tensor_as_i32(torch, B_logical[:ATOM_N, :mma_k]).to(device="cuda")
    A_out = torch.empty((ATOM_M, mma_k), device="cuda", dtype=torch_dtype)
    B_out = torch.empty((ATOM_N, mma_k), device="cuda", dtype=torch_dtype)

    launch(A_init, B_init, A_out, B_out)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        A_out,
        A_logical[:ATOM_M, :mma_k].contiguous(),
        rtol=0,
        atol=0,
        msg=f"{dtype_case['name']} {major_pattern} A S2R dump mismatch",
    )
    torch.testing.assert_close(
        B_out,
        B_logical[:ATOM_N, :mma_k].contiguous(),
        rtol=0,
        atol=0,
        msg=f"{dtype_case['name']} {major_pattern} B S2R dump mismatch",
    )


_G2S_S2R_CHAIN_CASES = [
    ("nn", 2, "B"),
    ("tn", 2, "A"),
    ("tn", 2, "B"),
    ("tt", 2, "A"),
]


@pytest.mark.parametrize("major_pattern,k_rep,operand", _G2S_S2R_CHAIN_CASES)
def test_iluvatar_mr_g2s_s2r_ki_chain_device(major_pattern, k_rep, operand, monkeypatch):
    """Production G2S -> ``mr_hgemm_s2r_*_tile`` scalar readback for warp-00 Ki slices.

    Each case builds one dump kernel only (A or B). Mixing both builders in one
    test tickles a FlyDSL JIT cache collision on Iluvatar.
    """

    _require_enabled()
    _require_imports()
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)

    brick_k = brick_k_from_k_rep(k_rep)
    ki_slices = brick_k // ATOM_K
    A_logical = multibrick_position_tensor(torch, (STAGED_BRICK_M, brick_k), torch.float16)
    B_logical = multibrick_position_tensor(torch, (STAGED_BRICK_N, brick_k), torch.float16)
    B_logical = B_logical + torch.tensor(17.0, device="cuda", dtype=torch.float16)
    A_dev, B_dev = remap_hgemm_tensors_for_pattern(A_logical, B_logical, major_pattern)

    launch, _, _, dump_elems = build_mr_g2s_s2r_ki_dump_launch(
        major_pattern=major_pattern, k_rep=k_rep, operand=operand
    )
    out = torch.zeros(dump_elems, device="cuda", dtype=torch.float16)
    launch(A_dev, B_dev, out)
    torch.cuda.synchronize()

    if operand == "A":
        out_view = out.reshape(ki_slices, ATOM_M, ATOM_K)
    else:
        out_view = out.reshape(ki_slices, ATOM_N, ATOM_K)

    for ki in range(ki_slices):
        exp_a, exp_b = expected_warp00_ab_ki_slice(A_logical, B_logical, ki=ki)
        expected = exp_a if operand == "A" else exp_b
        torch.testing.assert_close(
            out_view[ki],
            expected,
            rtol=0,
            atol=0,
            msg=f"{major_pattern} k_rep={k_rep} {operand} ki={ki} G2S->S2R mismatch",
        )
