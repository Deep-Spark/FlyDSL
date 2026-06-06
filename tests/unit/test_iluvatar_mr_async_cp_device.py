# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Opt-in Iluvatar MR SME async-copy device correctness tests.

One case per (dtype, swizzle) SME Load variant: each launches several thread
blocks that copy a position-encoded matrix global -> shared with one
warp-collective SME async copy per tile, wait for completion, then read shared
-> global and check an exact match. Position encoding exposes subtle
physical-layout / swizzle bugs.

Set ``FLYDSL_ILUVATAR_RUN_MR_ASYNC_CP=1`` to run (needs an Iluvatar device).

Stage coverage notes
--------------------

* Original multibrick tests fixed ``K=64`` (BK=64) only; they cannot catch ``k_rep=2``
  (BK=32) brick-count bugs.
* Single-block launches cannot catch multi-CTA ``bid_x`` / ``bid_y`` G2S slice bugs.

Additional tests below add ``BK=32`` and a 2-CTA M-grid G2S dump path.
"""

import os
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l2_device]

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kernels.iluvatar_mr_common import SMEM_ROWS, WARP_SIZE, major_pattern_id  # noqa: E402
from kernels.iluvatar_mr_operand_copy import (  # noqa: E402
    mr_hgemm_g2s_issue_operands,
    mr_pattern_g2s_sme_config,
    mr_sme_shared_view,
)
from tests.unit.iluvatar_mr_hgemm_test_common import (  # noqa: E402
    STAGED_BRICK_K_DEFAULT,
    STAGED_BRICK_M,
    STAGED_BRICK_N,
    STAGED_WARPS_M,
    STAGED_WARPS_N,
    brick_k_from_k_rep,
    expected_multibrick_a_dump,
    expected_multibrick_b_dump,
    multibrick_position_tensor,
    remap_hgemm_tensors_for_pattern,
    staged_cta_config,
)

BLOCKS = 4
THREADS = WARP_SIZE

_MULTIBRICK_DTYPE_CASES = [
    {
        "name": "b8",
        "torch_dtype": "int8",
        "fx_dtype": "Int8",
        "elem_bits": 8,
        "row_atom": "MRAsyncCpRow8b",
        "row_swizzle": "Row8b",
        "scalar_atom": "UniversalCopy8b",
    },
    {
        "name": "b16",
        "torch_dtype": "float16",
        "fx_dtype": "Float16",
        "elem_bits": 16,
        "row_atom": "MRAsyncCpRow16b",
        "row_swizzle": "Row16b",
        "scalar_atom": "UniversalCopy16b",
    },
    {
        "name": "b32",
        "torch_dtype": "float32",
        "fx_dtype": "Float32",
        "elem_bits": 32,
        "row_atom": "MRAsyncCpNoSwizzle",
        "row_swizzle": "NoSwizzle",
        "scalar_atom": "UniversalCopy32b",
    },
]

# Plain-data case specs (resolved to flydsl/torch handles inside the test, after
# the Iluvatar backend env is set). Each tile is one 16 x 512b = 8192-bit SME
# footprint, so tile_n = 8192 / elem_bits. The padded source row stride must keep
# the SME descriptor stride (src_stride_n * elem_bytes) a multiple of 64 bytes,
# otherwise the SME load scrambles data -- hence f16/i8 use 64B-multiple strides.
_CASES = [
    # NoSwizzle 32-bit row path: baseline SME descriptor + shared-layout interpretation.
    {
        "name": "f32_no_swizzle_row",
        "torch_dtype": "float32",
        "fx_dtype": "Float32",
        "elem_bits": 32,
        "swizzle": "NoSwizzle",
        "scalar_atom": "UniversalCopy32b",
        "m": 32,
        "n": 64,
        "src_stride_n": 80,
        "tile_n": 16,
    },
    # Col swizzle with 32-bit values: verifies the colxfb path at f32 width.
    {
        "name": "f32_col_swizzle",
        "torch_dtype": "float32",
        "fx_dtype": "Float32",
        "elem_bits": 32,
        "swizzle": "Col",
        "scalar_atom": "UniversalCopy32b",
        "m": 32,
        "n": 32,
        "src_stride_n": 48,
        "tile_n": 16,
    },
    # Row16b swizzle with fp16: verifies the rowxfb16 physical write/readback layout.
    {
        "name": "f16_row16b_swizzle",
        "torch_dtype": "float16",
        "fx_dtype": "Float16",
        "elem_bits": 16,
        "swizzle": "Row16b",
        "scalar_atom": "UniversalCopy16b",
        "m": 32,
        "n": 64,
        "src_stride_n": 96,  # 96*2B = 192B, multiple of 64B
        "tile_n": 32,
    },
    # Col swizzle with fp16: verifies the colxfb path at the common 16x32 footprint.
    {
        "name": "f16_col_swizzle",
        "torch_dtype": "float16",
        "fx_dtype": "Float16",
        "elem_bits": 16,
        "swizzle": "Col",
        "scalar_atom": "UniversalCopy16b",
        "m": 32,
        "n": 64,
        "src_stride_n": 96,  # 96*2B = 192B, multiple of 64B
        "tile_n": 32,
    },
    # Row8b swizzle with int8: verifies the rowxfb8 ModSwizzle shared layout.
    {
        "name": "i8_row8b_swizzle",
        "torch_dtype": "int8",
        "fx_dtype": "Int8",
        "elem_bits": 8,
        "swizzle": "Row8b",
        "scalar_atom": "UniversalCopy8b",
        "m": 32,
        "n": 128,
        "src_stride_n": 192,  # 192*1B = 192B, multiple of 64B
        "tile_n": 64,
    },
    # Col swizzle with int8: verifies the widest colxfb footprint.
    {
        "name": "i8_col_swizzle",
        "torch_dtype": "int8",
        "fx_dtype": "Int8",
        "elem_bits": 8,
        "swizzle": "Col",
        "scalar_atom": "UniversalCopy8b",
        "m": 32,
        "n": 128,
        "src_stride_n": 192,  # 192*1B = 192B, multiple of 64B
        "tile_n": 64,
    },
]


def _require_enabled() -> None:
    if os.environ.get("FLYDSL_ILUVATAR_RUN_MR_ASYNC_CP", "").lower() not in {"1", "true", "yes", "on"}:
        pytest.skip("set FLYDSL_ILUVATAR_RUN_MR_ASYNC_CP=1 to run the Iluvatar MR async-copy device tests")


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
        pytest.skip(f"torch is required for the Iluvatar MR async-copy device tests: {exc}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA-compatible device is not available")
    return torch


def _configure_iluvatar_env(monkeypatch):
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)


def _position_encoded_tensors(torch, *, dtype, matrix_m, matrix_n, src_stride_n):
    """Create per-position encoded tensors to expose subtle layout bugs."""
    storage = torch.zeros((BLOCKS, matrix_m, src_stride_n), device="cuda", dtype=dtype)

    block_idx = torch.arange(BLOCKS, device="cuda", dtype=torch.int32).view(BLOCKS, 1, 1)
    row_idx = torch.arange(matrix_m, device="cuda", dtype=torch.int32).view(1, matrix_m, 1)
    col_idx = torch.arange(matrix_n, device="cuda", dtype=torch.int32).view(1, 1, matrix_n)

    encoded = block_idx * (matrix_m * matrix_n) + row_idx * matrix_n + col_idx

    # int8 cannot represent all positions uniquely for large matrices; still use
    # a high-variance deterministic mapping to avoid block/tile-wise constants.
    if dtype == torch.int8:
        encoded = (encoded * 73 + 19) % 255 - 127

    values = encoded.to(dtype)
    storage[:, :, :matrix_n] = values

    src = storage[:, :, :matrix_n]
    dst = torch.empty(BLOCKS * matrix_m * matrix_n, device="cuda", dtype=dtype)
    return src, dst, values.contiguous().reshape(-1)


def _compile_multibrick_async_copy_dump_kernel(
    flyc,
    fx,
    ixdl,
    major_pattern: str,
    dtype_case,
    *,
    brick_m: int = STAGED_BRICK_M,
    brick_n: int = STAGED_BRICK_N,
    brick_k: int = STAGED_BRICK_K_DEFAULT,
    use_block_idx_m: bool = False,
):
    pattern_id = major_pattern_id(major_pattern)
    elem_bits = dtype_case["elem_bits"]
    elem_bytes = elem_bits // 8
    fx_dtype = getattr(fx, dtype_case["fx_dtype"])
    scalar_atom_factory = getattr(fx, dtype_case["scalar_atom"])
    row_atom_factory = getattr(ixdl, dtype_case["row_atom"])
    row_swizzle = getattr(ixdl.SMESwizzle, dtype_case["row_swizzle"])
    cta = staged_cta_config(
        major_pattern=major_pattern,
        brick_k=brick_k,
        brick_m=brick_m,
        brick_n=brick_n,
        warps_m=STAGED_WARPS_M,
        warps_n=STAGED_WARPS_N,
        elem_bits=elem_bits,
    )
    values_per_sme_row = cta["values_per_sme_row"]
    threads = cta["threads"]
    a_atoms_total = cta["a_atoms_total"]
    b_atoms_total = cta["b_atoms_total"]
    a_per_warp = cta["a_per_warp"]
    b_per_warp = cta["b_per_warp"]
    smem_elems = cta["smem_elems"]
    b_n_chunks = cta["b_n_chunks"]
    b_logical_stride = cta["b_logical_stride"]
    brick_elems = cta["brick_elems"]
    grid_m = 2 if use_block_idx_m else 1
    a_logical_m = brick_m * grid_m if use_block_idx_m else brick_m
    a_logical_stride = (1, a_logical_m) if pattern_id in (1, 3) else (brick_k, 1)

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def async_copy_dump_kernel(A: fx.Tensor, B: fx.Tensor, A_out: fx.Tensor, B_out: fx.Tensor):
        tid = fx.thread_idx.x
        warp_id = tid // WARP_SIZE
        lane_id = tid % WARP_SIZE
        bid_x, _, _ = fx.block_idx
        a_tile_id = bid_x if use_block_idx_m else fx.Int32(0)
        if fx.const_expr(use_block_idx_m):
            a_out_cta_base = bid_x * fx.Int32(a_atoms_total * brick_elems)
        else:
            a_out_cta_base = fx.Int32(0)

        # The input tensors use pattern-specific host physical layouts. Re-view
        # them as logical A(M,K) / B(N,K) operands before tiling so the rest of
        # the G2S test uses one indexing scheme; only the SME atom/swizzle below
        # remains pattern-specific.
        a_logical_view = fx.make_view(
            fx.get_iter(A),
            fx.make_layout((a_logical_m, brick_k), a_logical_stride),
        )
        b_logical_view = fx.make_view(
            fx.get_iter(B),
            fx.make_layout((brick_n, brick_k), b_logical_stride),
        )
        g_A = fx.slice(fx.flat_divide(a_logical_view, (brick_m, brick_k)), (None, None, a_tile_id, None))
        g_B = fx.slice(fx.flat_divide(b_logical_view, (brick_n, brick_k)), (None, None, 0, None))

        smem_elem_base = fx.recast_iter(
            fx.PointerType.get(fx_dtype.ir_type, fx.AddressSpace.Shared),
            fx.get_dyn_shared(),
        )

        g2s_sme = mr_pattern_g2s_sme_config(
            pattern_id,
            fx_dtype,
            row_atom=row_atom_factory,
            row_swizzle=row_swizzle,
        )
        if fx.const_expr(pattern_id == 1 or pattern_id == 3):
            # Col-A SME leading stride is the full logical M extent (not one CTA bm).
            a_leading = a_logical_m
        else:
            a_leading = brick_k
        if fx.const_expr(pattern_id == 0 or pattern_id == 1):
            b_leading = brick_n
        else:
            b_leading = brick_k

        tile_smem = fx.make_tile(SMEM_ROWS, values_per_sme_row)
        tile_smem_A = (
            fx.make_tile(values_per_sme_row, SMEM_ROWS)
            if fx.const_expr(pattern_id == 1 or pattern_id == 3)
            else tile_smem
        )
        tile_smem_B = (
            fx.make_tile(values_per_sme_row, SMEM_ROWS)
            if fx.const_expr(pattern_id == 0 or pattern_id == 1)
            else tile_smem
        )
        sme_A = ixdl.make_sme_gmem_tensor(g_A[None, None, 0], leading_stride=a_leading)
        sme_B = ixdl.make_sme_gmem_tensor(g_B[None, None, 0], leading_stride=b_leading)
        g_A_div = fx.zipped_divide(sme_A, tile_smem_A)
        g_B_div = fx.zipped_divide(sme_B, tile_smem_B)

        mr_hgemm_g2s_issue_operands(
            pattern_id=pattern_id,
            warp_id=warp_id,
            a_per_warp=a_per_warp,
            b_per_warp=b_per_warp,
            g_A_div=g_A_div,
            g_B_div=g_B_div,
            g2s_sme=g2s_sme,
            smem_base=smem_elem_base,
            elem_dtype=fx_dtype,
            bm=brick_m,
            bn=brick_n,
            bk=brick_k,
            stage_base=fx.Int32(0),
            values_per_sme_row=values_per_sme_row,
        )
        ixdl.cp_async_wait_group(0)
        fx.gpu.barrier()

        scalar_atom = fx.make_copy_atom(scalar_atom_factory(), fx_dtype)
        tiled_st_k = fx.make_tiled_copy_tv(
            scalar_atom,
            fx.make_layout(
                (SMEM_ROWS, WARP_SIZE // SMEM_ROWS),
                (1, SMEM_ROWS),
            ),
            fx.make_layout(
                (1, values_per_sme_row // (WARP_SIZE // SMEM_ROWS)),
                (1, 1),
            ),
        )
        tiled_st_mn = fx.make_tiled_copy_tv(
            scalar_atom,
            fx.make_layout(
                (values_per_sme_row, WARP_SIZE // values_per_sme_row),
                (1, values_per_sme_row),
            ),
            fx.make_layout(
                (1, SMEM_ROWS // (WARP_SIZE // values_per_sme_row)),
                (1, 1),
            ),
        )
        st_k = tiled_st_k.get_slice(lane_id)
        st_mn = tiled_st_mn.get_slice(lane_id)

        for t in fx.range_constexpr(a_per_warp):
            atom_idx = warp_id * fx.Int32(a_per_warp) + fx.Int32(t)
            a_off = atom_idx * fx.Int32(SMEM_ROWS * values_per_sme_row)
            smem_tile = mr_sme_shared_view(
                smem_elem_base,
                a_off,
                g2s_sme.a_sme_sw,
                fx_dtype,
                major=g2s_sme.a_smem_major,
            )
            dst_tile = fx.make_view(
                fx.add_offset(fx.get_iter(A_out), a_off + a_out_cta_base),
                (
                    fx.make_layout(
                        (values_per_sme_row, SMEM_ROWS),
                        (SMEM_ROWS, 1),
                    )
                    if fx.const_expr(pattern_id == 1 or pattern_id == 3)
                    else fx.make_layout(
                        (SMEM_ROWS, values_per_sme_row),
                        (values_per_sme_row, 1),
                    )
                ),
            )
            if fx.const_expr(pattern_id == 1 or pattern_id == 3):
                frag = fx.make_fragment_like(st_mn.partition_S(smem_tile))
                fx.copy(scalar_atom, st_mn.partition_S(smem_tile), frag)
                fx.copy(scalar_atom, frag, st_mn.partition_D(dst_tile))
            else:
                frag = fx.make_fragment_like(st_k.partition_S(smem_tile))
                fx.copy(scalar_atom, st_k.partition_S(smem_tile), frag)
                fx.copy(scalar_atom, frag, st_k.partition_D(dst_tile))

        for t in fx.range_constexpr(b_per_warp):
            atom_idx = warp_id * fx.Int32(b_per_warp) + fx.Int32(t)
            if fx.const_expr(pattern_id == 0 or pattern_id == 1):
                ni = atom_idx % fx.Int32(brick_n // values_per_sme_row)
                ki = atom_idx // fx.Int32(brick_n // values_per_sme_row)
                b_linear = ki * fx.Int32(b_n_chunks) + ni
                b_off = fx.Int32(brick_m * brick_k) + b_linear * fx.Int32(SMEM_ROWS * values_per_sme_row)
            else:
                b_off = fx.Int32(brick_m * brick_k) + atom_idx * fx.Int32(SMEM_ROWS * values_per_sme_row)
            dst_off = atom_idx * fx.Int32(SMEM_ROWS * values_per_sme_row)
            smem_tile = mr_sme_shared_view(
                smem_elem_base,
                b_off,
                g2s_sme.b_sme_sw,
                fx_dtype,
                major=g2s_sme.b_smem_major,
            )
            dst_tile = fx.make_view(
                fx.add_offset(fx.get_iter(B_out), dst_off),
                (
                    fx.make_layout(
                        (values_per_sme_row, SMEM_ROWS),
                        (SMEM_ROWS, 1),
                    )
                    if fx.const_expr(pattern_id == 0 or pattern_id == 1)
                    else fx.make_layout(
                        (SMEM_ROWS, values_per_sme_row),
                        (values_per_sme_row, 1),
                    )
                ),
            )
            if fx.const_expr(pattern_id == 0 or pattern_id == 1):
                frag = fx.make_fragment_like(st_mn.partition_S(smem_tile))
                fx.copy(scalar_atom, st_mn.partition_S(smem_tile), frag)
                fx.copy(scalar_atom, frag, st_mn.partition_D(dst_tile))
            else:
                frag = fx.make_fragment_like(st_k.partition_S(smem_tile))
                fx.copy(scalar_atom, st_k.partition_S(smem_tile), frag)
                fx.copy(scalar_atom, frag, st_k.partition_D(dst_tile))

    @flyc.jit
    def launch(
        A: fx.Tensor,
        B: fx.Tensor,
        A_out: fx.Tensor,
        B_out: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        async_copy_dump_kernel(A, B, A_out, B_out).launch(
            grid=(grid_m, 1, 1),
            block=(threads, 1, 1),
            smem=smem_elems * elem_bytes,
            stream=stream,
        )

    return launch, a_atoms_total, b_atoms_total, values_per_sme_row, grid_m


def _run_multibrick_g2s_dump_check(
    torch,
    flyc,
    fx,
    ixdl,
    *,
    major_pattern: str,
    dtype_case,
    brick_k: int,
    brick_m: int = STAGED_BRICK_M,
    brick_n: int = STAGED_BRICK_N,
    total_m: int | None = None,
    use_block_idx_m: bool = False,
):
    """Run G2S multibrick dump and compare against host reference."""
    total_m = brick_m if total_m is None else total_m
    launch, a_atoms_total, b_atoms_total, values_per_sme_row, grid_m = _compile_multibrick_async_copy_dump_kernel(
        flyc,
        fx,
        ixdl,
        major_pattern,
        dtype_case,
        brick_m=brick_m,
        brick_n=brick_n,
        brick_k=brick_k,
        use_block_idx_m=use_block_idx_m,
    )
    torch_dtype = getattr(torch, dtype_case["torch_dtype"])
    brick_elems = SMEM_ROWS * values_per_sme_row
    A_logical = multibrick_position_tensor(torch, (total_m, brick_k), torch_dtype)
    B_logical = multibrick_position_tensor(torch, (brick_n, brick_k), torch_dtype)
    if torch_dtype != torch.int8:
        B_logical = B_logical + torch.tensor(17.0, device="cuda", dtype=torch_dtype)
    A_dev, B_dev = remap_hgemm_tensors_for_pattern(A_logical, B_logical, major_pattern)
    A_out = torch.empty(grid_m * a_atoms_total * brick_elems, device="cuda", dtype=torch_dtype)
    B_out = torch.empty(b_atoms_total * brick_elems, device="cuda", dtype=torch_dtype)

    launch(A_dev, B_dev, A_out, B_out)
    torch.cuda.synchronize()

    if use_block_idx_m:
        pattern_id = major_pattern_id(major_pattern)
        for bx in range(grid_m):
            a_slice = A_logical[bx * brick_m : (bx + 1) * brick_m, :]
            if pattern_id in (0, 2):
                a_dev_slice = A_dev[bx * brick_m : (bx + 1) * brick_m, :]
            elif pattern_id in (1, 3):
                a_dev_slice = A_dev[:, bx * brick_m : (bx + 1) * brick_m]
            else:
                a_dev_slice = a_slice
            expected_a = expected_multibrick_a_dump(
                torch, a_slice, a_dev_slice, major_pattern, brick_k, values_per_sme_row
            )
            got_a = A_out[bx * a_atoms_total * brick_elems : (bx + 1) * a_atoms_total * brick_elems]
            torch.testing.assert_close(
                got_a,
                expected_a,
                rtol=0,
                atol=0,
                msg=f"{dtype_case['name']} {major_pattern} A multi-CTA G2S dump mismatch block={bx}",
            )
    else:
        torch.testing.assert_close(
            A_out,
            expected_multibrick_a_dump(torch, A_logical, A_dev, major_pattern, brick_k, values_per_sme_row),
            rtol=0,
            atol=0,
            msg=f"{dtype_case['name']} {major_pattern} A multi-brick async-copy dump mismatch",
        )
    torch.testing.assert_close(
        B_out,
        expected_multibrick_b_dump(torch, B_dev, major_pattern, brick_n, brick_k, values_per_sme_row),
        rtol=0,
        atol=0,
        msg=f"{dtype_case['name']} {major_pattern} B multi-brick async-copy dump mismatch",
    )


@pytest.mark.parametrize("spec", _CASES, ids=[c["name"] for c in _CASES])
def test_mr_async_cp_single_atom_layout_device(spec, monkeypatch):
    _require_enabled()
    flyc, fx, ixdl = _require_imports()
    torch = _require_torch()

    _configure_iluvatar_env(monkeypatch)

    fx_dtype = getattr(fx, spec["fx_dtype"])
    torch_dtype = getattr(torch, spec["torch_dtype"])
    swizzle = getattr(ixdl.SMESwizzle, spec["swizzle"])
    scalar_atom_factory = getattr(fx, spec["scalar_atom"])

    m = spec["m"]
    n = spec["n"]
    src_stride_n = spec["src_stride_n"]
    tile_m = 16  # one SME instruction always moves a 16 x 512b tile
    tile_n = spec["tile_n"]
    tile_elems = tile_m * tile_n
    tile_rows = m // tile_m
    tile_cols = n // tile_n
    matrix_elems = m * n
    smem_bytes = matrix_elems * (spec["elem_bits"] // 8)

    # Readback tiling: 64 lanes over the (tile_m, tile_n) tile.
    threads_n = THREADS // tile_m
    val_n = tile_n // threads_n

    @flyc.kernel
    def kernel(src: fx.Tensor, dst: fx.Tensor):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        # Physical shared-memory layout for this swizzle state, in element
        # granularity. major=K gives a (tile_m, tile_n)-shaped layout whose
        # logical (m, n) coordinates match the row-major source/destination.
        smem_phys = ixdl.make_sme_shared_layout(swizzle, fx_dtype, major=ixdl.SMEMajor.K)
        # Compact contiguous footprint view for the SME load (keeps the tile as
        # one atom unit; the SME instruction ignores this layout for placement).
        load_layout = fx.make_layout((tile_m, tile_n), (1, tile_m))

        smem = fx.make_view(fx.get_dyn_shared(fx_dtype), fx.make_layout(matrix_elems, 1))
        sme_src = ixdl.make_sme_gmem_tensor(src, leading_stride=src_stride_n)
        sme_src_iter = fx.get_iter(sme_src)
        smem_iter = fx.get_iter(smem)
        dst_iter = fx.get_iter(dst)

        async_atom = fx.make_copy_atom(ixdl.MRAsyncCp(swizzle), fx_dtype)
        scalar_atom = fx.make_copy_atom(scalar_atom_factory(), fx_dtype)

        tiled_ld = fx.make_tiled_copy_tv(async_atom, fx.make_layout((1, 1), (1, 1)), load_layout)
        tiled_st = fx.make_tiled_copy_tv(
            scalar_atom,
            fx.make_layout((tile_m, threads_n), (1, tile_m)),
            fx.make_layout((1, val_n), (1, 1)),
        )

        src_block_offset = bid * fx.Index(m * src_stride_n)
        dst_block_offset = bid * fx.Index(matrix_elems)

        # Phase 1: one warp-collective SME async copy per tile. Use
        # fx.range_constexpr (attribute on the already-captured ``fx``) rather than
        # a bare captured ``range_constexpr`` name: the AST rewriter rewrites the
        # constexpr loop and drops the standalone free var, tripping the
        # ``__code__`` free-var count check for closure kernels.
        for tm in fx.range_constexpr(tile_rows):
            for tn in fx.range_constexpr(tile_cols):
                tile_id = tm * tile_cols + tn
                src_off = fx.Int32(src_block_offset + fx.Index(tm * tile_m * src_stride_n + tn * tile_n))
                smem_off = fx.Int32(tile_id * tile_elems)
                src_ld = fx.make_view(fx.add_offset(sme_src_iter, src_off), load_layout)
                smem_ld = fx.make_view(fx.add_offset(smem_iter, smem_off), load_layout)
                ld = tiled_ld.get_slice(tid)
                fx.copy(async_atom, ld.partition_S(src_ld), ld.partition_D(smem_ld))

        ixdl.cp_async_commit_group()
        ixdl.cp_async_wait_group(0)
        fx.gpu.barrier()

        # Phase 2: scalar tiled readback shared -> register -> global.
        for tm in fx.range_constexpr(tile_rows):
            for tn in fx.range_constexpr(tile_cols):
                tile_id = tm * tile_cols + tn
                smem_off = fx.Int32(tile_id * tile_elems)
                dst_off = fx.Int32(dst_block_offset + fx.Index(tm * tile_m * n + tn * tile_n))
                smem_tile = fx.make_view(fx.add_offset(smem_iter, smem_off), smem_phys)
                dst_tile = fx.make_view(fx.add_offset(dst_iter, dst_off), fx.make_layout((tile_m, tile_n), (n, 1)))
                st = tiled_st.get_slice(tid)
                part_smem = st.partition_S(smem_tile)
                part_dst = st.partition_D(dst_tile)
                frag = fx.make_fragment_like(part_smem)
                fx.copy(scalar_atom, part_smem, frag)
                fx.copy(scalar_atom, frag, part_dst)

    @flyc.jit
    def launch(src: fx.Tensor, dst: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(src, dst).launch(
            grid=(BLOCKS, 1, 1),
            block=(THREADS, 1, 1),
            smem=smem_bytes,
            stream=stream,
        )

    src, dst, expected = _position_encoded_tensors(
        torch, dtype=torch_dtype, matrix_m=m, matrix_n=n, src_stride_n=src_stride_n
    )
    launch(src, dst)
    torch.cuda.synchronize()
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype_case", _MULTIBRICK_DTYPE_CASES, ids=[c["name"] for c in _MULTIBRICK_DTYPE_CASES])
@pytest.mark.parametrize(
    "major_pattern",
    ("nt", "nn", "tn", "tt"),
    ids=[
        "A-row_B-col",
        "A-row_B-row",
        "A-col_B-row",
        "A-col_B-col",
    ],
)
def test_mr_async_cp_multibrick_layout_device(major_pattern, dtype_case, monkeypatch):
    """Multi-brick async-copy placement for A/B operand layouts.

    "Brick" here means one warp-collective MR async-copy instruction footprint,
    not a GEMM tile. One instruction writes 16 rows x 64 bytes to shared memory:

        elem bits        one brick logical footprint
        ---------        ---------------------------
        b8               16 x 64 values
        b16              16 x 32 values
        b32              16 x 16 values

    A larger logical operand tile is decomposed into multiple instruction
    footprints. For b16, a row/K-major 256 x 64 A operand is split like this:

        K ->
              32 values      32 values
           +-------------+-------------+
        M  | brick 0     | brick 1     | 16 rows
        |  +-------------+-------------+
        v  | brick 2     | brick 3     | 16 rows
           +-------------+-------------+
           |    ...      |    ...      |
           +-------------+-------------+
           | brick 30    | brick 31    |
           +-------------+-------------+

    With 16 warps, each warp owns two of the 32 b16 bricks:

        warp 0  -> brick 0,  brick 1
        warp 1  -> brick 2,  brick 3
        ...
        warp 15 -> brick 30, brick 31

    The same physical footprint may be interpreted through different SME views:

        K-major view:   16 rows x values_per_sme_row
        MN-major view:  values_per_sme_row x 16 rows

    This test does not validate GEMM compute. It validates that after
    multi-brick async copies, shared memory can be read back through the
    corresponding A/B SME logical view for b8, b16, and b32.
    """

    _require_enabled()
    flyc, fx, ixdl = _require_imports()
    torch = _require_torch()

    _configure_iluvatar_env(monkeypatch)

    _run_multibrick_g2s_dump_check(
        torch,
        flyc,
        fx,
        ixdl,
        major_pattern=major_pattern,
        dtype_case=dtype_case,
        brick_k=STAGED_BRICK_K_DEFAULT,
    )


def test_mr_async_cp_runtime_goffset_loop_device(monkeypatch):
    """Runtime scf.for K-loop with loop-carried gOffset advancement.

    One warp per block walks ``K_TILES`` consecutive column tiles of a 16-row band
    inside a real ``fx.range`` loop whose carried state is the source / shared
    element offsets. The SME descriptor base stays loop-invariant; only the narrow
    per-tile offset advances (emitted as the hardware gOffset operand), mirroring
    the ``gOffset += tile_n`` pattern in production Iluvatar SME loops. f32 /
    NoSwizzle is representative -- the gOffset path is dtype-independent.
    """
    _require_enabled()
    flyc, fx, ixdl = _require_imports()
    torch = _require_torch()

    _configure_iluvatar_env(monkeypatch)

    swizzle = ixdl.SMESwizzle.NoSwizzle
    tile_m = 16
    tile_n = 16
    tile_elems = tile_m * tile_n
    k_tiles = 4
    m = tile_m
    n = tile_n * k_tiles
    src_stride_n = 80  # 80 * 4B = 320B = 5 * 64B, keeps the descriptor 64B-aligned
    matrix_elems = m * n
    smem_bytes = matrix_elems * 4

    threads_n = THREADS // tile_m
    val_n = tile_n // threads_n

    @flyc.kernel
    def kernel(src: fx.Tensor, dst: fx.Tensor):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        smem_phys = ixdl.make_sme_shared_layout(swizzle, fx.Float32, major=ixdl.SMEMajor.K)
        load_layout = fx.make_layout((tile_m, tile_n), (1, tile_m))

        smem = fx.make_view(fx.get_dyn_shared(fx.Float32), fx.make_layout(matrix_elems, 1))
        sme_src = ixdl.make_sme_gmem_tensor(src, leading_stride=src_stride_n)
        sme_src_iter = fx.get_iter(sme_src)
        smem_iter = fx.get_iter(smem)
        dst_iter = fx.get_iter(dst)

        async_atom = fx.make_copy_atom(ixdl.MRAsyncCp(swizzle), fx.Float32)
        scalar_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)

        tiled_ld = fx.make_tiled_copy_tv(async_atom, fx.make_layout((1, 1), (1, 1)), load_layout)
        tiled_st = fx.make_tiled_copy_tv(
            scalar_atom,
            fx.make_layout((tile_m, threads_n), (1, tile_m)),
            fx.make_layout((1, val_n), (1, 1)),
        )

        src_block_offset = bid * fx.Index(m * src_stride_n)
        dst_block_offset = bid * fx.Index(matrix_elems)

        # Phase 1: runtime K-loop carrying [src_col_offset, smem_offset]. Only
        # these narrow offsets advance; the descriptor base is loop-invariant.
        init_state = [fx.Int32(0), fx.Int32(0)]
        for _k, state in fx.range(0, k_tiles, 1, init=init_state):
            col_off = state[0]
            smem_off = fx.Int32(state[1])
            src_off = fx.Int32(src_block_offset + fx.Index(col_off))
            src_ld = fx.make_view(fx.add_offset(sme_src_iter, src_off), load_layout)
            smem_ld = fx.make_view(fx.add_offset(smem_iter, smem_off), load_layout)
            ld = tiled_ld.get_slice(tid)
            fx.copy(async_atom, ld.partition_S(src_ld), ld.partition_D(smem_ld))
            yield [fx.Int32(col_off + fx.Int32(tile_n)), fx.Int32(smem_off + fx.Int32(tile_elems))]

        ixdl.cp_async_commit_group()
        ixdl.cp_async_wait_group(0)
        fx.gpu.barrier()

        # Phase 2: scalar tiled readback shared -> register -> global. Use
        # fx.range_constexpr (attribute on the already-captured ``fx``) instead of
        # a bare captured ``range_constexpr`` name: the AST rewriter rewrites the
        # constexpr loop and drops the standalone free var, tripping the
        # ``__code__`` free-var count check for closure kernels.
        for tn in fx.range_constexpr(k_tiles):
            smem_off = fx.Int32(tn * tile_elems)
            dst_off = fx.Int32(dst_block_offset + fx.Index(tn * tile_n))
            smem_tile = fx.make_view(fx.add_offset(smem_iter, smem_off), smem_phys)
            dst_tile = fx.make_view(fx.add_offset(dst_iter, dst_off), fx.make_layout((tile_m, tile_n), (n, 1)))
            st = tiled_st.get_slice(tid)
            part_smem = st.partition_S(smem_tile)
            part_dst = st.partition_D(dst_tile)
            frag = fx.make_fragment_like(part_smem)
            fx.copy(scalar_atom, part_smem, frag)
            fx.copy(scalar_atom, frag, part_dst)

    @flyc.jit
    def launch(src: fx.Tensor, dst: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(src, dst).launch(
            grid=(BLOCKS, 1, 1),
            block=(THREADS, 1, 1),
            smem=smem_bytes,
            stream=stream,
        )

    src, dst, expected = _position_encoded_tensors(
        torch, dtype=torch.float32, matrix_m=m, matrix_n=n, src_stride_n=src_stride_n
    )
    launch(src, dst)
    torch.cuda.synchronize()
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)


_B16_DTYPE = next(c for c in _MULTIBRICK_DTYPE_CASES if c["name"] == "b16")


@pytest.mark.parametrize(
    "major_pattern",
    ("nt", "nn", "tn", "tt"),
    ids=["A-row_B-col", "A-row_B-row", "A-col_B-row", "A-col_B-col"],
)
def test_mr_async_cp_multibrick_bk32_device(major_pattern, monkeypatch):
    """G2S multibrick dump at BK=32 (k_rep=2), the shape that exposed nn/tn bugs."""

    _require_enabled()
    flyc, fx, ixdl = _require_imports()
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)

    _run_multibrick_g2s_dump_check(
        torch,
        flyc,
        fx,
        ixdl,
        major_pattern=major_pattern,
        dtype_case=_B16_DTYPE,
        brick_k=brick_k_from_k_rep(2),
    )


@pytest.mark.parametrize(
    "major_pattern",
    ("nt", "nn", "tn", "tt"),
    ids=["A-row_B-col", "A-row_B-row", "A-col_B-row", "A-col_B-col"],
)
def test_mr_async_cp_multibrick_multi_cta_m_device(major_pattern, monkeypatch):
    """G2S multibrick dump with grid_m=2 (512x256 logical A, two M CTAs)."""

    _require_enabled()
    flyc, fx, ixdl = _require_imports()
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    _run_multibrick_g2s_dump_check(
        torch,
        flyc,
        fx,
        ixdl,
        major_pattern=major_pattern,
        dtype_case=_B16_DTYPE,
        brick_k=STAGED_BRICK_K_DEFAULT,
        total_m=512,
        use_block_idx_m=True,
    )
