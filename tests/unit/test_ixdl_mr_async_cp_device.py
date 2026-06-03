# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Opt-in Iluvatar MR SME async-copy device correctness tests.

Device counterpart to ``test_ixdl_mr_async_cp.py`` (IR-only). One case per
(dtype, swizzle) SME Load variant: each launches several thread blocks that copy
a position-encoded matrix global -> shared with one warp-collective SME async
copy per tile, wait for completion, then read shared -> global and check an exact
match. Position encoding exposes subtle physical-layout / swizzle bugs.

Set ``FLYDSL_ILUVATAR_RUN_MR_ASYNC_CP=1`` to run (needs an Iluvatar device).
"""

import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l2_device]

_REPO_ROOT = Path(__file__).resolve().parents[2]

BLOCKS = 4
THREADS = 64

# Plain-data case specs (resolved to flydsl/torch handles inside the test, after
# the Iluvatar backend env is set). Each tile is one 16 x 512b = 8192-bit SME
# footprint, so tile_n = 8192 / elem_bits. The padded source row stride must keep
# the SME descriptor stride (src_stride_n * elem_bytes) a multiple of 64 bytes,
# otherwise the SME load scrambles data -- hence f16/i8 use 64B-multiple strides.
_CASES = [
    {
        "name": "f32_row_major",
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
    {
        "name": "f32_col_major",
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
    {
        "name": "f16_row_major",
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
    {
        "name": "f16_col_major",
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
    {
        "name": "i8_row_major",
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
    {
        "name": "i8_col_major",
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


@pytest.mark.parametrize("spec", _CASES, ids=[c["name"] for c in _CASES])
def test_mr_async_cp_device(spec, monkeypatch):
    _require_enabled()
    flyc, fx, ixdl = _require_imports()
    torch = _require_torch()

    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)

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
                dst_tile = fx.make_view(
                    fx.add_offset(dst_iter, dst_off), fx.make_layout((tile_m, tile_n), (n, 1))
                )
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


def test_mr_async_cp_device_loop(monkeypatch):
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

    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)

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
