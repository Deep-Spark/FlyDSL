# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Iluvatar MR ``sl_barrier_alu`` device correctness.

Covers a two-warp shared handoff (``sl_waitmem(lm=0)`` + ``sl_barrier_alu``)
and a two-stage SME G2S pipeline that issues the next copy before waiting.
"""

import os
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.iluvatar_lower]

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kernels.gemm.iluvatar.common import WARP_SIZE  # noqa: E402

BLOCKS = 4
TILE_F32 = 256  # 1KB NoSwizzle SME tile (16 x 16 f32)
TILES = 2
SME_ROW_F32 = 16
EXCHANGE_THREADS = 2 * WARP_SIZE


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
        pytest.skip(f"torch is required for the Iluvatar MR sl_barrier_alu device tests: {exc}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA-compatible device is not available")
    return torch


def _configure_iluvatar_env(monkeypatch):
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)


def test_mr_sl_barrier_alu_smem_exchange_device(monkeypatch):
    """Two-warp shared publish: sl_waitmem(lm=0) + sl_barrier_alu, exact swap."""
    flyc, fx, ixdl = _require_imports()
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)

    @flyc.kernel(known_block_size=[EXCHANGE_THREADS, 1, 1])
    def sl_barrier_alu_smem_exchange(src: fx.Tensor, dst: fx.Tensor):
        tid = fx.thread_idx.x
        src_iter = fx.get_iter(src)
        dst_iter = fx.get_iter(dst)
        smem = fx.make_view(fx.get_dyn_shared(fx.Float32), fx.make_layout(EXCHANGE_THREADS, 1))
        smem_iter = fx.get_iter(smem)

        fx.ptr_store(
            fx.ptr_load(fx.add_offset(src_iter, fx.make_int_tuple(tid)), fx.Float32),
            fx.add_offset(smem_iter, fx.make_int_tuple(tid)),
        )
        # Drain STS before the other warp reads; ALU barrier syncs the CTA
        # without an extra memory waitcnt.
        ixdl.sl_waitmem(lm=0)
        ixdl.sl_barrier_alu()
        peer = tid ^ fx.Int32(WARP_SIZE)
        fx.ptr_store(
            fx.ptr_load(fx.add_offset(smem_iter, fx.make_int_tuple(peer)), fx.Float32),
            fx.add_offset(dst_iter, fx.make_int_tuple(tid)),
        )

    @flyc.jit
    def launch(src: fx.Tensor, dst: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        sl_barrier_alu_smem_exchange(src, dst).launch(
            grid=(1, 1, 1),
            block=(EXCHANGE_THREADS, 1, 1),
            smem=EXCHANGE_THREADS * 4,
            stream=stream,
        )

    src = torch.arange(EXCHANGE_THREADS, device="cuda", dtype=torch.float32)
    dst = torch.full((EXCHANGE_THREADS,), -1.0, device="cuda", dtype=torch.float32)
    launch(src, dst)
    torch.cuda.synchronize()
    torch.testing.assert_close(dst, src.roll(WARP_SIZE), rtol=0, atol=0)


def test_mr_sl_barrier_alu_g2s_pipeline_device(monkeypatch):
    """Issue G2S, sl_barrier_alu, issue next, sl_waitmem(g2s=1/0); exact dump."""
    flyc, fx, ixdl = _require_imports()
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)

    tile_elems = TILE_F32
    vpt = tile_elems // WARP_SIZE

    @flyc.kernel(known_block_size=[WARP_SIZE, 1, 1])
    def sl_barrier_alu_g2s_pipeline(src: fx.Tensor, dst: fx.Tensor):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        load_layout = fx.make_layout((16, 16), (1, 16))
        dump_layout = fx.make_layout((WARP_SIZE, vpt), (vpt, 1))

        smem = fx.make_view(fx.get_dyn_shared(fx.Float32), fx.make_layout(TILES * tile_elems, 1))
        smem_iter = fx.get_iter(smem)
        dst_iter = fx.get_iter(dst)
        sme_src = ixdl.make_sme_gmem_tensor(src, leading_stride=SME_ROW_F32)
        sme_src_iter = fx.get_iter(sme_src)

        async_atom = fx.make_copy_atom(ixdl.MRAsyncCp(ixdl.SMESwizzle.NoSwizzle), fx.Float32)
        tiled_ld = fx.make_tiled_copy_tv(async_atom, fx.make_layout((1, 1), (1, 1)), load_layout)
        ld = tiled_ld.get_slice(tid)
        scalar_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
        tiled_dump = fx.make_tiled_copy_tv(
            scalar_atom,
            fx.make_layout((WARP_SIZE, 1), (vpt, 1)),
            fx.make_layout((1, vpt), (0, 1)),
        )
        dump = tiled_dump.get_slice(tid)

        block_base = bid * fx.Index(TILES * tile_elems)

        src0 = fx.make_view(fx.add_offset(sme_src_iter, fx.Int32(block_base)), load_layout)
        smem0 = fx.make_view(smem_iter, load_layout)
        fx.copy(async_atom, ld.partition_S(src0), ld.partition_D(smem0))
        ixdl.cp_async_commit_group()

        # ALU CTA sync only; the next G2S issue stays in flight.
        ixdl.sl_barrier_alu()

        src1 = fx.make_view(
            fx.add_offset(sme_src_iter, fx.Int32(block_base + fx.Index(tile_elems))),
            load_layout,
        )
        smem1 = fx.make_view(fx.add_offset(smem_iter, fx.Int32(tile_elems)), load_layout)
        fx.copy(async_atom, ld.partition_S(src1), ld.partition_D(smem1))
        ixdl.cp_async_commit_group()

        ixdl.sl_waitmem(g2s=1)
        smem0_dump = fx.make_view(smem_iter, dump_layout)
        dst0 = fx.make_view(fx.add_offset(dst_iter, fx.Int32(block_base)), dump_layout)
        frag0 = fx.make_fragment_like(dump.partition_S(smem0_dump))
        fx.copy(scalar_atom, dump.partition_S(smem0_dump), frag0)
        fx.copy(scalar_atom, frag0, dump.partition_D(dst0))

        ixdl.sl_waitmem(g2s=0)
        smem1_dump = fx.make_view(fx.add_offset(smem_iter, fx.Int32(tile_elems)), dump_layout)
        dst1 = fx.make_view(
            fx.add_offset(dst_iter, fx.Int32(block_base + fx.Index(tile_elems))),
            dump_layout,
        )
        frag1 = fx.make_fragment_like(dump.partition_S(smem1_dump))
        fx.copy(scalar_atom, dump.partition_S(smem1_dump), frag1)
        fx.copy(scalar_atom, frag1, dump.partition_D(dst1))

    @flyc.jit
    def launch(src: fx.Tensor, dst: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        sl_barrier_alu_g2s_pipeline(src, dst).launch(
            grid=(BLOCKS, 1, 1),
            block=(WARP_SIZE, 1, 1),
            smem=TILES * tile_elems * 4,
            stream=stream,
        )

    n = BLOCKS * TILES * tile_elems
    src = torch.arange(n, device="cuda", dtype=torch.float32)
    dst = torch.full((n,), -1.0, device="cuda", dtype=torch.float32)
    launch(src, dst)
    torch.cuda.synchronize()
    torch.testing.assert_close(dst, src, rtol=0, atol=0)
