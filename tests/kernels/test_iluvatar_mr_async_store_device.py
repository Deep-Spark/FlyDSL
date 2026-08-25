# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Iluvatar MR SME async-store (shared -> global) device correctness tests.

``MRAsyncStore`` is the warp-collective S2G counterpart of ``MRAsyncCp``: one
instruction moves 64/128/256 bytes from shared memory to a global destination
described by an SME gmem tensor (``ixdl.cp_async.store.b{64,128,256}`` ->
``bi_sme_store_b64/b128/b256``).

Each case moves a position-encoded 1KB f32 tile per block and checks an exact
match; position encoding exposes subtle offset/placement bugs. Two fill
variants cover the realistic producer paths for the shared stage:

* scalar stores (LDG -> register -> STS), drained with ``sl_waitmem(lm=0)``;
* an SME G2S load (``MRAsyncCp``), drained with ``cp_async_wait_group(0)`` --
  the vendor index_select pipeline shape (G2S -> S2G through shared memory).
"""

import os
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.iluvatar_lower]

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

BLOCKS = 4
THREADS = 64  # one warp per block: the SME store is warp-collective
TILE_F32 = 256  # 1KB shared stage per block
SME_ROW_F32 = 16  # descriptor leading stride: 16 f32 = 64B rows

_STORE_WIDTHS = [
    {"store_bytes": 64, "elems": 16},
    {"store_bytes": 128, "elems": 32},
    {"store_bytes": 256, "elems": 64},
]


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
        pytest.skip(f"torch is required for the Iluvatar MR async-store device tests: {exc}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA-compatible device is not available")
    return torch


def _configure_iluvatar_env(monkeypatch):
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)


def _position_encoded_src(torch):
    """One position-encoded 1KB tile per block: value == flat element index."""
    return torch.arange(BLOCKS * TILE_F32, device="cuda", dtype=torch.float32)


def _emit_store_phase(fx, ixdl, *, tid, bid, smem_iter, dst, store_bytes, elems_per_store):
    """Drain the shared tile to ``dst`` with warp-collective SME stores."""
    sme_dst = ixdl.make_sme_gmem_tensor(dst, leading_stride=SME_ROW_F32)
    sme_dst_iter = fx.get_iter(sme_dst)

    store_atom = fx.make_copy_atom(ixdl.MRAsyncStore(store_bytes), fx.Float32)
    # Rank-2 (16, elems/16) view of each linear store segment: the tiled-copy
    # partitioning machinery expects the thr/val layouts and the partitioned
    # views to be rank-2 (same contract as the MRAsyncCp G2S tests).
    store_layout = fx.make_layout((16, elems_per_store // 16), (1, 16))
    tiled_st = fx.make_tiled_copy_tv(store_atom, fx.make_layout((1, 1), (1, 1)), store_layout)
    st = tiled_st.get_slice(tid)

    dst_block_off = bid * fx.Index(TILE_F32)
    for i in fx.range_constexpr(TILE_F32 // elems_per_store):
        smem_off = fx.Int32(i * elems_per_store)
        dst_off = fx.Int32(dst_block_off + fx.Index(i * elems_per_store))
        smem_view = fx.make_view(fx.add_offset(smem_iter, smem_off), store_layout)
        dst_view = fx.make_view(fx.add_offset(sme_dst_iter, dst_off), store_layout)
        fx.copy(store_atom, st.partition_S(smem_view), st.partition_D(dst_view))
    ixdl.sl_waitmem(s2g=0)


@pytest.mark.parametrize("spec", _STORE_WIDTHS, ids=[f"b{c['store_bytes']}" for c in _STORE_WIDTHS])
def test_mr_async_store_scalar_fill_device(spec, monkeypatch):
    """Scalar-filled shared tile -> SME store -> global, exact match."""
    flyc, fx, ixdl = _require_imports()
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)

    store_bytes = spec["store_bytes"]
    elems_per_store = spec["elems"]

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def kernel(src: fx.Tensor, dst: fx.Tensor):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        smem = fx.make_view(fx.get_dyn_shared(fx.Float32), fx.make_layout(TILE_F32, 1))
        smem_iter = fx.get_iter(smem)
        src_iter = fx.get_iter(src)

        # Phase 1: scalar fill of the shared tile (64 lanes x 4 f32), rank-2 TV.
        vpt = TILE_F32 // THREADS
        tile2d = fx.make_layout((THREADS, vpt), (vpt, 1))
        scalar_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
        tiled_fill = fx.make_tiled_copy_tv(
            scalar_atom,
            fx.make_layout((THREADS, 1), (vpt, 1)),
            fx.make_layout((1, vpt), (0, 1)),
        )
        fill = tiled_fill.get_slice(tid)
        src_tile = fx.make_view(
            fx.add_offset(src_iter, fx.Int32(bid * fx.Index(TILE_F32))),
            tile2d,
        )
        smem_tile = fx.make_view(smem_iter, tile2d)
        frag = fx.make_fragment_like(fill.partition_S(src_tile))
        fx.copy(scalar_atom, fill.partition_S(src_tile), frag)
        fx.copy(scalar_atom, frag, fill.partition_D(smem_tile))
        # Drain shared-memory writes before the SME engine reads the stage.
        ixdl.sl_waitmem(lm=0)
        fx.gpu.barrier()

        # Phase 2: warp-collective SME stores shared -> global.
        _emit_store_phase(
            fx, ixdl, tid=tid, bid=bid, smem_iter=smem_iter, dst=dst,
            store_bytes=store_bytes, elems_per_store=elems_per_store,
        )

    @flyc.jit
    def launch(src: fx.Tensor, dst: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(src, dst).launch(
            grid=(BLOCKS, 1, 1),
            block=(THREADS, 1, 1),
            smem=TILE_F32 * 4,
            stream=stream,
        )

    src = _position_encoded_src(torch)
    dst = torch.full((BLOCKS * TILE_F32,), -1.0, device="cuda", dtype=torch.float32)
    launch(src, dst)
    torch.cuda.synchronize()
    torch.testing.assert_close(dst, src, rtol=0, atol=0)


@pytest.mark.parametrize("spec", _STORE_WIDTHS, ids=[f"b{c['store_bytes']}" for c in _STORE_WIDTHS])
def test_mr_async_store_g2s_s2g_roundtrip_device(spec, monkeypatch):
    """Vendor-pipeline shape: SME G2S load -> shared -> SME S2G store."""
    flyc, fx, ixdl = _require_imports()
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)

    store_bytes = spec["store_bytes"]
    elems_per_store = spec["elems"]

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def kernel(src: fx.Tensor, dst: fx.Tensor):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        smem = fx.make_view(fx.get_dyn_shared(fx.Float32), fx.make_layout(TILE_F32, 1))
        smem_iter = fx.get_iter(smem)

        # Phase 1: one warp-collective SME G2S load (16 x 16 f32 = 1KB tile).
        # NoSwizzle keeps the shared stage linear, so the S2G stores below
        # drain it in order.
        load_layout = fx.make_layout((16, 16), (1, 16))
        sme_src = ixdl.make_sme_gmem_tensor(src, leading_stride=SME_ROW_F32)
        sme_src_iter = fx.get_iter(sme_src)
        async_atom = fx.make_copy_atom(ixdl.MRAsyncCp(ixdl.SMESwizzle.NoSwizzle), fx.Float32)
        tiled_ld = fx.make_tiled_copy_tv(async_atom, fx.make_layout((1, 1), (1, 1)), load_layout)
        ld = tiled_ld.get_slice(tid)
        src_off = fx.Int32(bid * fx.Index(TILE_F32))
        src_ld = fx.make_view(fx.add_offset(sme_src_iter, src_off), load_layout)
        smem_ld = fx.make_view(smem_iter, load_layout)
        fx.copy(async_atom, ld.partition_S(src_ld), ld.partition_D(smem_ld))
        ixdl.cp_async_commit_group()
        ixdl.cp_async_wait_group(0)

        # Phase 2: warp-collective SME stores shared -> global.
        _emit_store_phase(
            fx, ixdl, tid=tid, bid=bid, smem_iter=smem_iter, dst=dst,
            store_bytes=store_bytes, elems_per_store=elems_per_store,
        )

    @flyc.jit
    def launch(src: fx.Tensor, dst: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(src, dst).launch(
            grid=(BLOCKS, 1, 1),
            block=(THREADS, 1, 1),
            smem=TILE_F32 * 4,
            stream=stream,
        )

    src = _position_encoded_src(torch)
    dst = torch.full((BLOCKS * TILE_F32,), -1.0, device="cuda", dtype=torch.float32)
    launch(src, dst)
    torch.cuda.synchronize()
    torch.testing.assert_close(dst, src, rtol=0, atol=0)
