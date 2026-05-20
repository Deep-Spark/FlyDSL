# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import os

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.iluvatar as ix

TILE_M = 16
TILE_N = 16
TILE_BITS = TILE_M * TILE_N * 32
SMEM_BYTES = TILE_M * TILE_N * 4
ELEMS_PER_LANE = (TILE_M * TILE_N) // 64


@flyc.kernel
def tiled_copy_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    pitch_elems: fx.Int32,
    pitch_bytes: fx.Int32,
):
    tid = fx.thread_idx.x
    tile_row = fx.block_idx.y * TILE_M
    tile_col = fx.block_idx.x * TILE_N

    src_offset = tile_row * pitch_elems + tile_col
    dst_offset = tile_row * pitch_elems + tile_col

    g_src_layout = fx.make_layout((TILE_M, TILE_N), (pitch_elems, 1))
    s_layout = fx.make_layout((TILE_M, TILE_N), (TILE_N, 1))

    g2s = fx.make_copy_atom(ix.AsyncCopy16x16B32Row(), TILE_BITS)
    g2s = g2s.set_value("stride_byte", pitch_bytes)

    s_base = fx.get_dyn_shared(fx.Int32)
    sA = fx.make_view(s_base, s_layout)

    gA = fx.make_view(fx.add_offset(fx.get_iter(A), fx.make_int_tuple(src_offset)), g_src_layout)
    gB_base = fx.add_offset(fx.get_iter(B), fx.make_int_tuple(dst_offset))

    fx.copy(g2s, gA, sA)
    ix.cp_async_commit_group()
    ix.cp_async_wait_group(0)

    lane_base = tid * ELEMS_PER_LANE
    for i in fx.range_constexpr(ELEMS_PER_LANE):
        elem_idx = lane_base + i
        src_ptr = fx.add_offset(s_base, fx.make_int_tuple(elem_idx))
        row = elem_idx // TILE_N
        col = elem_idx % TILE_N
        dst_idx = row * pitch_elems + col
        dst_ptr = fx.add_offset(gB_base, fx.make_int_tuple(dst_idx))
        fx.ptr_store(fx.ptr_load(src_ptr), dst_ptr)


@flyc.jit
def tiled_copy(
    A: fx.Tensor,
    B: fx.Tensor,
    pitch_elems: fx.Int32,
    pitch_bytes: fx.Int32,
    m: fx.Constexpr[int],
    n: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    grid = (n // TILE_N, m // TILE_M, 1)
    tiled_copy_kernel(A, B, pitch_elems, pitch_bytes).launch(
        grid=grid, block=(64, 1, 1), smem=SMEM_BYTES, stream=stream
    )


def _configure_iluvatar_env():
    os.environ.setdefault("FLYDSL_COMPILE_BACKEND", "iluvatar")
    os.environ.setdefault("FLYDSL_RUNTIME_KIND", "iluvatar")
    os.environ.setdefault("ARCH", "ivcore11")


if __name__ == "__main__":
    _configure_iluvatar_env()

    m = 128
    n = 128
    pitch_elems = n
    pitch_bytes = pitch_elems * 4

    A = torch.arange(m * n, dtype=torch.int32, device="cuda").reshape(m, n)
    B = torch.zeros((m, n), dtype=torch.int32, device="cuda")

    tiled_copy(A, B, pitch_elems, pitch_bytes, m, n, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    ok = torch.equal(A, B)
    print("Result correct:", ok)
    if not ok:
        print("A[:4, :8]:\n", A[:4, :8])
        print("B[:4, :8]:\n", B[:4, :8])
