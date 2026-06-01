# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tiled copy using Iluvatar MR SME async copy.

This is the Iluvatar MR variant of ``02-tiledCopy.py``. Each CTA contains
multiple warps; every warp copies one 16x16 f32 tile from global memory to its
own shared-memory segment with ``ixdl.cp_async.16x16.b32.row``. After all async
copies complete, each warp writes its tile back to global memory with ordinary
scalar copy atoms.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("FLYDSL_COMPILE_BACKEND", "iluvatar")
os.environ.setdefault("FLYDSL_RUNTIME_KIND", "iluvatar")
os.environ.setdefault("ARCH", "ivcore11")

import torch  # noqa: E402

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
import flydsl.expr.ixdl as ixdl  # noqa: E402
from flydsl.expr import range_constexpr  # noqa: E402


WARP_SIZE = 64
WARPS_PER_BLOCK = 4
TILE_M = 16
TILE_N = 16
TILE_ELEMS = TILE_M * TILE_N
ELEMS_PER_LANE = TILE_ELEMS // WARP_SIZE
SMEM_BYTES = WARPS_PER_BLOCK * TILE_ELEMS * 4

M = TILE_M * 2
N = TILE_N * 4
SRC_STRIDE_N = 80
TOTAL_TILES = (M // TILE_M) * (N // TILE_N)
GRID_BLOCKS = TOTAL_TILES // WARPS_PER_BLOCK


@flyc.kernel
def copy_kernel_iluvatar_mr(A: fx.Tensor, B: fx.Tensor):
    # Launch shape is block=(64, 4, 1).  The x dimension is the lane id inside
    # one MR wave, and the y dimension selects which wave/warp in the CTA is
    # responsible for a tile.  Keep x at 64 so the MR CopyAtom sees the
    # 0..63 thread layout it expects.
    lane_id = fx.gpu.thread_id("x")
    warp_id = fx.gpu.thread_id("y")
    bid = fx.gpu.block_id("x")

    tiles_n = N // TILE_N

    # Shared memory is split into one contiguous 16x16 f32 segment per warp.
    # This lets several warp-level async copies run in the same CTA without
    # clobbering each other's destination tile.
    smem = fx.make_view(fx.get_dyn_shared(fx.Float32), fx.make_layout(WARPS_PER_BLOCK * TILE_ELEMS, 1))

    # Wrap the global tensor in the FlyIXDL SME address space.  MR SME load uses
    # a descriptor carrying the leading stride, so this example intentionally
    # passes the padded source stride rather than the visible matrix width.
    sme_A = ixdl.make_sme_gmem_tensor(A, leading_stride=SRC_STRIDE_N)

    # MRAsyncCpNoSwizzle lowers to ixdl.cp_async.16x16.b32.row for f32.  The
    # scalar atom below is only used after the async copy completes, to read
    # shared memory back and write the result to global memory for verification.
    async_atom = fx.make_copy_atom(ixdl.MRAsyncCpNoSwizzle(), fx.Float32)
    scalar_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    tile_layout = fx.make_layout(TILE_ELEMS, 1)
    smem_elems = fx.logical_divide(smem, fx.make_layout(1, 1))
    B_elems = fx.logical_divide(B, fx.make_layout(1, 1))
    reg = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)

    # First phase: each warp issues exactly one SME async copy for its tile.
    # The compile-time loop creates one branch per warp slot; inside each branch
    # the shared-memory segment offset is a constant, while bid still chooses
    # which group of tiles this CTA handles.
    for tile_in_block in range_constexpr(WARPS_PER_BLOCK):
        if warp_id == fx.Index(tile_in_block):
            tile_id = bid * fx.Index(WARPS_PER_BLOCK) + fx.Index(tile_in_block)
            tile_row = tile_id // fx.Index(tiles_n)
            tile_col = tile_id % fx.Index(tiles_n)

            # Source offsets are in elements.  Use SRC_STRIDE_N for row steps
            # because A is a view of padded storage, and the SME descriptor was
            # built with that same leading stride.
            src_offset = fx.Int32(tile_row * fx.Index(TILE_M * SRC_STRIDE_N) + tile_col * fx.Index(TILE_N))
            smem_offset = fx.Int32(tile_in_block * TILE_ELEMS)
            src_tile = fx.make_view(fx.add_offset(fx.get_iter(sme_A), src_offset), tile_layout)
            smem_tile = fx.make_view(fx.add_offset(fx.get_iter(smem), smem_offset), tile_layout)
            fx.copy_atom_call(async_atom, src_tile, smem_tile)

    # All warps commit their outstanding async copies, wait for completion, then
    # synchronize the CTA before any lane reads from shared memory.
    ixdl.cp_async_commit_group()
    ixdl.cp_async_wait_group(0)
    fx.gpu.barrier()

    # Second phase: each warp reads back the tile it copied.  This is not the
    # fast path one would use in a GEMM; it is deliberately simple so the example
    # can verify global -> shared -> global round-trip correctness.
    for tile_in_block in range_constexpr(WARPS_PER_BLOCK):
        if warp_id == fx.Index(tile_in_block):
            tile_id = bid * fx.Index(WARPS_PER_BLOCK) + fx.Index(tile_in_block)
            tile_row = tile_id // fx.Index(tiles_n)
            tile_col = tile_id % fx.Index(tiles_n)
            tile_base = tile_row * fx.Index(TILE_M * SRC_STRIDE_N) + tile_col * fx.Index(TILE_N)

            # Each lane copies four f32 values from the 16x16 tile.  local_idx is
            # the row-major coordinate within the warp's shared-memory segment.
            base = lane_id * fx.Index(ELEMS_PER_LANE)
            for i in range_constexpr(ELEMS_PER_LANE):
                local_idx = base + fx.Index(i)
                row = local_idx // fx.Index(TILE_N)
                col = local_idx % fx.Index(TILE_N)
                smem_idx = fx.Int32(tile_in_block * TILE_ELEMS) + fx.Int32(local_idx)

                # B is a flat tensor, so dst_idx is the linearized logical
                # matrix coordinate.  Keeping B flat avoids mixing this example
                # with multidimensional memref layout semantics.
                dst_idx = fx.Int32(
                    (tile_row * fx.Index(TILE_M) + row) * fx.Index(N) + tile_col * fx.Index(TILE_N) + col
                )

                fx.copy_atom_call(scalar_atom, fx.slice(smem_elems, (None, smem_idx)), reg)
                fx.copy_atom_call(scalar_atom, reg, fx.slice(B_elems, (None, dst_idx)))


@flyc.jit
def tiledCopyIluvatarMR(A: fx.Tensor, B: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    copy_kernel_iluvatar_mr(A, B).launch(
        grid=(GRID_BLOCKS, 1, 1),
        block=(WARP_SIZE, WARPS_PER_BLOCK, 1),
        smem=SMEM_BYTES,
        stream=stream,
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-compatible Iluvatar device is not available")

    storage = torch.randint(0, 10, (M, SRC_STRIDE_N), dtype=torch.float32).cuda()
    A = storage[:, :N]
    B = torch.zeros(M * N, dtype=torch.float32).cuda()

    tiledCopyIluvatarMR(A, B, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    actual = B.reshape(M, N)
    is_correct = torch.allclose(actual, A)
    print("Result correct:", is_correct)
    if not is_correct:
        print("A:", A)
        print("B:", actual)


if __name__ == "__main__":
    main()
