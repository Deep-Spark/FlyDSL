# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tiled copy using the Iluvatar MR NoSwizzle SME async copy.

Each warp copies one 16x16 f32 tile A -> B in two phases via the CuTe-style
TiledCopy flow (``make_tiled_copy`` + ``partition_S/D`` + ``copy``):

1. Global -> shared (G2S): one warp-collective ``MRAsyncCp(NoSwizzle)`` SME
   instruction moves the whole tile. A single logical issuer (``(1,1)`` thread
   layout) owns the 16x16 footprint; the 64-lane cooperation is internal to the
   hardware instruction.
2. Shared -> register -> global (S2R): 64 lanes each read 4 f32 from shared
   memory and write them back to global memory.

For NoSwizzle the SME instruction writes ``A(m, n)`` to ``smem[m*TILE_N + n]``
(K-major / row-major), so the readback views shared memory through the physical
layout from ``make_sme_shared_layout`` to recover logical ``(m, n)``.
"""

import os

os.environ.setdefault("FLYDSL_COMPILE_BACKEND", "iluvatar")
os.environ.setdefault("FLYDSL_RUNTIME_KIND", "iluvatar")
os.environ.setdefault("ARCH", "ivcore11")

import torch  # noqa: E402

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
import flydsl.expr.ixdl as ixdl  # noqa: E402
from kernels.iluvatar_common import WARP_SIZE  # noqa: E402
from kernels.iluvatar_mr_common import ATOM_M, ATOM_N  # noqa: E402

WARPS_PER_BLOCK = 4
TILE_M = ATOM_M
TILE_N = ATOM_N
TILE_ELEMS = TILE_M * TILE_N
SMEM_BYTES = WARPS_PER_BLOCK * TILE_ELEMS * 4

M = TILE_M * 2
N = TILE_N * 4
SRC_STRIDE_N = 80
TOTAL_TILES = (M // TILE_M) * (N // TILE_N)
GRID_BLOCKS = TOTAL_TILES // WARPS_PER_BLOCK

@flyc.kernel(known_block_size=[WARP_SIZE * WARPS_PER_BLOCK, 1, 1])
def copy_kernel_iluvatar_mr(A: fx.Tensor, B: fx.Tensor):
    lane_id = fx.thread_idx.x % 64
    warp_id = fx.thread_idx.x // 64
    bid = fx.block_idx.x
    tiles_n = N // TILE_N

    swizzle = ixdl.SMESwizzle.NoSwizzle

    # Physical layout the NoSwizzle SME instruction writes: K-major (16,16):(16,1),
    # i.e. A(m, n) -> smem[m*TILE_N + n]. The S2R readback views smem through this
    # layout to recover logical (m, n).
    smem_phys_layout = ixdl.make_sme_shared_layout(swizzle, fx.Float32, major=ixdl.SMEMajor.K)

    # Wrap the global source in the SME address space; the descriptor carries the
    # padded leading stride, so visible width and storage stride may differ.
    sme_A = ixdl.make_sme_gmem_tensor(A, leading_stride=SRC_STRIDE_N)
    sme_A_iter = fx.get_iter(sme_A)
    B_iter = fx.get_iter(B)

    # One contiguous 16x16 f32 shared-memory segment per warp.
    smem = fx.make_view(fx.get_dyn_shared(fx.Float32), fx.make_layout(WARPS_PER_BLOCK * TILE_ELEMS, 1))

    async_atom = fx.make_copy_atom(ixdl.MRAsyncCp(swizzle), fx.Float32)
    scalar_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)

    # G2S TiledCopy: one logical issuer ((1,1) thread layout) owns the whole tile;
    # the 64-lane cooperation is inside the warp-collective SME instruction.
    tiled_ld = fx.make_tiled_copy_tv(
        async_atom,
        fx.make_layout((1, 1), (1, 1)),
        fx.make_layout((TILE_M, TILE_N), (1, TILE_M)),
    )

    # S2R TiledCopy: 64 lanes x 4 f32 over the 16x16 tile. With K-major smem this
    # thread/value layout maps lane -> smem[lane + 64*k], so the reads coalesce
    # into one block-load per value.
    tiled_st = fx.make_tiled_copy(
        scalar_atom,
        fx.make_layout(((16, 4), 4), ((16, 1), 4)),
        (16, 16),
    )

    # Global tile this warp is responsible for.
    tile_id = bid * fx.Index(WARPS_PER_BLOCK) + warp_id
    tile_row = tile_id // fx.Index(tiles_n)
    tile_col = tile_id % fx.Index(tiles_n)

    src_offset = fx.Int32(tile_row * fx.Index(TILE_M * SRC_STRIDE_N) + tile_col * fx.Index(TILE_N))
    smem_offset = fx.Int32(warp_id * fx.Index(TILE_ELEMS))
    dst_offset = fx.Int32(tile_row * fx.Index(TILE_M * N) + tile_col * fx.Index(TILE_N))

    # Phase 1 (G2S): one cp_async per warp tile, then wait (single-phase copy).
    src_ld = fx.make_view(fx.add_offset(sme_A_iter, src_offset), fx.make_layout((TILE_M, TILE_N), (1, TILE_M)))
    smem_ld = fx.make_view(fx.add_offset(fx.get_iter(smem), smem_offset), fx.make_layout((TILE_M, TILE_N), (1, TILE_M)))
    ld = tiled_ld.get_slice(lane_id)
    fx.copy(async_atom, ld.partition_S(src_ld), ld.partition_D(smem_ld))

    ixdl.cp_async_commit_group()
    ixdl.cp_async_wait_group(0)
    fx.gpu.barrier()

    # Phase 2 (S2R): read the tile through the physical smem layout, write to B.
    smem_tile = fx.make_view(fx.add_offset(fx.get_iter(smem), smem_offset), smem_phys_layout)
    dst_tile = fx.make_view(fx.add_offset(B_iter, dst_offset), fx.make_layout((TILE_M, TILE_N), (N, 1)))
    st = tiled_st.get_slice(lane_id)
    part_smem = st.partition_S(smem_tile)
    part_dst = st.partition_D(dst_tile)
    frag = fx.make_fragment_like(part_smem)
    fx.copy(scalar_atom, part_smem, frag)
    fx.copy(scalar_atom, frag, part_dst)


@flyc.jit
def tiledCopyIluvatarMR(A: fx.Tensor, B: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    copy_kernel_iluvatar_mr(A, B).launch(
        grid=(GRID_BLOCKS, 1, 1),
        block=(WARP_SIZE * WARPS_PER_BLOCK, 1, 1),
        smem=SMEM_BYTES,
        stream=stream,
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-compatible Iluvatar device is not available")

    storage = torch.randint(0, 10, (M, SRC_STRIDE_N), dtype=torch.float32).cuda()
    A = storage[:, :N]
    B = torch.zeros(M, N, dtype=torch.float32).cuda()

    tiledCopyIluvatarMR(A, B, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    is_correct = torch.allclose(B, A)
    print("Result correct:", is_correct)
    if not is_correct:
        print("A:", A)
        print("B:", B)


if __name__ == "__main__":
    main()
