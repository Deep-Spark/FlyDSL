# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tiled copy using Iluvatar MR SME async copy (tiled-copy paradigm).

This is the Iluvatar MR variant of ``02-tiledCopy.py``. It uses the
``make_tiled_copy`` + ``partition_S/D`` + ``copy`` flow rather than hand-rolled
index math:

* The global -> shared async load uses the MR SME CopyAtom
  (``ixdl.cp_async.16x16.b32.row`` for f32). The atom is a single logical issuer
  (thread layout ``Layout<1>``) that owns the whole 8192-bit (16x16 f32)
  footprint, so the TiledCopy uses a ``(1,1)`` thread layout. The 64-lane
  cooperation lives inside the warp-collective SME instruction and needs no lane
  guard.
* The physical shared-memory layout comes from ``make_sme_shared_layout``
  (swizzle / data placement lives on the shared tensor, not the atom -- CopyAtom
  and shared-layout are kept orthogonal). For NoSwizzle this is the K-major
  INTER tile ``(16,16):(16,1)``, i.e. ``A(m, n)`` lands at ``smem[m*TILE_N + n]``.
* The shared -> global readback uses a plain scalar CopyAtom tiled across the
  64 lanes of the warp, viewing the shared block through that physical layout.

Each CTA owns ``WARPS_PER_BLOCK`` output tiles, one per warp.
"""

import os

os.environ.setdefault("FLYDSL_COMPILE_BACKEND", "iluvatar")
os.environ.setdefault("FLYDSL_RUNTIME_KIND", "iluvatar")
os.environ.setdefault("ARCH", "ivcore11")

import torch  # noqa: E402

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
import flydsl.expr.ixdl as ixdl  # noqa: E402

WARP_SIZE = 64
WARPS_PER_BLOCK = 4
TILE_M = 16
TILE_N = 16
TILE_ELEMS = TILE_M * TILE_N
ELEMS_PER_LANE = TILE_ELEMS // WARP_SIZE  # 4 f32 per lane on readback
SMEM_BYTES = WARPS_PER_BLOCK * TILE_ELEMS * 4

M = TILE_M * 2
N = TILE_N * 4
SRC_STRIDE_N = 80
TOTAL_TILES = (M // TILE_M) * (N // TILE_N)
GRID_BLOCKS = TOTAL_TILES // WARPS_PER_BLOCK


@flyc.kernel
def copy_kernel_iluvatar_mr(A: fx.Tensor, B: fx.Tensor):
    # block = (64, WARPS_PER_BLOCK, 1): x is the lane inside one MR wave, y picks
    # which warp / output tile this CTA slot handles.
    lane_id = fx.thread_idx.x
    warp_id = fx.thread_idx.y
    bid = fx.block_idx.x
    tiles_n = N // TILE_N

    swizzle = ixdl.SMESwizzle.NoSwizzle

    # Physical shared-memory layout for this swizzle state, in element
    # granularity. For NoSwizzle this is the K-major INTER tile (16,16):(16,1):
    # the SME instruction writes A(m, n) to smem[m*TILE_N + n], so this is the
    # layout the readback must use to recover logical (m, n) coordinates.
    # make_sme_shared_layout returns the value-granular layout the SME
    # instruction physically writes, with the byte-granular swizzle (design doc
    # flydsl-mr-async-cp-tiledcopy-alignment.html section 9.1).
    smem_phys_layout = ixdl.make_sme_shared_layout(swizzle, fx.Float32, major=ixdl.SMEMajor.K)

    # Wrap the global source in the FlyIXDL SME address space. The descriptor
    # carries the padded leading stride, so the visible matrix width and the
    # storage stride can differ.
    sme_A = ixdl.make_sme_gmem_tensor(A, leading_stride=SRC_STRIDE_N)
    sme_A_iter = fx.get_iter(sme_A)
    B_iter = fx.get_iter(B)

    # One contiguous 16x16 f32 segment of shared memory per warp.
    smem = fx.make_view(fx.get_dyn_shared(fx.Float32), fx.make_layout(WARPS_PER_BLOCK * TILE_ELEMS, 1))

    async_atom = fx.make_copy_atom(ixdl.MRAsyncCp(swizzle), fx.Float32)
    scalar_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)

    # Async load TiledCopy: a single logical issuer (thread layout 1) owns the
    # whole 8192-bit footprint (src layout (1,8192):(0,1)) -> exactly one
    # ixdl.cp_async per warp tile. The CopyAtom only consumes the tile base
    # pointers; the SME NoSwizzle instruction itself writes the gmem tile
    # row-major (A(m, n) -> smem[m*TILE_N + n]) regardless of the view layout, so
    # the load views use a compact (TILE_M, TILE_N) layout purely to keep the
    # 256-element footprint as one contiguous atom unit.
    tiled_ld = fx.make_tiled_copy_tv(
        async_atom,
        fx.make_layout((1, 1), (1, 1)),
        fx.make_layout((TILE_M, TILE_N), (1, TILE_M)),
    )

    # Readback TiledCopy: 64 lanes, ELEMS_PER_LANE f32 each, over the 16x16 tile.
    # The shared block is row-major (m*TILE_N + n), so view it as (16,16):(16,1).
    tiled_st = fx.make_tiled_copy_tv(
        scalar_atom,
        fx.make_layout((TILE_M, WARP_SIZE // TILE_M), (1, TILE_M)),
        fx.make_layout((1, ELEMS_PER_LANE), (1, 1)),
    )

    # Global tile this warp is responsible for.
    tile_id = bid * fx.Index(WARPS_PER_BLOCK) + warp_id
    tile_row = tile_id // fx.Index(tiles_n)
    tile_col = tile_id % fx.Index(tiles_n)

    src_offset = fx.Int32(tile_row * fx.Index(TILE_M * SRC_STRIDE_N) + tile_col * fx.Index(TILE_N))
    smem_offset = fx.Int32(warp_id * fx.Index(TILE_ELEMS))
    dst_offset = fx.Int32(tile_row * fx.Index(TILE_M * N) + tile_col * fx.Index(TILE_N))

    # Phase 1: one warp-collective SME async copy per tile.
    src_ld = fx.make_view(fx.add_offset(sme_A_iter, src_offset), fx.make_layout((TILE_M, TILE_N), (1, TILE_M)))
    smem_ld = fx.make_view(fx.add_offset(fx.get_iter(smem), smem_offset), fx.make_layout((TILE_M, TILE_N), (1, TILE_M)))
    ld = tiled_ld.get_slice(lane_id)
    fx.copy(async_atom, ld.partition_S(src_ld), ld.partition_D(smem_ld))

    ixdl.cp_async_commit_group()
    ixdl.cp_async_wait_group(0)
    fx.gpu.barrier()

    # Phase 2: scalar tiled readback shared -> register -> global.
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
        block=(WARP_SIZE, WARPS_PER_BLOCK, 1),
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
