# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

# Companion Iluvatar MR tiledMma pipeline HGEMM example. Update this path if
# renamed; doc/comments refer to it as "the tiledMma pipeline HGEMM example".
_TILEDMMA_PIPELINE_HGEMM_EXAMPLE = "examples/03-tiledMma-iluvatar-mr-pipeline-hgemm.py"

_DOC = """Tiled copy using Iluvatar MR SME async copy (tiled-copy paradigm).

This is the Iluvatar MR variant of ``02-tiledCopy.py``. **This example** is the
**teaching** reference for Iluvatar SME async copy and shared-memory layout. For
a tiledMma pipeline HGEMM that makes different trade-offs on the same
primitives, see the companion example
(``{tiledmma_pipeline_hgemm_example}``) and ``kernels.iluvatar_mr_hgemm``.

Design overview
---------------

This example uses the CuTe-style **TiledCopy** flow
(``make_tiled_copy`` + ``partition_S/D`` + ``copy``) rather than hand-rolled
index math. Each CTA owns ``WARPS_PER_BLOCK`` output tiles, one per warp.

**1. Global -> shared (async G2S)**

* **What we do here:** generic ``MRAsyncCp(swizzle)`` + ``make_tiled_copy_tv`` with
  a ``(1,1)`` thread layout. One logical issuer owns the whole 16x16 f32 tile;
  the 64-lane warp cooperation is inside the warp-collective SME instruction.
  We call ``partition_S/D`` once per tile and issue exactly one ``cp_async``.
* **Why not the tiledMma pipeline HGEMM style** (``copy_atom_call`` in a loop over
  SME bricks, as in the companion example)? That path exists for large CTA tiles where
  each warp issues **many** SME bricks per K-stage, A and B need **different**
  swizzle atoms (``Row16b`` vs ``Col``), and work must be **explicitly divided**
  across warps. Here each warp moves only **one** 16x16 tile, so TiledCopy keeps
  the code minimal and matches the FlyDSL layout-algebra docs. A ``(1,1)`` TiledCopy
  issuer per brick would be equivalent but more verbose for no gain in this
  single-tile case.
* **Sync:** ``cp_async_commit_group`` + ``cp_async_wait_group(0)`` + ``barrier``.
  We wait explicitly because this is a **single-phase** copy (no double buffer, no
  compute to hide latency). The tiledMma pipeline HGEMM example omits per-stage
  ``wait`` and relies on stage barriers inside a software-pipelined K-loop instead.

**2. Shared -> register (S2R)**

* **What we do here:** ``make_tiled_copy_tv`` with ``UniversalCopy32b``, a
  hand-built 64-lane thread layout, and the **physical** smem layout from
  ``make_sme_shared_layout``. Fragments are read from swizzled smem and written
  back to global memory in logical row-major order.
* **Why not the tiledMma pipeline HGEMM style** (``make_tiled_copy_A/B(tiled_mma)`` +
  ``retile(frag_A)``)? That path is **MMA-coupled**: register layout is dictated
  by ``MRMma`` / TCU operand requirements. **This example has no MMA** -- the
  destination is plain global memory, not a TCU fragment. Using
  ``make_tiled_copy_A/B`` here would pull in MMA tiling rules that do not apply
  and would not produce a correct logical readback.
* **Physical layout:** swizzle / placement live on the shared **tensor**
  (``make_sme_shared_layout``), not on the CopyAtom. For NoSwizzle + K-major
  this is ``(16,16):(16,1)``, i.e. ``A(m, n)`` lands at ``smem[m*TILE_N + n]``.
  The async-load views use a compact ``(16,16):(1,16)`` footprint only to name
  one contiguous atom unit; the readback **must** view smem through the physical
  layout to recover logical coordinates.

When to use this pattern vs the tiledMma pipeline HGEMM example
-----------------------------------------------------------------

* **Use this example** to learn SME async copy, to debug layout/swizzle issues, or
  for simple elementwise / copy kernels (one tile per warp, explicit wait).
* **Use the tiledMma pipeline HGEMM example / ``kernels.iluvatar_mr_hgemm``** for
  HGEMM and other compute-bound kernels: multi-brick G2S, double-buffered
  K-pipeline, swizzled A/B atoms, and MMA-bound S2R.
"""
__doc__ = _DOC.format(tiledmma_pipeline_hgemm_example=_TILEDMMA_PIPELINE_HGEMM_EXAMPLE)

import os

os.environ.setdefault("FLYDSL_COMPILE_BACKEND", "iluvatar")
os.environ.setdefault("FLYDSL_RUNTIME_KIND", "iluvatar")
os.environ.setdefault("ARCH", "ivcore11")

import torch  # noqa: E402

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
import flydsl.expr.ixdl as ixdl  # noqa: E402
from kernels.iluvatar_mr_common import ATOM_M, ATOM_N, WARP_SIZE  # noqa: E402

WARPS_PER_BLOCK = 4
TILE_M = ATOM_M
TILE_N = ATOM_N
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

    # Generic MRAsyncCp(swizzle) is enough for one NoSwizzle f32 tile. The tiledMma
    # pipeline HGEMM example uses layout-specific Row16b/Col atoms because A/B differ.
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

    # Phase 1 (G2S): TiledCopy + one cp_async per warp tile.
    # Not copy_atom_call/brick loops (tiledMma pipeline HGEMM style): one 16x16
    # tile/warp
    # needs no per-warp work split or A/B-specific Row16b/Col atoms.
    src_ld = fx.make_view(fx.add_offset(sme_A_iter, src_offset), fx.make_layout((TILE_M, TILE_N), (1, TILE_M)))
    smem_ld = fx.make_view(fx.add_offset(fx.get_iter(smem), smem_offset), fx.make_layout((TILE_M, TILE_N), (1, TILE_M)))
    # tiled_ld uses thr_layout (1,1): one logical issuer for the whole 16x16 tile.
    # get_slice(lane_id) is the usual TiledCopy API, but partition_S/D still cover
    # the full tile on every lane; the 64-lane work is inside the warp-collective SME
    # cp_async, not a per-lane split (contrast tiled_st below, thr_layout 64).
    ld = tiled_ld.get_slice(lane_id)
    fx.copy(async_atom, ld.partition_S(src_ld), ld.partition_D(smem_ld))

    # Explicit wait: single-phase copy with no compute pipeline to hide latency.
    # The tiledMma pipeline HGEMM example omits wait_group and uses stage barriers.
    ixdl.cp_async_commit_group()
    ixdl.cp_async_wait_group(0)
    fx.gpu.barrier()

    # Phase 2 (S2R): generic make_tiled_copy_tv -> GMEM, not make_tiled_copy_A/B.
    # Destination is logical global memory, not an MRMma/TCU fragment (the tiledMma
    # pipeline HGEMM example couples S2R to tiled_mma for that case).
    smem_tile = fx.make_view(fx.add_offset(fx.get_iter(smem), smem_offset), smem_phys_layout)
    dst_tile = fx.make_view(fx.add_offset(B_iter, dst_offset), fx.make_layout((TILE_M, TILE_N), (N, 1)))
    # tiled_st uses thr_layout 64: get_slice(lane_id) really does give each lane
    # ELEMS_PER_LANE (=4) f32 via partition_S/D.
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
