# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Iluvatar MR SME async copy inside a runtime K-loop (gOffset advancement).

This is the loop variant of ``02-tiledCopy-iluvatar-mr.py``. A single warp walks
``K_TILES`` consecutive 16x16 column tiles of one row band using a runtime
``scf.for`` loop whose carried state is the *source / shared byte offsets*:

    for k, (src_off, smem_off) in fx.range(0, K_TILES, 1, init=[0, 0]):
        <one warp-collective SME async copy of tile k>
        yield [src_off + TILE_N, smem_off + TILE_ELEMS]

The point is the source addressing. The SME descriptor (global base + leading
stride) is loop-invariant, so it is built once and hoisted out of the loop; only
the narrow per-tile offset advances. With the gOffset lowering (design doc
section 10) that per-tile offset is emitted as the hardware ``gOffset`` operand
(a 32-bit add, or folded into the ``goffimm`` immediate) instead of a 64-bit GEP
on the base pointer -- mirroring the ``gOffset += TILE_N`` pattern seen in real
Iluvatar SME GEMM/attention loops.

After the loop the ``K_TILES`` shared tiles are read back to global through the
NoSwizzle physical layout and checked against the source.
"""

import os

os.environ.setdefault("FLYDSL_COMPILE_BACKEND", "iluvatar")
os.environ.setdefault("FLYDSL_RUNTIME_KIND", "iluvatar")
os.environ.setdefault("ARCH", "ivcore11")

import torch  # noqa: E402

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
import flydsl.expr.ixdl as ixdl  # noqa: E402
from flydsl.expr import range_constexpr  # noqa: E402

WARP_SIZE = 64
TILE_M = 16
TILE_N = 16
TILE_ELEMS = TILE_M * TILE_N
ELEMS_PER_LANE = TILE_ELEMS // WARP_SIZE  # 4 f32 per lane on readback

K_TILES = 4  # number of column tiles walked by the runtime loop
M = TILE_M
N = TILE_N * K_TILES
SRC_STRIDE_N = 80  # 80 * 4B = 320B = 5 * 64B, keeps the SME descriptor 64B-aligned
SMEM_BYTES = K_TILES * TILE_ELEMS * 4


@flyc.kernel
def copy_loop_kernel(A: fx.Tensor, B: fx.Tensor):
    lane_id = fx.thread_idx.x

    swizzle = ixdl.SMESwizzle.NoSwizzle

    # NoSwizzle physical shared layout: A(m, n) lands at smem[m*TILE_N + n].
    smem_phys_layout = ixdl.make_sme_shared_layout(swizzle, fx.Float32, major=ixdl.SMEMajor.K)
    # Compact footprint view for the SME load (one contiguous atom unit).
    load_layout = fx.make_layout((TILE_M, TILE_N), (1, TILE_M))

    sme_A = ixdl.make_sme_gmem_tensor(A, leading_stride=SRC_STRIDE_N)
    sme_A_iter = fx.get_iter(sme_A)
    B_iter = fx.get_iter(B)

    smem = fx.make_view(fx.get_dyn_shared(fx.Float32), fx.make_layout(K_TILES * TILE_ELEMS, 1))
    smem_iter = fx.get_iter(smem)

    async_atom = fx.make_copy_atom(ixdl.MRAsyncCp(swizzle), fx.Float32)
    scalar_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)

    tiled_ld = fx.make_tiled_copy_tv(
        async_atom,
        fx.make_layout((1, 1), (1, 1)),
        load_layout,
    )
    tiled_st = fx.make_tiled_copy_tv(
        scalar_atom,
        fx.make_layout((TILE_M, WARP_SIZE // TILE_M), (1, TILE_M)),
        fx.make_layout((1, ELEMS_PER_LANE), (1, 1)),
    )

    # Phase 1: runtime K-loop, carrying the source / shared element offsets. The
    # descriptor is loop-invariant; only these offsets advance (gOffset path).
    init_state = [fx.Int32(0), fx.Int32(0)]
    for _k, state in fx.range(0, K_TILES, 1, init=init_state):
        src_off = fx.Int32(state[0])
        smem_off = fx.Int32(state[1])

        src_ld = fx.make_view(fx.add_offset(sme_A_iter, src_off), load_layout)
        smem_ld = fx.make_view(fx.add_offset(smem_iter, smem_off), load_layout)
        ld = tiled_ld.get_slice(lane_id)
        fx.copy(async_atom, ld.partition_S(src_ld), ld.partition_D(smem_ld))

        next_src = fx.Int32(src_off + fx.Int32(TILE_N))
        next_smem = fx.Int32(smem_off + fx.Int32(TILE_ELEMS))
        yield [next_src, next_smem]

    ixdl.cp_async_commit_group()
    ixdl.cp_async_wait_group(0)
    fx.gpu.barrier()

    # Phase 2: scalar tiled readback shared -> register -> global, tile by tile.
    for tn in range_constexpr(K_TILES):
        smem_off = fx.Int32(tn * TILE_ELEMS)
        dst_off = fx.Int32(tn * TILE_N)
        smem_tile = fx.make_view(fx.add_offset(smem_iter, smem_off), smem_phys_layout)
        dst_tile = fx.make_view(fx.add_offset(B_iter, dst_off), fx.make_layout((TILE_M, TILE_N), (N, 1)))
        st = tiled_st.get_slice(lane_id)
        part_smem = st.partition_S(smem_tile)
        part_dst = st.partition_D(dst_tile)
        frag = fx.make_fragment_like(part_smem)
        fx.copy(scalar_atom, part_smem, frag)
        fx.copy(scalar_atom, frag, part_dst)


@flyc.jit
def asyncCopyLoopIluvatarMR(A: fx.Tensor, B: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    copy_loop_kernel(A, B).launch(
        grid=(1, 1, 1),
        block=(WARP_SIZE, 1, 1),
        smem=SMEM_BYTES,
        stream=stream,
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-compatible Iluvatar device is not available")

    storage = torch.randint(0, 10, (M, SRC_STRIDE_N), dtype=torch.float32).cuda()
    A = storage[:, :N]
    B = torch.zeros(M, N, dtype=torch.float32).cuda()

    asyncCopyLoopIluvatarMR(A, B, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    is_correct = torch.allclose(B, A)
    print("Result correct:", is_correct)
    if not is_correct:
        print("A:", A)
        print("B:", B)


if __name__ == "__main__":
    main()
