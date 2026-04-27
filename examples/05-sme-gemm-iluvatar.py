"""CuTe-style GEMM (16x16x32 FP16) for Iluvatar ivcore11 with SME G2S.

Demonstrates CuTe layout system integration with Iluvatar's SME async
copy and SLB swizzle:
  - G2S: explicit SME helper (block-collective, sme_load_16x1b64_rowxfb16)
  - Shared memory: ComposedLayout with SwizzleAttr(1, 6, 2) encoding
    the rowxfb16 XOR swizzle in the layout system
  - S2R: CuTe fx.copy with SLBLoad(16) — swizzle applied automatically
    by layoutCrd2Idx during partition
  - MMA: CuTe fx.gemm with MMAD(16,16,16,f16) — two K iterations
  - R2G: CuTe fx.copy with UniversalCopy(32) — proven path

Config: BM=16, BN=16, BK=32, 64 threads (1 warp), 2048 bytes SMEM.
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

BM, BN, BK = 16, 16, 32


@flyc.kernel
def gemm_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
):
    tid = fx.thread_idx.x

    # ── CuTe tile views (same pattern as 04-tiledMma-iluvatar.py) ──
    tileA = fx.make_tile(BM, BK)
    tileB = fx.make_tile(BN, BK)
    tileC = fx.make_tile(BM, BN)

    bA = fx.slice(fx.zipped_divide(A, tileA), (None, 0))
    bB = fx.slice(fx.zipped_divide(B, tileB), (None, 0))
    bC = fx.slice(fx.zipped_divide(C, tileC), (None, 0))

    # ── MMA atom: carries the A/B/C thread-value layouts that the
    #   make_tiled_copy_{A,B} below derive their thread mapping from.
    mma_atom = fx.make_mma_atom(fx.flyixdl.MMAD(16, 16, 16, fx.Float16))
    tiled_mma = fx.make_tiled_mma(
        mma_atom, fx.make_layout((1, 1, 1), (1, 1, 0))
    )
    thr_mma = tiled_mma.thr_slice(tid)

    # ── Shared memory with SME swizzle layout (auto-derived from MMA atom) ──
    sA = fx.flyixdl.make_smem_tile(BM, BK, fx.Float16, for_mma=tiled_mma)
    sB = fx.flyixdl.make_smem_tile(BN, BK, fx.Float16, for_mma=tiled_mma)

    # ── S2R copy (SLBLoad + swizzled SMEM = correct addresses) ──
    s2r_atom = fx.make_copy_atom(fx.flyixdl.SLBLoad(16), fx.Float16)
    tiled_s2r_A = fx.make_tiled_copy_A(s2r_atom, tiled_mma)
    tiled_s2r_B = fx.make_tiled_copy_B(s2r_atom, tiled_mma)

    # ── R2G copy (UniversalCopy, proven) ──
    r2g_atom = fx.make_copy_atom(fx.UniversalCopy(32), fx.Float32)
    tiled_r2g = fx.make_tiled_copy_C(r2g_atom, tiled_mma)

    # ── Partitions ──
    thr_s2r_A = tiled_s2r_A.get_slice(tid)
    thr_s2r_B = tiled_s2r_B.get_slice(tid)
    s2r_src_A = thr_s2r_A.partition_S(sA)
    s2r_src_B = thr_s2r_B.partition_S(sB)

    partition_A = thr_mma.partition_A(sA)
    partition_B = thr_mma.partition_B(sB)
    partition_C = thr_mma.partition_C(bC)

    frag_A = thr_mma.make_fragment_A(partition_A)
    frag_B = thr_mma.make_fragment_B(partition_B)
    frag_C = thr_mma.make_fragment_C(partition_C)

    s2r_frag_A = thr_s2r_A.retile(frag_A)
    s2r_frag_B = thr_s2r_B.retile(frag_B)

    thr_r2g = tiled_r2g.get_slice(tid)
    r2g_dst = thr_r2g.partition_S(bC)
    r2g_frag = thr_r2g.retile(frag_C)

    # ── G2S: explicit SME (block-collective, not per-thread) ──
    stride_a = fx.ixdl._i32_const(BK * 2)  # row stride = 32 f16 = 64 bytes
    stride_b = fx.ixdl._i32_const(BK * 2)
    fx.ixdl.sme_g2s_tile(bA, sA, stride_a)
    fx.ixdl.sme_g2s_tile(bB, sB, stride_b)
    fx.ixdl.barrier()

    # ── S2R + MMA (CuTe) ──
    fx.copy(s2r_atom, s2r_src_A, s2r_frag_A, pred=None)
    fx.copy(s2r_atom, s2r_src_B, s2r_frag_B, pred=None)
    fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)

    # ── R2G (CuTe, UniversalCopy) ──
    fx.copy(r2g_atom, r2g_frag, r2g_dst, pred=None)


@flyc.jit
def sme_gemm(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    gemm_kernel(A, B, C).launch(
        grid=(1, 1, 1), block=(64, 1, 1), stream=stream
    )


if __name__ == "__main__":
    M, N, K = BM, BN, BK
    A = torch.randn(M, K, dtype=torch.float16).cuda()
    B = torch.randn(N, K, dtype=torch.float16).cuda()
    C = torch.zeros(M, N, dtype=torch.float32).cuda()

    sme_gemm(A, B, C, stream=torch.cuda.Stream())

    torch.cuda.synchronize()

    expected = A.float() @ B.float().T
    is_correct = torch.allclose(C, expected, atol=1e-2, rtol=1e-2)
    print("Result correct:", is_correct)
    if not is_correct:
        print("Max diff:", (C - expected).abs().max().item())
        print("Expected:", expected[:4, :4])
        print("Got:", C[:4, :4])
