"""Multi-block CuTe-style GEMM (256x256x128 FP16) with SME G2S for Iluvatar.

Extends 05-sme-gemm-iluvatar.py to multiple blocks and K-loop:
  - Grid: (M/BM, N/BN) = (16, 16) blocks
  - K-loop: K/BK = 4 iterations (unrolled via range_constexpr)
  - Per iteration: SME G2S → barrier → S2R (SLBLoad) → MMA → barrier
  - Shared memory: ComposedLayout with SwizzleAttr(1, 6, 2)
  - Each block: 1 warp (64 threads), 1 MMAD atom (16x16x16)
  - fx.gemm auto-iterates BK/16 = 2 inner K tiles per iteration

Config: BM=16, BN=16, BK=32, M=256, N=256, K=128.
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

M, N, K = 256, 256, 128
BM, BN, BK = 16, 16, 32
K_ITERS = K // BK


@flyc.kernel
def gemm_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
):
    tid = fx.thread_idx.x
    bid_m = fx.block_idx.x
    bid_n = fx.block_idx.y

    # ── Tile views ──
    tileA = fx.make_tile(BM, BK)
    tileB = fx.make_tile(BN, BK)
    tileC = fx.make_tile(BM, BN)

    gA = fx.zipped_divide(A, tileA)
    gB = fx.zipped_divide(B, tileB)
    gC = fx.zipped_divide(C, tileC)

    bC = fx.slice(gC, (None, (bid_m, bid_n)))

    # ── MMA atom ──
    mma_atom = fx.make_mma_atom(fx.flyixdl.MMAD(16, 16, 16, fx.Float16))
    tiled_mma = fx.make_tiled_mma(
        mma_atom, fx.make_layout((1, 1, 1), (1, 1, 0))
    )
    thr_mma = tiled_mma.thr_slice(tid)

    # ── Shared memory with SME swizzle layout (auto-derived from MMA atom) ──
    sA = fx.flyixdl.make_smem_tile(BM, BK, fx.Float16, for_mma=tiled_mma)
    sB = fx.flyixdl.make_smem_tile(BN, BK, fx.Float16, for_mma=tiled_mma)

    # ── S2R copy (SLBLoad + swizzled SMEM) ──
    s2r_atom = fx.make_copy_atom(fx.flyixdl.SLBLoad(16), fx.Float16)
    tiled_s2r_A = fx.make_tiled_copy_A(s2r_atom, tiled_mma)
    tiled_s2r_B = fx.make_tiled_copy_B(s2r_atom, tiled_mma)

    # ── R2G copy (UniversalCopy) ──
    r2g_atom = fx.make_copy_atom(fx.UniversalCopy(32), fx.Float32)
    tiled_r2g = fx.make_tiled_copy_C(r2g_atom, tiled_mma)

    # ── Partitions on shared memory (fixed mapping, reused across K iters) ──
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

    # ── SME row strides (global matrix, not tile) ──
    stride_a = fx.ixdl._i32_const(K * 2)
    stride_b = fx.ixdl._i32_const(K * 2)

    # ── K-loop (unrolled) ──
    for ki in fx.range_constexpr(K_ITERS):
        bA = fx.slice(gA, (None, (bid_m, ki)))
        bB = fx.slice(gB, (None, (bid_n, ki)))

        # G2S: SME async copy to shared memory
        fx.ixdl.sme_g2s_tile(bA, sA, stride_a)
        fx.ixdl.sme_g2s_tile(bB, sB, stride_b)
        fx.ixdl.barrier()

        # S2R: shared → register (swizzle applied by ExpandCopyComposedSrcLowering)
        fx.copy(s2r_atom, s2r_src_A, s2r_frag_A, pred=None)
        fx.copy(s2r_atom, s2r_src_B, s2r_frag_B, pred=None)

        # MMA: BK/16 = 2 inner K iters handled by fx.gemm
        fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)

        fx.ixdl.barrier()

    # ── Store C ──
    thr_r2g = tiled_r2g.get_slice(tid)
    r2g_dst = thr_r2g.partition_S(bC)
    r2g_frag = thr_r2g.retile(frag_C)
    fx.copy(r2g_atom, r2g_frag, r2g_dst, pred=None)


@flyc.jit
def sme_gemm(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    gemm_kernel(A, B, C).launch(
        grid=(M // BM, N // BN, 1), block=(64, 1, 1), stream=stream
    )


if __name__ == "__main__":
    A = torch.randn(M, K, dtype=torch.float16).cuda()
    B = torch.randn(N, K, dtype=torch.float16).cuda()
    C = torch.zeros(M, N, dtype=torch.float32).cuda()

    sme_gemm(A, B, C, stream=torch.cuda.Stream())

    torch.cuda.synchronize()

    expected = A.float() @ B.float().T
    is_correct = torch.allclose(C, expected, atol=1e-1, rtol=1e-1)
    print("Result correct:", is_correct)
    if not is_correct:
        max_diff = (C - expected).abs().max().item()
        print("Max diff:", max_diff)
        print("Expected[0:4,0:4]:", expected[:4, :4])
        print("Got[0:4,0:4]:", C[:4, :4])
    else:
        max_diff = (C - expected).abs().max().item()
        print(f"Max diff: {max_diff:.6f}")
