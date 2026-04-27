"""Multi-block CuTe-style GEMM (256x256x128 FP16) for Iluvatar ivcore11.

Extends 04-tiledMma-iluvatar.py to multiple blocks with full K accumulation:
  - Grid: (M/BM, N/BN) = (16, 16) blocks
  - K accumulation: handled automatically by fx.gemm — partition_A/B
    have a K dimension that ExpandGemmOpLowering iterates over
  - Each block: 1 warp (64 threads), 1 MMAD atom (16x16x16)
  - Copy: UniversalCopy for G2R (A/B load) and R2G (C store)

Config: BM=16, BN=16, BK=128 (full K), M=256, N=256, K=128.
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

M, N, K = 256, 256, 128
BM, BN = 16, 16


@flyc.kernel
def gemm_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
):
    tid = fx.thread_idx.x
    bid_m = fx.block_idx.x
    bid_n = fx.block_idx.y

    tileA = fx.make_tile(BM, K)
    tileB = fx.make_tile(BN, K)
    tileC = fx.make_tile(BM, BN)

    gA = fx.zipped_divide(A, tileA)
    gB = fx.zipped_divide(B, tileB)
    gC = fx.zipped_divide(C, tileC)

    bA = fx.slice(gA, (None, (bid_m, 0)))
    bB = fx.slice(gB, (None, (bid_n, 0)))
    bC = fx.slice(gC, (None, (bid_m, bid_n)))

    # ── MMA atom ──
    mma_atom = fx.make_mma_atom(fx.flyixdl.MMAD(16, 16, 16, fx.Float16))
    tiled_mma = fx.make_tiled_mma(
        mma_atom, fx.make_layout((1, 1, 1), (1, 1, 0))
    )
    thr_mma = tiled_mma.thr_slice(tid)

    # ── Copy atoms ──
    copy_atom_ab = fx.make_copy_atom(fx.UniversalCopy(16), fx.Float16)
    tiled_copy_A = fx.make_tiled_copy_A(copy_atom_ab, tiled_mma)
    tiled_copy_B = fx.make_tiled_copy_B(copy_atom_ab, tiled_mma)

    copy_atom_c = fx.make_copy_atom(fx.UniversalCopy(32), fx.Float32)
    tiled_copy_C = fx.make_tiled_copy_C(copy_atom_c, tiled_mma)

    # ── Partitions ──
    thr_copy_A = tiled_copy_A.get_slice(tid)
    thr_copy_B = tiled_copy_B.get_slice(tid)
    thr_copy_C = tiled_copy_C.get_slice(tid)

    copy_src_A = thr_copy_A.partition_S(bA)
    copy_src_B = thr_copy_B.partition_S(bB)
    copy_dst_C = thr_copy_C.partition_S(bC)

    partition_A = thr_mma.partition_A(bA)
    partition_B = thr_mma.partition_B(bB)
    partition_C = thr_mma.partition_C(bC)

    frag_A = thr_mma.make_fragment_A(partition_A)
    frag_B = thr_mma.make_fragment_B(partition_B)
    frag_C = thr_mma.make_fragment_C(partition_C)

    copy_frag_A = thr_copy_A.retile(frag_A)
    copy_frag_B = thr_copy_B.retile(frag_B)
    copy_frag_C = thr_copy_C.retile(frag_C)

    # ── Load A, B from global → fragments ──
    fx.copy(copy_atom_ab, copy_src_A, copy_frag_A, pred=None)
    fx.copy(copy_atom_ab, copy_src_B, copy_frag_B, pred=None)

    # ── GEMM: fx.gemm auto-iterates over K dimension ──
    fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)

    # ── Store C ──
    fx.copy(copy_atom_c, copy_frag_C, copy_dst_C, pred=None)


@flyc.jit
def tiledMma(
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

    tiledMma(A, B, C, stream=torch.cuda.Stream())

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
