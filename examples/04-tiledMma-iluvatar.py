"""CuTe-style TiledMMA example for Iluvatar ivcore11.

Demonstrates the FlyIXDL dialect atoms:
  - fx.flyixdl.MMAD(16, 16, 16, fx.Float16)  — TCU MMA atom
  - fx.flyixdl.SLBLoad(32)                    — S2R copy atom

This example performs a single-tile GEMM (16x16x16, FP16->FP32)
using the CuTe partition / copy / gemm flow.
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

block_m = 16
block_n = 16
block_k = 16


@flyc.kernel
def gemm_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
):
    tid = fx.thread_idx.x

    tileA = fx.make_tile(block_m, block_k)
    tileB = fx.make_tile(block_n, block_k)
    tileC = fx.make_tile(block_m, block_n)

    bA = fx.zipped_divide(A, tileA)
    bB = fx.zipped_divide(B, tileB)
    bC = fx.zipped_divide(C, tileC)

    bA = fx.slice(bA, (None, 0))
    bB = fx.slice(bB, (None, 0))
    bC = fx.slice(bC, (None, 0))

    mma_atom = fx.make_mma_atom(fx.flyixdl.MMAD(16, 16, 16, fx.Float16))
    tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (1, 1, 0)))
    thr_mma = tiled_mma.thr_slice(tid)

    copy_atom_ab = fx.make_copy_atom(fx.UniversalCopy(16), fx.Float16)
    tiled_copy_A = fx.make_tiled_copy_A(copy_atom_ab, tiled_mma)
    tiled_copy_B = fx.make_tiled_copy_B(copy_atom_ab, tiled_mma)

    copy_atom_c = fx.make_copy_atom(fx.UniversalCopy(32), fx.Float32)
    tiled_copy_C = fx.make_tiled_copy_C(copy_atom_c, tiled_mma)

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

    fx.copy(copy_atom_ab, copy_src_A, copy_frag_A, pred=None)
    fx.copy(copy_atom_ab, copy_src_B, copy_frag_B, pred=None)

    fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)

    fx.copy(copy_atom_c, copy_frag_C, copy_dst_C, pred=None)


@flyc.jit
def tiledMma(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    gemm_kernel(A, B, C).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


if __name__ == "__main__":
    M, N, K = block_m, block_n, block_k
    A = torch.randn(M, K, dtype=torch.float16).cuda()
    B = torch.randn(N, K, dtype=torch.float16).cuda()
    C = torch.zeros(M, N, dtype=torch.float32).cuda()

    tiledMma(A, B, C, stream=torch.cuda.Stream())

    torch.cuda.synchronize()

    expected = (A.float() @ B.float().T)
    is_correct = torch.allclose(C, expected, atol=1e-2, rtol=1e-2)
    print("Result correct:", is_correct)
    if not is_correct:
        print("Max diff:", (C - expected).abs().max().item())
        print("Expected:", expected[:4, :4])
        print("Got:", C[:4, :4])
