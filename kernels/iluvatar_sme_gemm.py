#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""ivcore11 (Iluvatar MR) CuTe-style FP16 SME GEMM.

Computes ``C[M, N] = A[M, K] @ B[N, K]^T`` with FP16 inputs and an FP32
accumulator/output, using the three-layer FlyIXDL model:

1. layout  : ``fx.flyixdl.make_smem_tile`` builds ``ComposedLayout = Swizzle . Offset . Outer``
2. transfer: ``fx.ixdl.sme_g2s_tile`` issues cooperative G->S ``ixdl.cp.async``
3. compute : ``SLBLoad`` + ``fx.copy`` (S->R) + ``fx.gemm`` (MMAD 16x16x16) + ``UniversalCopy`` (R->G)

Phase A is single-warp / single-block (``BM=BN=16, BK=32``, ``block=(64,1,1)``).
Setting ``M/N/K`` larger automatically enables the multi-block grid + K-loop
(Phase B); the G2S stride uses the global ``K*2`` bytes (f16), not ``BK*2``.

Run on device::

    # one canonical entry (pins CoreX userspace + iluvatar backend + PYTHONPATH):
    source scripts/env_iluvatar.sh
    python3 kernels/iluvatar_sme_gemm.py
    # or simply:  bash scripts/run_iluvatar_device.sh kernel

``ARCH`` / ``FLYDSL_COMPILE_BACKEND`` / ``FLYDSL_RUNTIME_KIND`` default to the
ivcore11 / iluvatar values below (see :func:`_bootstrap_iluvatar_env`); export
them yourself to override.
"""

import os


def _bootstrap_iluvatar_env():
    """Pin the iluvatar compile/runtime backend before flydsl resolves it.

    ``FLYDSL_COMPILE_BACKEND`` defaults to ``rocm`` inside flydsl, so the ixdl
    lowering pipeline (``convert-fly-to-ixdl``) is only selected when these are
    set. ``setdefault`` keeps any explicit caller/CI override intact.
    """
    os.environ.setdefault("ARCH", "ivcore11")
    os.environ.setdefault("FLYDSL_COMPILE_BACKEND", "iluvatar")
    os.environ.setdefault("FLYDSL_RUNTIME_KIND", "iluvatar")


_bootstrap_iluvatar_env()

import flydsl.compiler as flyc
import flydsl.expr as fx

# MMAD hardware shape is fixed to 16x16x16 on ivcore11.
MMA_M = 16
MMA_N = 16
MMA_K = 16


def build_sme_gemm(M, N, K, *, BM=16, BN=16, BK=32):
    """Return a jitted ``sme_gemm(A, B, C, stream)`` for the given problem size."""
    assert BM % MMA_M == 0 and BN % MMA_N == 0
    assert BK % MMA_K == 0 and BK % 2 == 0
    K_ITERS = K // BK

    # Dynamic shared bytes for the two f16 SLB tiles (sA at 0, sB at BM*BK).
    # get_dyn_shared does NOT register a size, so the launch must request it;
    # without this the @__dynamic_shared__ buffer is 0 bytes and the SME
    # cp.async / SLBLoad operate on empty shared memory (frags read as zero).
    SMEM_BYTES = (BM * BK + BN * BK) * 2

    @flyc.kernel
    def gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x
        bid_m = fx.block_idx.x
        bid_n = fx.block_idx.y

        tileA = fx.make_tile(BM, BK)
        tileB = fx.make_tile(BN, BK)
        tileC = fx.make_tile(BM, BN)

        gA = fx.zipped_divide(A, tileA)
        gB = fx.zipped_divide(B, tileB)
        gC = fx.zipped_divide(C, tileC)
        bC = fx.slice(gC, (None, (bid_m, bid_n)))

        mma_atom = fx.make_mma_atom(fx.flyixdl.MMAD(MMA_M, MMA_N, MMA_K, fx.Float16))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (1, 1, 0)))
        thr_mma = tiled_mma.thr_slice(tid)

        # Layer 1: swizzled SLB tiles (share one dynamic-smem buffer via offset).
        sA = fx.flyixdl.make_smem_tile(BM, BK, fx.Float16, for_mma=tiled_mma)
        sB = fx.flyixdl.make_smem_tile(
            BN, BK, fx.Float16, for_mma=tiled_mma, base_offset_elems=BM * BK
        )

        # Layer 3: S->R via SLBLoad and R->G via UniversalCopy.
        s2r_atom = fx.make_copy_atom(fx.flyixdl.SLBLoad(16), fx.Float16)
        tiled_s2r_A = fx.make_tiled_copy_A(s2r_atom, tiled_mma)
        tiled_s2r_B = fx.make_tiled_copy_B(s2r_atom, tiled_mma)

        r2g_atom = fx.make_copy_atom(fx.UniversalCopy(32), fx.Float32)
        tiled_r2g = fx.make_tiled_copy_C(r2g_atom, tiled_mma)

        thr_s2r_A = tiled_s2r_A.get_slice(tid)
        thr_s2r_B = tiled_s2r_B.get_slice(tid)
        s2r_src_A = thr_s2r_A.partition_S(sA)
        s2r_src_B = thr_s2r_B.partition_S(sB)

        # make_fragment_* partitions the tile internally; pass the raw SLB/global
        # tiles (NOT a thr_mma.partition_* result) -- double partitioning yields a
        # degenerate fragment layout for the asymmetric B operand.
        frag_A = thr_mma.make_fragment_A(sA)
        frag_B = thr_mma.make_fragment_B(sB)
        frag_C = thr_mma.make_fragment_C(bC)

        s2r_frag_A = thr_s2r_A.retile(frag_A)
        s2r_frag_B = thr_s2r_B.retile(frag_B)

        # Zero the accumulator: fx.gemm does D = A*B + C, and make_fragment_C
        # leaves the register fragment uninitialized (lowers to ub.poison ->
        # NaN/garbage on device). Must clear before the K-loop accumulation.
        frag_C.fill(0.0)

        # Multi-block invariant: G2S byte stride is the GLOBAL K*2 for f16.
        stride_a = fx.ixdl._i32_const(K * 2)
        stride_b = fx.ixdl._i32_const(K * 2)

        for ki in fx.range_constexpr(K_ITERS):
            bA = fx.slice(gA, (None, (bid_m, ki)))
            bB = fx.slice(gB, (None, (bid_n, ki)))

            # Layer 2: cooperative G->S.
            fx.ixdl.sme_g2s_tile(bA, sA, stride_a)
            fx.ixdl.sme_g2s_tile(bB, sB, stride_b)
            fx.ixdl.cp_async_wait_group(0)
            fx.ixdl.barrier()

            fx.copy(s2r_atom, s2r_src_A, s2r_frag_A, pred=None)
            fx.copy(s2r_atom, s2r_src_B, s2r_frag_B, pred=None)
            fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)

            fx.ixdl.barrier()

        thr_r2g = tiled_r2g.get_slice(tid)
        r2g_dst = thr_r2g.partition_S(bC)
        r2g_frag = thr_r2g.retile(frag_C)
        fx.copy(r2g_atom, r2g_frag, r2g_dst, pred=None)

    @flyc.jit
    def sme_gemm(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        gemm_kernel(A, B, C).launch(
            grid=(M // BM, N // BN, 1),
            block=(64, 1, 1),
            smem=SMEM_BYTES,
            stream=stream,
        )

    return sme_gemm


def _main():
    import torch

    M, N, K = 16, 16, 32  # Phase A: single warp, single block.
    sme_gemm = build_sme_gemm(M, N, K)

    A = torch.randn(M, K, dtype=torch.float16).cuda()
    B = torch.randn(N, K, dtype=torch.float16).cuda()
    C = torch.zeros(M, N, dtype=torch.float32).cuda()

    sme_gemm(A, B, C, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    expected = A.float() @ B.float().T
    max_diff = (C - expected).abs().max().item()
    print("Result correct:", torch.allclose(C, expected, atol=1e-1, rtol=1e-1))
    print(f"Max diff: {max_diff:.6f}")


if __name__ == "__main__":
    _main()
