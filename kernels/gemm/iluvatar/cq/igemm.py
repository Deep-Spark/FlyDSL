# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""CQ IGEMM bring-up: optional ``mma_tile`` over CQMma s8 long-mtx / base tile.

Single-warp fragment MMA correctness path (same contract as ``cq/hgemm.py``).
Default ``mma_tile=\"base\"`` is 16x16x32 s8->i32. u8 / FP8 atoms are covered by
FileCheck and ``CQMma``; this harness stays on signed i8.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.ixdl as ixdl
from kernels.gemm.iluvatar.common import ATOM_K_B8, WARP_SIZE
from kernels.gemm.iluvatar.cq.common import MMA_TILE_BASE, parse_mma_tile


def compile_iluvatar_cq_igemm(
    *,
    mma_tile: str = MMA_TILE_BASE,
    a_value: int = 1,
    b_value: int = 2,
):
    """Compile a CQ IGEMM MMA-correctness kernel with optional long-mtx tile.

    Args:
        mma_tile: ``\"base\"`` (default) or FeatureLongMtx ``\"32x32\"`` /
            ``\"16x64\"`` / ``\"64x16\"`` (K stays 32).
        a_value / b_value: constant int8 fragment fills.

    Returns:
        JIT launch ``(A, B, C, stream=...)`` over one s8 CQMma atom tile.
    """
    tile = parse_mma_tile(mma_tile, atom_k=ATOM_K_B8)
    atom_m, atom_n, atom_k = tile.atom_m, tile.atom_n, tile.atom_k
    fx_dtype = fx.Int8
    fx_acc = fx.Int32
    copy_atom_c_factory = fx.UniversalCopy32b

    @flyc.kernel(known_block_size=[WARP_SIZE, 1, 1])
    def cq_igemm_mma_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x
        gA = fx.make_view(fx.get_iter(A), fx.make_layout((atom_m, atom_k), (atom_k, 1)))
        gB = fx.make_view(fx.get_iter(B), fx.make_layout((atom_n, atom_k), (atom_k, 1)))
        gC = fx.make_view(fx.get_iter(C), fx.make_layout((atom_m, atom_n), (atom_n, 1)))

        mma_atom = fx.make_mma_atom(ixdl.CQMma(atom_m, atom_n, atom_k, fx_dtype, fx_dtype, fx_acc))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (1, 1, 1)))
        thr_mma = tiled_mma.thr_slice(tid)

        frag_A = thr_mma.make_fragment_A(gA)
        frag_B = thr_mma.make_fragment_B(gB)
        frag_C = thr_mma.make_fragment_C(gC)

        frag_A.fill(a_value)
        frag_B.fill(b_value)
        frag_C.fill(0)
        fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)

        copy_atom_c = fx.make_copy_atom(copy_atom_c_factory(), fx_acc)
        tiled_copy_C = fx.make_tiled_copy_C(copy_atom_c, tiled_mma)
        thr_copy_C = tiled_copy_C.get_slice(tid)
        fx.copy(copy_atom_c, thr_copy_C.retile(frag_C), thr_copy_C.partition_D(gC), pred=None)

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        cq_igemm_mma_kernel(A, B, C).launch(grid=(1, 1, 1), block=(WARP_SIZE, 1, 1), stream=stream)

    launch.cq_mma_tile = tile  # type: ignore[attr-defined]
    return launch


__all__ = ["compile_iluvatar_cq_igemm"]
