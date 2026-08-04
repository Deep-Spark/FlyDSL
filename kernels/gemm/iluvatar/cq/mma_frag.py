# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""CQ HGEMM / IGEMM fragment bring-up: constant A/B + CQMma (no G2S/S2R).

Used by ``examples/03-tiledMma-iluvatar-cq-mma-tile.py`` to validate base vs
FeatureLongMtx tiles. Pipelined HGEMM lives in ``cq/hgemm.py``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.ixdl as ixdl
from kernels.gemm.iluvatar.common import ATOM_K_B16, WARP_SIZE
from kernels.gemm.iluvatar.cq.common import MMA_TILE_BASE, parse_mma_tile


def compile_iluvatar_cq_hgemm_mma_frag(
    *,
    elem_dtype=fx.Float16,
    mma_tile: str = MMA_TILE_BASE,
    a_value=1.0,
    b_value=2.0,
):
    """Compile a CQ HGEMM MMA-correctness kernel with optional long-mtx tile.

    Args:
        elem_dtype: ``fx.Float16`` or ``fx.BFloat16``.
        mma_tile: ``\"base\"`` (default) or FeatureLongMtx ``\"32x32\"`` /
            ``\"16x64\"`` / ``\"64x16\"``.
        a_value / b_value: constant fragment fill values for the correctness path.

    Returns:
        JIT launch ``(A, B, C, stream=...)`` over one MMA atom tile
        ``(atom_m, atom_k)`` / ``(atom_n, atom_k)`` / ``(atom_m, atom_n)``.
    """
    if elem_dtype not in (fx.Float16, fx.BFloat16):
        raise TypeError(f"CQ HGEMM elem_dtype must be Float16 or BFloat16, got {elem_dtype}")

    tile = parse_mma_tile(mma_tile, atom_k=ATOM_K_B16)
    atom_m, atom_n, atom_k = tile.atom_m, tile.atom_n, tile.atom_k
    fx_acc = fx.Float32
    copy_atom_c_factory = fx.UniversalCopy32b

    @flyc.kernel(known_block_size=[WARP_SIZE, 1, 1])
    def cq_hgemm_mma_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x
        gA = fx.make_view(fx.get_iter(A), fx.make_layout((atom_m, atom_k), (atom_k, 1)))
        gB = fx.make_view(fx.get_iter(B), fx.make_layout((atom_n, atom_k), (atom_k, 1)))
        gC = fx.make_view(fx.get_iter(C), fx.make_layout((atom_m, atom_n), (atom_n, 1)))

        mma_atom = fx.make_mma_atom(ixdl.CQMma(atom_m, atom_n, atom_k, elem_dtype, elem_dtype, fx_acc))
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
        cq_hgemm_mma_kernel(A, B, C).launch(grid=(1, 1, 1), block=(WARP_SIZE, 1, 1), stream=stream)

    launch.cq_mma_tile = tile  # type: ignore[attr-defined]
    return launch


# Back-compat alias for the fragment bring-up entry used by the mma-tile example.
compile_iluvatar_cq_hgemm_mma_tile = compile_iluvatar_cq_hgemm_mma_frag

__all__ = [
    "compile_iluvatar_cq_hgemm_mma_frag",
    "compile_iluvatar_cq_hgemm_mma_tile",
]
