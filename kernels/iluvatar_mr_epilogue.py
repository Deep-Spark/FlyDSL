# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Reusable Iluvatar MR HGEMM epilogue helpers.

Three store paths (each exposed as its own function):

* ``mr_hgemm_epilogue_store_shfl`` — warp shuffle + packed i32 store (fp16)
* ``mr_hgemm_epilogue_store_tiled`` — ``trunc_f`` + ``make_tiled_copy_C`` (fp16)
* ``mr_hgemm_epilogue_store_read_c_accum`` — ``make_tiled_copy_C`` (fp32)

``mr_hgemm_epilogue_store`` dispatches on ``store_mode`` for production kernels.
"""

import flydsl.expr as fx
from flydsl.expr.typing import Vector as Vec

from kernels.iluvatar_mr_common import ATOM_M, ATOM_N, TCU_LANE_COLS, WARP_SIZE

EPILOGUE_STORE_SHFL = "shfl"
EPILOGUE_STORE_TILED = "tiled"
EPILOGUE_STORE_READ_C_ACCUM = "read_c_accum"


def mr_hgemm_epilogue_store_shfl(
    *,
    lane_id,
    accs,
    gC_warp,
    c_global_n: int,
    warp_atoms_m: int,
    warp_atoms_n: int,
):
    """fp16 shuffle/packed-i32 store (``no_c_read`` + ``shfl``)."""
    c_warp_n = ATOM_N * warp_atoms_n

    lane_col = lane_id % fx.Int32(TCU_LANE_COLS)
    lane_row = lane_id // fx.Int32(TCU_LANE_COLS)
    lane_voffset = lane_row * fx.Int32(c_global_n // 2) + lane_col
    lane_select0 = lane_row * fx.Int32(TCU_LANE_COLS) + (lane_col * fx.Int32(2)) % fx.Int32(TCU_LANE_COLS)
    lane_select1 = lane_select0 + fx.Int32(1)
    lane_em = lane_col // fx.Int32(8)
    width_i32 = fx.Int32(WARP_SIZE)
    mask16 = fx.Int32(0xFFFF)
    mask_hi = fx.Int32(0xFFFF0000)

    c_warp_ptr = fx.get_iter(gC_warp)
    c_byte_ptr = fx.recast_iter(
        fx.PointerType.get(fx.Int8.ir_type, c_warp_ptr.memspace),
        c_warp_ptr,
    )

    for im in fx.range_constexpr(warp_atoms_m):
        mi = im * TCU_LANE_COLS
        for ei in fx.range_constexpr(4):
            for ni in fx.range_constexpr(0, c_warp_n, TCU_LANE_COLS * 2):
                jn0 = ni // TCU_LANE_COLS
                jn1 = jn0 + 1
                tile_half_soffset = fx.Int32((mi + ei * 4) * c_global_n + ni)

                f32_0 = Vec(accs[im][jn0].load())[ei]
                f32_1 = Vec(accs[im][jn1].load())[ei]
                h0 = fx.arith.trunc_f(fx.Float16.ir_type, f32_0)
                h1 = fx.arith.trunc_f(fx.Float16.ir_type, f32_1)
                hval_i32 = Vec(Vec.from_elements([h0, h1], fx.Float16)).bitcast(fx.Int32)[0]

                hvall = hval_i32.shuffle_idx(lane_select0, width_i32)
                hvalh = hval_i32.shuffle_idx(lane_select1, width_i32)
                val0 = (hvall & mask16) | (hvalh << fx.Int32(16))
                val1 = hvall.shrui(fx.Int32(16)) | (hvalh & mask_hi)
                val = fx.arith.select(
                    fx.arith.cmpi(fx.arith.CmpIPredicate.ne, lane_em, fx.Int32(0)),
                    val1,
                    val0,
                )

                store_byte_off = lane_voffset * fx.Int32(4) + tile_half_soffset * fx.Int32(2)
                store_ptr = fx.recast_iter(
                    fx.PointerType.get(fx.Int32.ir_type, c_warp_ptr.memspace),
                    fx.add_offset(c_byte_ptr, fx.make_int_tuple(store_byte_off)),
                )
                fx.ptr_store(val, store_ptr)


def mr_hgemm_epilogue_store_tiled(
    *,
    lane_id,
    accs,
    gC_warp,
    tiled_mma,
    warp_atoms_m: int,
    warp_atoms_n: int,
):
    """fp16 tiled ``make_tiled_copy_C`` store (``no_c_read`` + ``tiled``)."""
    gC_atoms = fx.flat_divide(gC_warp, (ATOM_M, ATOM_N))

    copy_atom_c_f16 = fx.make_copy_atom(fx.UniversalCopy16b(), fx.Float16)
    tiled_copy_c_f16 = fx.make_tiled_copy_C(copy_atom_c_f16, tiled_mma)
    thr_copy_c_f16 = tiled_copy_c_f16.get_slice(lane_id)
    for im in fx.range_constexpr(warp_atoms_m):
        for jn in fx.range_constexpr(warp_atoms_n):
            c_tile = fx.slice(gC_atoms, (None, None, im, jn))
            acc = accs[im][jn]
            frag_f16 = fx.make_fragment_like(acc, fx.Float16.ir_type)
            acc_vec = acc.load()
            f16_vec = fx.arith.trunc_f(
                fx.T.VectorType.get(list(acc_vec.type.shape), fx.T.f16()),
                acc_vec,
            )
            frag_f16.store(f16_vec)
            fx.copy(
                copy_atom_c_f16,
                thr_copy_c_f16.retile(frag_f16),
                thr_copy_c_f16.partition_S(c_tile),
                pred=None,
            )


def mr_hgemm_epilogue_store_read_c_accum(
    *,
    lane_id,
    accs,
    gC_warp,
    tiled_mma,
    warp_atoms_m: int,
    warp_atoms_n: int,
):
    """fp32 tiled ``make_tiled_copy_C`` store (``read_c_accum`` epilogue)."""
    gC_atoms = fx.flat_divide(gC_warp, (ATOM_M, ATOM_N))

    copy_atom_c_f32 = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    tiled_copy_c_f32 = fx.make_tiled_copy_C(copy_atom_c_f32, tiled_mma)
    thr_copy_c_f32 = tiled_copy_c_f32.get_slice(lane_id)
    for im in fx.range_constexpr(warp_atoms_m):
        for jn in fx.range_constexpr(warp_atoms_n):
            c_tile = fx.slice(gC_atoms, (None, None, im, jn))
            acc = accs[im][jn]
            fx.copy(
                copy_atom_c_f32,
                thr_copy_c_f32.retile(acc),
                thr_copy_c_f32.partition_S(c_tile),
                pred=None,
            )


def mr_hgemm_epilogue_store(
    *,
    store_mode: str,
    lane_id,
    accs,
    gC_warp,
    c_global_n: int,
    tiled_mma,
    warp_atoms_m: int,
    warp_atoms_n: int,
):
    """Dispatch to the selected MR HGEMM C-store epilogue."""
    if fx.const_expr(store_mode == EPILOGUE_STORE_SHFL):
        mr_hgemm_epilogue_store_shfl(
            lane_id=lane_id,
            accs=accs,
            gC_warp=gC_warp,
            c_global_n=c_global_n,
            warp_atoms_m=warp_atoms_m,
            warp_atoms_n=warp_atoms_n,
        )
    elif fx.const_expr(store_mode == EPILOGUE_STORE_TILED):
        mr_hgemm_epilogue_store_tiled(
            lane_id=lane_id,
            accs=accs,
            gC_warp=gC_warp,
            tiled_mma=tiled_mma,
            warp_atoms_m=warp_atoms_m,
            warp_atoms_n=warp_atoms_n,
        )
    else:
        mr_hgemm_epilogue_store_read_c_accum(
            lane_id=lane_id,
            accs=accs,
            gC_warp=gC_warp,
            tiled_mma=tiled_mma,
            warp_atoms_m=warp_atoms_m,
            warp_atoms_n=warp_atoms_n,
        )
