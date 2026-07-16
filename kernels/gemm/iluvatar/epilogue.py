# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Reusable Iluvatar MR HGEMM epilogue helpers.

  mr_hgemm_epilogue_store_shfl — f16/bf16 via warp shuffle + packed i32 store
  mr_hgemm_epilogue_store_tiled — f16/bf16 via trunc_f + make_tiled_copy_C
  mr_hgemm_epilogue_store_read_c_accum — fp32 make_tiled_copy_C

mr_hgemm_epilogue_store dispatches on store_mode.
"""

import flydsl.expr as fx
from flydsl.expr.ixdl import byte_permute, stp_vs_b32
from flydsl.expr.typing import Vector as Vec
from kernels.gemm.iluvatar.common import WARP_SIZE
from kernels.gemm.iluvatar.mr.common import ATOM_M, ATOM_N, TCU_LANE_COLS

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
    out_dtype=fx.Float16,
):
    """f16/bf16 shuffle/packed-i32 store (``EPILOGUE_STORE_SHFL``). Parameters: see ``mr_hgemm_epilogue_store``."""
    c_warp_n = ATOM_N * warp_atoms_n

    lane_col = lane_id % fx.Int32(TCU_LANE_COLS)
    lane_row = lane_id // fx.Int32(TCU_LANE_COLS)
    lane_voffset = lane_row * fx.Int32(c_global_n // 2) + lane_col
    lane_select0 = lane_row * fx.Int32(TCU_LANE_COLS) + (lane_col * fx.Int32(2)) % fx.Int32(TCU_LANE_COLS)
    lane_select1 = lane_select0 + fx.Int32(1)
    lane_em = lane_col // fx.Int32(8)
    width_i32 = fx.Int32(WARP_SIZE)
    mask_lo = fx.Int32(0xFFFF)
    mask_hi = fx.Int32(0xFFFF0000)

    c_warp_ptr = fx.get_iter(gC_warp)
    c_byte_ptr = fx.recast_iter(
        fx.PointerType.get(fx.Int8.ir_type, c_warp_ptr.memspace),
        c_warp_ptr,
    )

    for mma_m in fx.range_constexpr(warp_atoms_m):
        phys_m = mma_m * TCU_LANE_COLS
        for ei in fx.range_constexpr(4):
            for phys_n in fx.range_constexpr(0, c_warp_n, TCU_LANE_COLS * 2):
                mma_n0 = phys_n // TCU_LANE_COLS
                mma_n1 = mma_n0 + 1
                tile_half_soffset = fx.Int32((phys_m + ei * 4) * c_global_n + phys_n)

                f32_0 = Vec(accs[mma_m][mma_n0].load())[ei]
                f32_1 = Vec(accs[mma_m][mma_n1].load())[ei]
                h0 = f32_0.to(out_dtype)
                h1 = f32_1.to(out_dtype)
                hval_i32 = Vec(Vec.from_elements([h0, h1], out_dtype)).bitcast(fx.Int32)[0]

                hvall = hval_i32.shuffle_idx(lane_select0, width_i32)
                hvalh = hval_i32.shuffle_idx(lane_select1, width_i32)
                val0 = (hvall & mask_lo) | (hvalh << fx.Int32(16))
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
    out_dtype=fx.Float16,
):
    """f16/bf16 ``make_tiled_copy_C`` store (``EPILOGUE_STORE_TILED``). Parameters: see ``mr_hgemm_epilogue_store``."""
    gC_atoms = fx.flat_divide(gC_warp, (ATOM_M, ATOM_N))

    copy_atom_c = fx.make_copy_atom(fx.UniversalCopy16b(), out_dtype)
    tiled_copy_c = fx.make_tiled_copy_C(copy_atom_c, tiled_mma)
    thr_copy_c = tiled_copy_c.get_slice(lane_id)
    for mma_m in fx.range_constexpr(warp_atoms_m):
        for mma_n in fx.range_constexpr(warp_atoms_n):
            c_tile = fx.slice(gC_atoms, (None, None, mma_m, mma_n))
            acc = accs[mma_m][mma_n]
            frag = fx.make_fragment_like(acc, out_dtype.ir_type)
            frag.store(Vec(acc.load()).to(out_dtype))
            fx.copy(
                copy_atom_c,
                thr_copy_c.retile(frag),
                thr_copy_c.partition_S(c_tile),
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
    """fp32 ``make_tiled_copy_C`` store (``EPILOGUE_STORE_READ_C_ACCUM``).

    Parameters: see ``mr_hgemm_epilogue_store``.
    """
    gC_atoms = fx.flat_divide(gC_warp, (ATOM_M, ATOM_N))

    copy_atom_c_f32 = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    tiled_copy_c_f32 = fx.make_tiled_copy_C(copy_atom_c_f32, tiled_mma)
    thr_copy_c_f32 = tiled_copy_c_f32.get_slice(lane_id)
    for mma_m in fx.range_constexpr(warp_atoms_m):
        for mma_n in fx.range_constexpr(warp_atoms_n):
            c_tile = fx.slice(gC_atoms, (None, None, mma_m, mma_n))
            acc = accs[mma_m][mma_n]
            fx.copy(
                copy_atom_c_f32,
                thr_copy_c_f32.retile(acc),
                thr_copy_c_f32.partition_S(c_tile),
                pred=None,
            )


def mr_igemm_epilogue_store_i32(
    *,
    lane_id,
    accs,
    gC_warp,
    c_global_n: int,
    warp_atoms_m: int,
    warp_atoms_n: int,
):
    """i32 store via ``stp_vs_b32`` (``llvm.bi.stp.vs.i32``).

    Bakes lane ``voffset`` once and issues each element as
    ``stp_vs_b32(val, base, voffset, soffset)`` with constexpr ``soffset`` tile
    byte offsets. Avoids combined V-offset ``store`` + GEP which shows higher
    Memory Throttle on ivcore11.

    For each M-atom, walk ``ei`` (row group) outer and ``jn`` (N-atom) inner
    so consecutive stores hit the same output row across N.
    """
    lane_row = lane_id.shrui(fx.Int32(4))  # TCU_LANE_COLS == 16
    lane_col = lane_id & fx.Int32(TCU_LANE_COLS - 1)
    voffset = (lane_row * fx.Int32(c_global_n) + lane_col) * fx.Int32(4)

    c_warp_ptr = fx.get_iter(gC_warp)

    for im in fx.range_constexpr(warp_atoms_m):
        loaded = [Vec(accs[im][jn].load()) for jn in range(warp_atoms_n)]
        for ei in fx.range_constexpr(4):
            row_soffset = fx.Int32((im * ATOM_M + ei * 4) * c_global_n * 4)
            for jn in fx.range_constexpr(warp_atoms_n):
                soffset = row_soffset + fx.Int32(jn * ATOM_N * 4)
                stp_vs_b32(loaded[jn][ei], c_warp_ptr, voffset, soffset)


def mr_igemm_epilogue_store_i8_packed(
    *,
    lane_id,
    warp_id,
    accs,
    gC_warp,
    smem_base,
    warp_atoms_m: int,
    warp_atoms_n: int,
    c_global_n: int,
):
    """int8 packed store (int8 GEMM, ``D = A @ B.T``, truncating cast).
    Per lane each MMA atom owns 4 rows {r, r+4, r+8, r+12} x 1
    For every group of 4 N-atoms (64 cols):
      1. pack each atom's 4 i8 rows into one i32 (truncating cast, wrap on overflow);
      2. scatter the 4 i32 into SMEM (32-bit writes,
         bank-conflict free) -> read back 4 i32 with the transpose swizzle;
      3. 6x ``byte_permute`` recombine -> 4 i32, each = 4 contiguous-N i8 of one row;
      4. ``stp_vs_b32`` coalesced store (``voffset`` + ``soffset``) per output row.

    No quant scale/bias/relu fusion.
    """
    if fx.const_expr(warp_atoms_n % 4 != 0):
        raise ValueError("i8 packed epilogue requires warp_atoms_n %% 4 == 0")
    warp_m = ATOM_M * warp_atoms_m
    warp_n = ATOM_N * warp_atoms_n
    groups_n = warp_atoms_n // 4

    lane_row = lane_id // fx.Int32(TCU_LANE_COLS)  # 0..3
    lane_col = lane_id % fx.Int32(TCU_LANE_COLS)  # 0..15
    lane01 = lane_col % fx.Int32(4)
    lane23 = lane_col // fx.Int32(4)
    voffset = lane_row * fx.Int32(c_global_n) + lane_col * fx.Int32(4)

    smem_warp_i32 = fx.recast_iter(
        fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared),
        smem_base,
    )
    warp_base = warp_id * fx.Int32(warp_m * warp_n // 4)

    def _pack_i32(acc):
        return Vec(acc.load()).to(fx.Int8).bitcast(fx.Int32)[0]

    # Mainloop has finished reading the pipeline smem; safe to reuse for staging.
    fx.gpu.barrier()
    # Phase 1: scatter-write all (im, group) blocks.
    for im in fx.range_constexpr(warp_atoms_m):
        for g in fx.range_constexpr(groups_n):
            block = warp_base + fx.Int32((im * groups_n + g) * 256)
            for e in fx.range_constexpr(4):
                src_e = _pack_i32(accs[im][g * 4 + e])
                idx = (
                    block
                    + lane01 * fx.Int32(64)
                    + lane_row * fx.Int32(16)
                    + (lane01 ^ fx.Int32(e)) * fx.Int32(4)
                    + lane23
                )
                fx.ptr_store(src_e, fx.add_offset(smem_warp_i32, fx.make_int_tuple(idx)))

    fx.gpu.barrier()

    # Phase 2: transpose-read + byte_permute recombine + stp.vs store.
    c_warp_ptr = fx.get_iter(gC_warp)
    for im in fx.range_constexpr(warp_atoms_m):
        for g in fx.range_constexpr(groups_n):
            block = warp_base + fx.Int32((im * groups_n + g) * 256)
            val = []
            for e in fx.range_constexpr(4):
                idx = block + fx.Int32(e * 64) + (lane_id ^ fx.Int32(e * 4))
                val.append(fx.ptr_load(fx.add_offset(smem_warp_i32, fx.make_int_tuple(idx))))
            t0 = byte_permute(val[0], val[1], 0x5140)
            t1 = byte_permute(val[2], val[3], 0x5140)
            ret0 = byte_permute(t0, t1, 0x5410)
            ret1 = byte_permute(t0, t1, 0x7632)
            t0 = byte_permute(val[0], val[1], 0x7362)
            t1 = byte_permute(val[2], val[3], 0x7362)
            ret2 = byte_permute(t0, t1, 0x5410)
            ret3 = byte_permute(t0, t1, 0x7632)
            rets = (ret0, ret1, ret2, ret3)
            for k in fx.range_constexpr(4):
                soffset = fx.Int32((im * 16 + k * 4) * c_global_n) + fx.Int32(g * 64)
                stp_vs_b32(rets[k], c_warp_ptr, voffset, soffset)


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
    out_dtype=fx.Float16,
):
    """Dispatch to the selected MR HGEMM C-store epilogue.

    Args:
        store_mode: One of ``EPILOGUE_STORE_SHFL`` (``"shfl"``), ``EPILOGUE_STORE_TILED``
            (``"tiled"``), or ``EPILOGUE_STORE_READ_C_ACCUM`` (``"read_c_accum"``).
            ``shfl`` and ``tiled`` write f16/bf16 without reading C; ``read_c_accum`` writes
            fp32 accumulators (C was loaded before MMA).
        lane_id: Lane index within the warp (0 .. WARP_SIZE-1).
        accs: ``[mma_m][mma_n]`` f32 MMA accumulator fragments for this warp.
        gC_warp: This warp's global C tile view: ``gC`` flat-divided by
            ``(warp_m, warp_n)``, then sliced to ``(warp_m_id, warp_n_id)``.
        c_global_n: Full problem N extent; only used by the ``shfl`` path for GMEM
            stride / packed-store addressing.
        tiled_mma: Tiled MMA from ``fx.make_tiled_mma``; required by ``tiled`` and
            ``read_c_accum`` (ignored by ``shfl``).
        warp_atoms_m: Count of ``atom_m`` (16) MMA tiles along M owned by this warp.
        warp_atoms_n: Count of ``atom_n`` (16) MMA tiles along N owned by this warp.
        out_dtype: Output element type for ``shfl`` / ``tiled`` (``Float16`` or ``BFloat16``);
            ignored by ``read_c_accum``.
    """
    if fx.const_expr(store_mode == EPILOGUE_STORE_SHFL):
        mr_hgemm_epilogue_store_shfl(
            lane_id=lane_id,
            accs=accs,
            gC_warp=gC_warp,
            c_global_n=c_global_n,
            warp_atoms_m=warp_atoms_m,
            warp_atoms_n=warp_atoms_n,
            out_dtype=out_dtype,
        )
    elif fx.const_expr(store_mode == EPILOGUE_STORE_TILED):
        mr_hgemm_epilogue_store_tiled(
            lane_id=lane_id,
            accs=accs,
            gC_warp=gC_warp,
            tiled_mma=tiled_mma,
            warp_atoms_m=warp_atoms_m,
            warp_atoms_n=warp_atoms_n,
            out_dtype=out_dtype,
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
