# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Reusable Iluvatar MR HGEMM / IGEMM epilogue helpers.

  mr_hgemm_epilogue_store_shfl -- f16/bf16 via warp shuffle + packed i32 store
  mr_hgemm_epilogue_store_tiled -- f16/bf16 via trunc_f + make_tiled_copy_C
  mr_hgemm_epilogue_store_read_c_accum -- fp32 make_tiled_copy_C
  mr_igemm_epilogue_store_i32 -- int32 direct store
  mr_igemm_epilogue_store_i8_packed -- int8 packed store (no quant scale)
  mr_igemm_epilogue_store_scaled -- dequant (+ optional bias) -> f16/bf16
      via PackSlb packed store

mr_hgemm_epilogue_store dispatches on store_mode.
"""

import flydsl.expr as fx
from flydsl.expr.ixdl import byte_permute, stp_vs_b32, stp_vs_pred_b32
from flydsl.expr.typing import Vector as Vec
from kernels.gemm.iluvatar.common import WARP_SIZE
from kernels.gemm.iluvatar.mr.common import ATOM_M, ATOM_N, TCU_LANE_COLS

EPILOGUE_STORE_SHFL = "shfl"
EPILOGUE_STORE_TILED = "tiled"
EPILOGUE_STORE_READ_C_ACCUM = "read_c_accum"
EPILOGUE_STORE_ATOMIC_SPLITK = "atomic_splitk"
EPILOGUE_STORE_SERIAL_SPLITK = "serial_splitk"

# PackSlb C stores use write-through (BK != 32).
_PACKSLB_STORE_KOP = 3
# 512B shared scratch per warp for b16 PackSlb (128 i32).
B16_PACK_SLB_BYTES_PER_WARP = 512
B16_PACK_SLB_I32_PER_WARP = B16_PACK_SLB_BYTES_PER_WARP // 4


def _b16x2_pack_slb(*, sk, lane_id, in1, in2):
    """Pack two i32 b16x2 pairs through 512B/warp shared into coalesced stores.

    ``in1`` = packed ``{val0, val2}``, ``in2`` = packed ``{val1, val3}`` as i32.
    Returns ``(out0, out1)`` ready for coalesced b16x2 stores.
    ``sk`` is this warp's 128-i32 shared scratch.
    """
    lane_row = lane_id.shrui(fx.Int32(4))
    lane_col = lane_id & fx.Int32(TCU_LANE_COLS - 1)
    lane0 = lane_col & fx.Int32(1)
    lane123 = lane_col.shrui(fx.Int32(1))
    xor_mask = lane0 * fx.Int32(8)
    base = lane0 * fx.Int32(64) + lane_row * fx.Int32(16) + lane123
    fx.ptr_store(in1, fx.add_offset(sk, fx.make_int_tuple(base ^ xor_mask)))
    fx.ptr_store(in2, fx.add_offset(sk, fx.make_int_tuple((base + fx.Int32(8)) ^ xor_mask)))
    loaded0 = fx.ptr_load(fx.add_offset(sk, fx.make_int_tuple(lane_id)))
    loaded1 = fx.ptr_load(fx.add_offset(sk, fx.make_int_tuple(fx.Int32(64) + (lane_id ^ fx.Int32(8)))))
    out0 = byte_permute(loaded0, loaded1, 0x5410)
    out1 = byte_permute(loaded0, loaded1, 0x7632)
    return out0, out1


def _pack_and_swizzle_4x2b_to_2x4b_slb(*, sk, lane_id, h0, h1, h2, h3, out_dtype):
    """Pack four b16 values at ``(r,c), (r,c+16), (r+4,c), (r+4,c+16)`` for PackSlb."""
    val02 = Vec(Vec.from_elements([h0, h2], out_dtype)).bitcast(fx.Int32)[0]
    val13 = Vec(Vec.from_elements([h1, h3], out_dtype)).bitcast(fx.Int32)[0]
    return _b16x2_pack_slb(sk=sk, lane_id=lane_id, in1=val02, in2=val13)


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


def _mr_hgemm_epilogue_store_tiled_like(
    *,
    lane_id,
    accs,
    gC_warp,
    tiled_mma,
    warp_atoms_m: int,
    warp_atoms_n: int,
    out_dtype,
    copy_atom_c,
):
    """Shared trunc + ``make_tiled_copy_C`` store loop; ``copy_atom_c`` selects plain vs atomic."""
    gC_atoms = fx.flat_divide(gC_warp, (ATOM_M, ATOM_N))
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
    copy_atom_c = fx.make_copy_atom(fx.UniversalCopy16b(), out_dtype)
    _mr_hgemm_epilogue_store_tiled_like(
        lane_id=lane_id,
        accs=accs,
        gC_warp=gC_warp,
        tiled_mma=tiled_mma,
        warp_atoms_m=warp_atoms_m,
        warp_atoms_n=warp_atoms_n,
        out_dtype=out_dtype,
        copy_atom_c=copy_atom_c,
    )


def mr_hgemm_epilogue_store_atomic_splitk(
    *,
    lane_id,
    accs,
    gC_warp,
    tiled_mma,
    warp_atoms_m: int,
    warp_atoms_n: int,
    out_dtype=fx.Float16,
):
    """Global Split-K atomic accumulate into C (``EPILOGUE_STORE_ATOMIC_SPLITK``).

    Same partitioning as ``mr_hgemm_epilogue_store_tiled``, but the C copy atom is
    a scalar ``UniversalAtomic`` add so every split-K CTA reduces its partial into
    C in place. ivcore11 has scalar f16/bf16 atomic fadd but no packed f16x2 atom,
    so this issues one atomic per element (copy-atom thr layout is scalar). C must
    be pre-zeroed by the split-K launch wrapper before the GEMM runs.
    """
    copy_atom_c = fx.make_copy_atom(
        fx.UniversalAtomic(fx.AtomicOp.Add, out_dtype, syncscope=fx.SyncScope.System),
        out_dtype,
    )
    _mr_hgemm_epilogue_store_tiled_like(
        lane_id=lane_id,
        accs=accs,
        gC_warp=gC_warp,
        tiled_mma=tiled_mma,
        warp_atoms_m=warp_atoms_m,
        warp_atoms_n=warp_atoms_n,
        out_dtype=out_dtype,
        copy_atom_c=copy_atom_c,
    )


def mr_hgemm_epilogue_store_serial_splitk(
    *,
    lane_id,
    accs,
    gC_warp,
    tiled_mma,
    warp_atoms_m: int,
    warp_atoms_n: int,
    out_dtype=fx.Float16,
):
    """CUTLASS SplitKSerial epilogue (``EPILOGUE_STORE_SERIAL_SPLITK``).

    Always load-add-store in f32 then trunc-store (ordered RMW -- not atomic).
    The caller must guarantee K-partition order (per-tile cmpxchg turnstile);
    partition 0 sees a pre-zeroed C and is equivalent to a plain store.
    """
    gC_atoms = fx.flat_divide(gC_warp, (ATOM_M, ATOM_N))

    copy_atom_c = fx.make_copy_atom(fx.UniversalCopy16b(), out_dtype)
    tiled_copy_c = fx.make_tiled_copy_C(copy_atom_c, tiled_mma)
    thr_copy_c = tiled_copy_c.get_slice(lane_id)

    for mma_m in fx.range_constexpr(warp_atoms_m):
        for mma_n in fx.range_constexpr(warp_atoms_n):
            c_tile = fx.slice(gC_atoms, (None, None, mma_m, mma_n))
            acc = accs[mma_m][mma_n]
            frag_c = fx.make_fragment_like(acc, out_dtype.ir_type)
            fx.copy(
                copy_atom_c,
                thr_copy_c.partition_S(c_tile),
                thr_copy_c.retile(frag_c),
                pred=None,
            )
            merged = Vec(acc.load()) + Vec(frag_c.load()).to(fx.Float32)
            frag_o = fx.make_fragment_like(acc, out_dtype.ir_type)
            frag_o.store(merged.to(out_dtype))
            fx.copy(
                copy_atom_c,
                thr_copy_c.retile(frag_o),
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


def mr_igemm_epilogue_store_scaled(
    *,
    lane_id,
    warp_id,
    warp_m_id,
    warp_n_id,
    m_tile,
    n_tile,
    accs,
    scale_a,
    scale_b,
    bias,
    gC_warp,
    smem_base,
    c_global_n: int,
    bm: int,
    bn: int,
    warp_m: int,
    warp_n: int,
    warp_atoms_m: int,
    warp_atoms_n: int,
    out_dtype,
    apply_bias: bool = False,
    skip_cta_barrier: bool = False,
    m_valid: int | None = None,
):
    """Scaled int32-acc store: dequant to bf16/fp16, then PackSlb.

    ``D = float(acc) * scale_a[m] * scale_b[n] [+ bias[n]]`` -> bf16/fp16, then
    pack 4xb16 ``(r,c)/(r,c+16)/(r+4,c)/(r+4,c+16)`` through 512B/warp shared
    and write coalesced i32 words. No alpha/beta/relu.

    Acc layout matches i32 IGEMM: lane owns rows ``{r,r+4,r+8,r+12}`` at
    ``lane_col`` in each 16x16 atom. Requires even ``warp_atoms_n``.
    When ``m_valid`` is set, rows with ``global_m >= m_valid`` are not stored
    (dynamic-M / short-M edge). It may be a Python ``int`` (folded at compile
    time) or an SSA value carrying the runtime row count.
    """
    if fx.const_expr(warp_atoms_n % 2 != 0):
        raise ValueError("scaled PackSlb requires even warp_atoms_n")

    # Python int folds to a constant; anything else is already an SSA row count
    # (dynamic-M launcher passes the runtime M).
    m_limit = None
    if fx.const_expr(m_valid is not None):
        m_limit = fx.Int32(m_valid) if fx.const_expr(isinstance(m_valid, int)) else m_valid

    lane_row = lane_id.shrui(fx.Int32(4))  # 0..3
    lane_col = lane_id & fx.Int32(TCU_LANE_COLS - 1)

    m_base = m_tile * fx.Int32(bm)
    n_base = n_tile * fx.Int32(bn)
    warp_m_base = warp_m_id * fx.Int32(warp_m)
    warp_n_base = warp_n_id * fx.Int32(warp_n)

    # Preload per-row scale_a and per-column scale_b / bias before the store.
    # The dq load runs ahead of the row guard below, so a tile hanging off the
    # end of M would read past ``scale_a``. Clamping the row to the last live
    # one keeps the read in bounds and lets callers pass ``scale_a`` at its
    # natural M length; the value fetched for a dead row is discarded with the
    # row itself.
    dq_m = []
    if fx.const_expr(m_valid is not None):
        m_last = m_limit - fx.Int32(1)
    for im in range(warp_atoms_m):
        row = []
        for ei in range(4):
            global_m = m_base + warp_m_base + fx.Int32(im * ATOM_M + ei * 4) + lane_row
            if fx.const_expr(m_valid is not None):
                over = fx.arith.cmpi(fx.arith.CmpIPredicate.ugt, global_m, m_last)
                global_m = fx.Int32(fx.arith.select(over, m_last, global_m))
            row.append(fx.Float32(scale_a[global_m]))
        dq_m.append(row)
    dq_n = []
    bias_n = []
    for jn in range(warp_atoms_n):
        global_n = n_base + warp_n_base + fx.Int32(jn * ATOM_N) + lane_col
        dq_n.append(fx.Float32(scale_b[global_n]))
        if fx.const_expr(apply_bias):
            bias_n.append(fx.Float32(bias[global_n]))

    if fx.const_expr(not skip_cta_barrier):
        fx.gpu.barrier()

    smem_warp_i32 = fx.recast_iter(
        fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared),
        smem_base,
    )
    sk = fx.add_offset(smem_warp_i32, fx.make_int_tuple(warp_id * fx.Int32(B16_PACK_SLB_I32_PER_WARP)))

    # Bake the per-lane byte offset once; each store is a constexpr tile soffset
    # plus that voffset. Write-through so the one-shot C burst does not fill L1.
    c_warp_ptr = fx.get_iter(gC_warp)
    voffset = (lane_row * fx.Int32(c_global_n) + lane_col * fx.Int32(2)) * fx.Int32(2)

    for im in fx.range_constexpr(warp_atoms_m):
        loaded = [Vec(accs[im][jn].load()) for jn in range(warp_atoms_n)]
        for ei_base in fx.range_constexpr(0, 4, 2):
            for jn in fx.range_constexpr(0, warp_atoms_n, 2):
                out0 = loaded[jn][ei_base].to(fx.Float32) * dq_m[im][ei_base] * dq_n[jn]
                out1 = loaded[jn + 1][ei_base].to(fx.Float32) * dq_m[im][ei_base] * dq_n[jn + 1]
                out2 = loaded[jn][ei_base + 1].to(fx.Float32) * dq_m[im][ei_base + 1] * dq_n[jn]
                out3 = loaded[jn + 1][ei_base + 1].to(fx.Float32) * dq_m[im][ei_base + 1] * dq_n[jn + 1]
                if fx.const_expr(apply_bias):
                    out0 = out0 + bias_n[jn]
                    out1 = out1 + bias_n[jn + 1]
                    out2 = out2 + bias_n[jn]
                    out3 = out3 + bias_n[jn + 1]
                packed0, packed1 = _pack_and_swizzle_4x2b_to_2x4b_slb(
                    sk=sk,
                    lane_id=lane_id,
                    h0=out0.to(out_dtype),
                    h1=out1.to(out_dtype),
                    h2=out2.to(out_dtype),
                    h3=out3.to(out_dtype),
                    out_dtype=out_dtype,
                )
                # Packed i32 word at (row, col) of bf16 C: byte = (row*N + col)*2
                # with col = jn*16 + lane_col*2 (two b16 values).
                soffset0 = fx.Int32((im * ATOM_M + ei_base * 4) * c_global_n * 2 + jn * ATOM_N * 2)
                soffset1 = fx.Int32((im * ATOM_M + (ei_base + 1) * 4) * c_global_n * 2 + jn * ATOM_N * 2)
                if fx.const_expr(m_valid is not None):
                    local_row0 = fx.Int32(im * ATOM_M + ei_base * 4) + lane_row
                    local_row1 = fx.Int32(im * ATOM_M + (ei_base + 1) * 4) + lane_row
                    global_m0 = m_base + warp_m_base + local_row0
                    global_m1 = m_base + warp_m_base + local_row1
                    ok0 = fx.arith.cmpi(fx.arith.CmpIPredicate.ult, global_m0, m_limit)
                    ok1 = fx.arith.cmpi(fx.arith.CmpIPredicate.ult, global_m1, m_limit)
                    stp_vs_pred_b32(ok0, packed0, c_warp_ptr, voffset, soffset0, _PACKSLB_STORE_KOP)
                    stp_vs_pred_b32(ok1, packed1, c_warp_ptr, voffset, soffset1, _PACKSLB_STORE_KOP)
                else:
                    stp_vs_b32(packed0, c_warp_ptr, voffset, soffset0, _PACKSLB_STORE_KOP)
                    stp_vs_b32(packed1, c_warp_ptr, voffset, soffset1, _PACKSLB_STORE_KOP)


def mr_igemm_epilogue_store_i8_pack_only(
    *,
    lane_id,
    accs,
    gC_warp,
    warp_atoms_m: int,
    warp_atoms_n: int,
    c_global_n: int,
):
    """int8 packed store via PackOnly (no SLB shuffle).

    For each group of 4 N-atoms, pack the same row-register across the group
    with 3x ``byte_permute`` (trunc-wrap), then ``stp_vs_b32``. Requires B
    ``N_SWIZZLE=4`` so those 4 atoms already hold consecutive logical N.
    """
    if fx.const_expr(warp_atoms_n % 4 != 0):
        raise ValueError("i8 PackOnly epilogue requires warp_atoms_n %% 4 == 0")

    lane_row = lane_id.shrui(fx.Int32(4))
    lane_col = lane_id & fx.Int32(TCU_LANE_COLS - 1)
    voffset = lane_row * fx.Int32(c_global_n) + lane_col * fx.Int32(4)
    c_warp_ptr = fx.get_iter(gC_warp)
    groups_n = warp_atoms_n // 4

    for im in fx.range_constexpr(warp_atoms_m):
        loaded = [Vec(accs[im][jn].load()) for jn in range(warp_atoms_n)]
        for g in fx.range_constexpr(groups_n):
            packed = []
            for ei in fx.range_constexpr(4):
                lo = byte_permute(loaded[g * 4 + 0][ei], loaded[g * 4 + 1][ei], 0x40)
                hi = byte_permute(loaded[g * 4 + 2][ei], loaded[g * 4 + 3][ei], 0x40)
                packed.append(byte_permute(lo, hi, 0x5410))
            for ei in fx.range_constexpr(4):
                soffset = fx.Int32((im * ATOM_M + ei * 4) * c_global_n)
                stp_vs_b32(packed[ei], c_warp_ptr, voffset, soffset + fx.Int32(g * 64))


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
    skip_cta_barrier: bool = False,
    pack_only: bool = False,
):
    """int8 packed store (int8 GEMM, ``D = A @ B.T``, truncating cast).

    ``pack_only=True``: PackOnly (register pack + ``stp_vs_b32``, no SLB); see
    ``mr_igemm_epilogue_store_i8_pack_only``. Requires matching B ``N_SWIZZLE=4``.

    ``pack_only=False`` (default): PackSlb path. Per lane each MMA atom owns 4 rows
    {r, r+4, r+8, r+12} x 1. For every group of 4 N-atoms (64 cols):
      1. pack each atom's 4 i8 rows into one i32 (truncating cast, wrap on overflow);
      2. scatter the 4 i32 into a fixed 1KB/warp SMEM slot (``SK``, reused per
         group -- no ``bm * bn`` scratch) -> read back 4 i32 with the transpose swizzle;
      3. 6x ``byte_permute`` recombine -> 4 i32, each = 4 contiguous-N i8 of one row;
      4. ``stp_vs_b32`` coalesced store (``voffset`` + ``soffset``) per output row.

    When ``skip_cta_barrier`` is True, ``smem_base`` must be a free pipeline stage
    that the mainloop no longer reads. Ignored for PackOnly.

    No quant scale/bias/relu fusion.
    """
    if fx.const_expr(pack_only):
        mr_igemm_epilogue_store_i8_pack_only(
            lane_id=lane_id,
            accs=accs,
            gC_warp=gC_warp,
            warp_atoms_m=warp_atoms_m,
            warp_atoms_n=warp_atoms_n,
            c_global_n=c_global_n,
        )
        return

    if fx.const_expr(warp_atoms_n % 4 != 0):
        raise ValueError("i8 packed epilogue requires warp_atoms_n %% 4 == 0")
    groups_n = warp_atoms_n // 4

    lane_row = lane_id.shrui(fx.Int32(4))  # 0..3
    lane_col = lane_id & fx.Int32(TCU_LANE_COLS - 1)  # 0..15
    lane01 = lane_col & fx.Int32(3)
    lane23 = lane_col.shrui(fx.Int32(2))
    voffset = lane_row * fx.Int32(c_global_n) + lane_col * fx.Int32(4)

    smem_warp_i32 = fx.recast_iter(
        fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared),
        smem_base,
    )
    sk = fx.add_offset(smem_warp_i32, fx.make_int_tuple(warp_id * fx.Int32(256)))

    def _pack_i32(acc):
        return Vec(acc.load()).to(fx.Int8).bitcast(fx.Int32)[0]

    if fx.const_expr(not skip_cta_barrier):
        # Mainloop has finished reading the pipeline smem; safe to reuse for staging.
        fx.gpu.barrier()
    c_warp_ptr = fx.get_iter(gC_warp)

    # Pack + transpose-read + byte_permute recombine per (im, group), reusing a
    # fixed 1KB/warp SK slot (no bm * bn scratch): each warp's own write-then-read
    # of its SK bytes needs no barrier between them.
    all_rets = []
    for im in fx.range_constexpr(warp_atoms_m):
        group_rets = []
        for g in fx.range_constexpr(groups_n):
            for e in fx.range_constexpr(4):
                src_e = _pack_i32(accs[im][g * 4 + e])
                idx = lane01 * fx.Int32(64) + lane_row * fx.Int32(16) + (lane01 ^ fx.Int32(e)) * fx.Int32(4) + lane23
                fx.ptr_store(src_e, fx.add_offset(sk, fx.make_int_tuple(idx)))
            val = []
            for e in fx.range_constexpr(4):
                idx = fx.Int32(e * 64) + (lane_id ^ fx.Int32(e * 4))
                val.append(fx.ptr_load(fx.add_offset(sk, fx.make_int_tuple(idx))))
            t0 = byte_permute(val[0], val[1], 0x5140)
            t1 = byte_permute(val[2], val[3], 0x5140)
            ret0 = byte_permute(t0, t1, 0x5410)
            ret1 = byte_permute(t0, t1, 0x7632)
            t0 = byte_permute(val[0], val[1], 0x7362)
            t1 = byte_permute(val[2], val[3], 0x7362)
            ret2 = byte_permute(t0, t1, 0x5410)
            ret3 = byte_permute(t0, t1, 0x7632)
            group_rets.append((ret0, ret1, ret2, ret3))
        all_rets.append(group_rets)

    # Store burst: im -> ei -> g so consecutive stores hit the same output row.
    for im in fx.range_constexpr(warp_atoms_m):
        for ei in fx.range_constexpr(4):
            row_soffset = fx.Int32((im * ATOM_M + ei * 4) * c_global_n)
            for g in fx.range_constexpr(groups_n):
                stp_vs_b32(all_rets[im][g][ei], c_warp_ptr, voffset, row_soffset + fx.Int32(g * 64))


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
