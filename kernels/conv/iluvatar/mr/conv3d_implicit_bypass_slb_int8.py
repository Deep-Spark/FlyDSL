# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar MR int8 implicit-GEMM Conv3D with activation SLB bypass.

This kernel implements the defining data flow of ixInfer's
``implConvolutionTcuBypassSlbKernel``:

* activations are gathered directly from packed NDHWC global memory into MMA
  register fragments and never occupy shared memory;
* weights are staged in an SME-swizzled B-only shared-memory tile with async
  SME global-to-shared copies;
* a 64x64x64 CTA is computed by one 64-lane MR warp using 16x16x32 int8 MMA;
* int32 accumulators are saturated to int8 in the epilogue.

The public wrapper accepts PyTorch-style NCDHW input and KCTRS weights. It
implements cross-correlation and supports groups, stride, padding, and dilation.
Packing and output unpacking are deliberately kept outside the timed kernel.
"""

# NOTE: do not add ``from __future__ import annotations``; FlyDSL inspects
# constexpr annotations while building kernels.

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.ixdl as ixdl
from flydsl.expr import arith
from flydsl.expr.typing import Vector as Vec
from kernels.gemm.iluvatar.common import WARP_SIZE
from kernels.gemm.iluvatar.mr.common import (
    ATOM_K_B8,
    ATOM_M,
    ATOM_N,
    MrOperandGeom,
    SMEM_ROWS,
)
from kernels.gemm.iluvatar.mr.operand_copy import (
    mr_g2s_sme_config,
    mr_gemm_g2s_issue_b_warp,
)
from kernels.gemm.iluvatar.mr.s2r import (
    mr_gemm_s2r_b_tile,
    mr_gemm_s2r_copy_b,
)


BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 64
WARP_M = 64
WARP_N = 64
WARPS_M = BLOCK_M // WARP_M
WARPS_N = BLOCK_N // WARP_N
BLOCK_THREADS = WARPS_M * WARPS_N * WARP_SIZE
WARP_ATOMS_M = WARP_M // ATOM_M
WARP_ATOMS_N = WARP_N // ATOM_N
K_ATOMS = BLOCK_K // ATOM_K_B8
PACK_I8 = 4
INT8_GEOM = MrOperandGeom.b8()

assert (BLOCK_M, BLOCK_N, BLOCK_K) == (64, 64, 64)
assert BLOCK_THREADS == 64
assert WARP_ATOMS_M == 4 and WARP_ATOMS_N == 4 and K_ATOMS == 2


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _triple(value, name: str) -> tuple[int, int, int]:
    if isinstance(value, int):
        return value, value, value
    if len(value) != 3:
        raise ValueError(f"{name} must be an int or a length-3 sequence")
    return tuple(int(v) for v in value)


def _output_shape(d, h, w, kt, kh, kw, stride, padding, dilation):
    st, sh, sw = stride
    pt, ph, pw = padding
    dt, dh, dw = dilation
    do = (d + 2 * pt - dt * (kt - 1) - 1) // st + 1
    ho = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    wo = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    if do <= 0 or ho <= 0 or wo <= 0:
        raise ValueError(f"invalid output shape ({do}, {ho}, {wo})")
    return do, ho, wo


@functools.lru_cache(maxsize=128)
def compile_conv3d_implicit_bypass_slb(
    n: int,
    c_per_group: int,
    d: int,
    h: int,
    w: int,
    k_per_group: int,
    kt: int,
    kh: int,
    kw: int,
    st: int = 1,
    sh: int = 1,
    sw: int = 1,
    pt: int = 0,
    ph: int = 0,
    pw: int = 0,
    dt: int = 1,
    dh: int = 1,
    dw: int = 1,
    groups: int = 1,
):
    """Build the MR BypassSlb int8 Conv3D launcher.

    Runtime tensors use the packed contract produced by
    :func:`prepare_conv3d_implicit_bypass_slb`:

    * ``x``: contiguous ``[N,D,H,W,groups*Cpad]`` int8;
    * ``weight``: contiguous ``[groups,Npad,KredPad]`` int8;
    * ``y``: contiguous ``[Mpad,groups*Npad]`` int8 workspace.
    """

    if min(n, c_per_group, d, h, w, k_per_group, kt, kh, kw, groups) <= 0:
        raise ValueError("all Conv3D extents and groups must be positive")
    if min(st, sh, sw, dt, dh, dw) <= 0:
        raise ValueError("stride and dilation must be positive")

    stride = (st, sh, sw)
    padding = (pt, ph, pw)
    dilation = (dt, dh, dw)
    do, ho, wo = _output_shape(d, h, w, kt, kh, kw, stride, padding, dilation)

    c_padded = _ceil_div(c_per_group, PACK_I8) * PACK_I8
    fold_c = c_padded <= BLOCK_K // 2
    reduction_k = kt * kh * kw * c_padded
    fold_xc = min(c_padded, BLOCK_K)
    fold_kw = min(kw, BLOCK_K // fold_xc)
    fold_kh = min(kh, BLOCK_K // fold_xc // fold_kw)
    loop_xc = _ceil_div(c_padded, fold_xc)
    loop_kw = _ceil_div(kw, fold_kw)
    loop_kh = _ceil_div(kh, fold_kh)
    k_tiles = kt * loop_xc * loop_kw * loop_kh
    reduction_k_padded = k_tiles * BLOCK_K
    m = n * do * ho * wo
    m_padded = _ceil_div(m, BLOCK_M) * BLOCK_M
    n_padded = _ceil_div(k_per_group, BLOCK_N) * BLOCK_N
    input_channels_padded = groups * c_padded
    output_channels_padded = groups * n_padded
    output_hw = ho * wo
    output_dhw = do * output_hw

    # BypassSlb allocates shared memory only for B. One int8 SME issue
    # moves 16x64 values, so the 64x64 B tile contains four chunks.
    b_elems = BLOCK_N * BLOCK_K
    b_atoms_total = b_elems // INT8_GEOM.cta_chunk_elems
    b_per_warp = b_atoms_total
    assert b_atoms_total == 4

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def conv3d_bypass_slb_int8_kernel(x: fx.Tensor, weight: fx.Tensor, y: fx.Tensor):
        tid = fx.thread_idx.x
        block_n, group_id, block_m = fx.block_idx
        lane_id = tid
        lane_row = lane_id // fx.Int32(ATOM_N)
        lane_col = lane_id % fx.Int32(ATOM_N)

        m_base = block_m * fx.Int32(BLOCK_M)
        n_base = block_n * fx.Int32(BLOCK_N)

        @fx.struct
        class ConvSmem:
            weight: fx.Array[fx.Int8, b_elems]

        smem_i8 = fx.SharedAllocator(static=True).allocate(ConvSmem).peek().weight.ptr
        x_ptr = fx.get_iter(x)
        x_i32 = fx.recast_iter(
            fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global),
            x_ptr,
        )

        mma_atom = fx.make_mma_atom(
            ixdl.MRMma(ATOM_M, ATOM_N, ATOM_K_B8, fx.Int8, fx.Int8, fx.Int32)
        )
        tiled_mma = fx.make_tiled_mma(
            mma_atom,
            fx.make_layout((1, 1, 1), (1, 1, 1)),
        )
        thr_mma = tiled_mma.thr_slice(lane_id)
        copy_atom_b = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int8)
        thr_copy_b = fx.make_tiled_copy_B(copy_atom_b, tiled_mma).get_slice(lane_id)
        g2s_sme = mr_g2s_sme_config(
            a_mn_major=False,
            b_mn_major=False,
            elem_dtype=fx.Int8,
            row_atom=ixdl.MRAsyncCpRow8b,
            row_swizzle=ixdl.SMESwizzle.Row8b,
        )
        tile_smem_b = fx.make_tile(
            SMEM_ROWS,
            INT8_GEOM.values_per_sme_row,
        )

        accs = []
        for mma_m in fx.range_constexpr(WARP_ATOMS_M):
            acc_row = m_base + fx.Int32(mma_m * ATOM_M)
            row_accs = []
            for mma_n in fx.range_constexpr(WARP_ATOMS_N):
                c_ptr = fx.add_offset(
                    fx.get_iter(y),
                    fx.make_int_tuple(
                        acc_row * fx.Int32(output_channels_padded)
                        + group_id * fx.Int32(n_padded)
                        + n_base
                        + fx.Int32(mma_n * ATOM_N)
                    ),
                )
                c_tile = fx.make_view(
                    c_ptr,
                    fx.make_layout((ATOM_M, ATOM_N), (output_channels_padded, 1)),
                )
                acc = thr_mma.make_fragment_C(c_tile)
                acc.fill(0)
                row_accs.append(acc)
            accs.append(row_accs)

        # Runtime reduction loop avoids compile-time IR growth for large filters.
        for k_tile, _ in fx.range(k_tiles, init=[]):
            k_tile_i32 = fx.Int32(k_tile)
            k_base = k_tile_i32 * fx.Int32(BLOCK_K)
            tile_h = k_tile_i32 % fx.Int32(loop_kh)
            tile_rem_h = k_tile_i32 // fx.Int32(loop_kh)
            tile_w = tile_rem_h % fx.Int32(loop_kw)
            tile_rem_w = tile_rem_h // fx.Int32(loop_kw)
            tile_c = tile_rem_w % fx.Int32(loop_xc)
            tile_d = tile_rem_w // fx.Int32(loop_xc)

            # B: async SME G2S, matching ixInfer's LoaderG2SB
            # (TRANSPOSE=true, ASYNC_G2S=true). Packed weights are K-major:
            # logical B(n,k) has stride (reduction_k_padded, 1), therefore
            # b_mn_major=False selects the Col SME path.
            weight_elem = (
                (group_id * fx.Int32(n_padded) + n_base)
                * fx.Int32(reduction_k_padded)
                + k_base
            )
            b_global_tile = fx.make_view(
                fx.add_offset(
                    fx.get_iter(weight),
                    fx.make_int_tuple(weight_elem),
                ),
                fx.make_layout(
                    (BLOCK_N, BLOCK_K),
                    (reduction_k_padded, 1),
                ),
            )
            sme_b = ixdl.make_sme_gmem_tensor(
                b_global_tile,
                leading_stride=reduction_k_padded,
            )
            b_cta_gmem_view = fx.zipped_divide(
                sme_b,
                tile_smem_b,
            )
            mr_gemm_g2s_issue_b_warp(
                a_mn_major=False,
                b_mn_major=False,
                warp_id=fx.Int32(0),
                b_per_warp=b_per_warp,
                b_cta_gmem_view=b_cta_gmem_view,
                g2s_sme=g2s_sme,
                smem_b=smem_i8,
                elem_dtype=fx.Int8,
                bm=BLOCK_M,
                bn=BLOCK_N,
                bk=BLOCK_K,
                geom=INT8_GEOM,
            )
            ixdl.cp_async_commit_group()

            fx.gpu.barrier()

            # Mirror ixInfer XReuseLoadG2SInfo::LoadG2V. Each lane loads four
            # packed rows from global memory, then SwizzleB8 rearranges them
            # into the two 16x32 A fragments for this 64-wide K tile.
            a_frags_lo = []
            a_frags_hi = []
            for mma_m in fx.range_constexpr(WARP_ATOMS_M):
                packed_rows = []
                for row_i in fx.range_constexpr(4):
                    row = (
                        m_base
                        + fx.Int32(mma_m * ATOM_M)
                        + lane_row * fx.Int32(4)
                        + fx.Int32(row_i)
                    )
                    fold_linear = lane_col * fx.Int32(PACK_I8)
                    if fx.const_expr(fold_c):
                        channel = (
                            tile_c * fx.Int32(fold_xc)
                            + fold_linear % fx.Int32(fold_xc)
                        )
                        fold_spatial = fold_linear // fx.Int32(fold_xc)
                        kernel_w = (
                            tile_w * fx.Int32(fold_kw)
                            + fold_spatial % fx.Int32(fold_kw)
                        )
                        kernel_h = (
                            tile_h * fx.Int32(fold_kh)
                            + fold_spatial // fx.Int32(fold_kw)
                        )
                    else:
                        channel = (
                            tile_c * fx.Int32(BLOCK_K) + fold_linear
                        )
                        kernel_w = tile_w
                        kernel_h = tile_h
                    kernel_d = tile_d

                    out_n = row // fx.Int32(output_dhw)
                    out_spatial = row % fx.Int32(output_dhw)
                    out_d = out_spatial // fx.Int32(output_hw)
                    out_hw_rem = out_spatial % fx.Int32(output_hw)
                    out_h = out_hw_rem // fx.Int32(wo)
                    out_w = out_hw_rem % fx.Int32(wo)

                    in_d = (
                        out_d * fx.Int32(st)
                        - fx.Int32(pt)
                        + kernel_d * fx.Int32(dt)
                    )
                    in_h = (
                        out_h * fx.Int32(sh)
                        - fx.Int32(ph)
                        + kernel_h * fx.Int32(dh)
                    )
                    in_w = (
                        out_w * fx.Int32(sw)
                        - fx.Int32(pw)
                        + kernel_w * fx.Int32(dw)
                    )
                    valid = (
                        (row < fx.Int32(m))
                        & (channel < fx.Int32(c_padded))
                        & (kernel_d < fx.Int32(kt))
                        & (kernel_h < fx.Int32(kh))
                        & (kernel_w < fx.Int32(kw))
                        & (in_d >= fx.Int32(0))
                        & (in_d < fx.Int32(d))
                        & (in_h >= fx.Int32(0))
                        & (in_h < fx.Int32(h))
                        & (in_w >= fx.Int32(0))
                        & (in_w < fx.Int32(w))
                    )
                    x_elem = (
                        ((((out_n * fx.Int32(d) + in_d) * fx.Int32(h) + in_h)
                        * fx.Int32(w) + in_w)
                        * fx.Int32(input_channels_padded))
                        + group_id * fx.Int32(c_padded)
                        + channel
                    )
                    safe_pack = arith.select(
                        valid,
                        x_elem // fx.Int32(PACK_I8),
                        fx.Int32(0),
                    )
                    raw = fx.Int32(
                        fx.ptr_load(
                            fx.add_offset(
                                x_i32,
                                fx.make_int_tuple(safe_pack),
                            )
                        )
                    )
                    packed_rows.append(
                        arith.select(valid, raw, fx.Int32(0))
                    )

                r0, r1, r2, r3 = packed_rows
                r6 = (r0 & fx.Int32(0xFFFF)) | (r2 << fx.Int32(16))
                r7 = (r1 & fx.Int32(0xFFFF)) | (r3 << fx.Int32(16))
                sr0 = (
                    (r6 & fx.Int32(0x00FF00FF))
                    | ((r7 << fx.Int32(8)) & fx.Int32(-0x00FF0100))
                )
                sr1 = (
                    ((r6 >> fx.Int32(8)) & fx.Int32(0x00FF00FF))
                    | (r7 & fx.Int32(-0x00FF0100))
                )

                r6_hi = (r0 >> fx.Int32(16)) | (r2 & fx.Int32(-0x10000))
                r7_hi = (r1 >> fx.Int32(16)) | (r3 & fx.Int32(-0x10000))
                sr2 = (
                    (r6_hi & fx.Int32(0x00FF00FF))
                    | ((r7_hi << fx.Int32(8)) & fx.Int32(-0x00FF0100))
                )
                sr3 = (
                    ((r6_hi >> fx.Int32(8)) & fx.Int32(0x00FF00FF))
                    | (r7_hi & fx.Int32(-0x00FF0100))
                )

                a_anchor = fx.add_offset(x_ptr, fx.make_int_tuple(fx.Int32(0)))
                a_tile = fx.make_view(
                    a_anchor,
                    fx.make_layout((ATOM_M, ATOM_K_B8), (ATOM_K_B8, 1)),
                )
                frag_a_lo = thr_mma.make_fragment_A(a_tile)
                frag_a_hi = thr_mma.make_fragment_A(a_tile)
                frag_a_lo.store(
                    Vec.from_elements(
                        [sr0, sr1],
                        fx.Int32,
                    ).bitcast(fx.Int8)
                )
                frag_a_hi.store(
                    Vec.from_elements(
                        [sr2, sr3],
                        fx.Int32,
                    ).bitcast(fx.Int8)
                )
                a_frags_lo.append(frag_a_lo)
                a_frags_hi.append(frag_a_hi)

            for mma_k in fx.range_constexpr(K_ATOMS):
                b_frags = []
                for mma_n in fx.range_constexpr(WARP_ATOMS_N):
                    b_tile = mr_gemm_s2r_b_tile(
                        a_mn_major=False,
                        b_mn_major=False,
                        mma_n=mma_n,
                        mma_k=mma_k,
                        g2s_sme=g2s_sme,
                        smem_b=smem_i8,
                        elem_dtype=fx.Int8,
                        warp_n_id=fx.Int32(0),
                        warp_atoms_n=WARP_ATOMS_N,
                        bm=BLOCK_M,
                        bn=BLOCK_N,
                        bk=BLOCK_K,
                        geom=INT8_GEOM,
                    )
                    frag_b = mr_gemm_s2r_copy_b(
                        copy_atom=copy_atom_b,
                        thr_copy_b=thr_copy_b,
                        thr_mma=thr_mma,
                        smem_b_tile=b_tile,
                    )
                    b_frags.append(frag_b)

                for mma_m in fx.range_constexpr(WARP_ATOMS_M):
                    if fx.const_expr(mma_k == 0):
                        frag_a = a_frags_lo[mma_m]
                    else:
                        frag_a = a_frags_hi[mma_m]
                    for mma_n in fx.range_constexpr(WARP_ATOMS_N):
                        fx.gemm(
                            mma_atom,
                            accs[mma_m][mma_n],
                            frag_a,
                            b_frags[mma_n],
                            accs[mma_m][mma_n],
                        )

            # All lanes must finish consuming B before the next overwrite.
            fx.gpu.barrier()

        y_ptr = fx.get_iter(y)
        for mma_m in fx.range_constexpr(WARP_ATOMS_M):
            for mma_n in fx.range_constexpr(WARP_ATOMS_N):
                values = Vec(accs[mma_m][mma_n].load())
                for elem_i in fx.range_constexpr(4):
                    value = values[elem_i]
                    value = arith.select(value > fx.Int32(127), fx.Int32(127), value)
                    value = arith.select(value < fx.Int32(-128), fx.Int32(-128), value)
                    out_row = (
                        m_base
                        + fx.Int32(mma_m * ATOM_M)
                        + lane_row
                        + fx.Int32(elem_i * 4)
                    )
                    out_col = (
                        group_id * fx.Int32(n_padded)
                        + n_base
                        + fx.Int32(mma_n * ATOM_N)
                        + lane_col
                    )
                    out_offset = (
                        out_row * fx.Int32(output_channels_padded) + out_col
                    )
                    fx.ptr_store(
                        fx.Int32(value).to(fx.Int8),
                        fx.add_offset(y_ptr, fx.make_int_tuple(out_offset)),
                    )

    grid = (
        n_padded // BLOCK_N,
        groups,
        m_padded // BLOCK_M,
    )
    block = (BLOCK_THREADS, 1, 1)

    @flyc.jit
    def launch(
        x: fx.Tensor,
        weight: fx.Tensor,
        y: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        conv3d_bypass_slb_int8_kernel(x, weight, y).launch(
            grid=grid,
            block=block,
            stream=stream,
        )

    return launch


def prepare_conv3d_implicit_bypass_slb(
    x,
    weight,
    *,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    """Pack NCDHW/KCTRS int8 tensors and allocate the padded output workspace."""

    if x.ndim != 5 or weight.ndim != 5:
        raise ValueError("x and weight must be 5D NCDHW/KCTRS tensors")
    if x.dtype != torch.int8 or weight.dtype != torch.int8:
        raise TypeError("x and weight must have torch.int8 dtype")
    if x.device != weight.device:
        raise ValueError("x and weight must be on the same device")

    n, c_total, d, h, w = (int(v) for v in x.shape)
    k_total, c_per_group, kt, kh, kw = (int(v) for v in weight.shape)
    if groups <= 0 or c_total % groups or k_total % groups:
        raise ValueError("input/output channels must be divisible by groups")
    if c_total // groups != c_per_group:
        raise ValueError("weight.shape[1] must equal input_channels/groups")

    stride = _triple(stride, "stride")
    padding = _triple(padding, "padding")
    dilation = _triple(dilation, "dilation")
    do, ho, wo = _output_shape(d, h, w, kt, kh, kw, stride, padding, dilation)

    k_per_group = k_total // groups
    c_padded = _ceil_div(c_per_group, PACK_I8) * PACK_I8
    fold_c = c_padded <= BLOCK_K // 2
    n_padded = _ceil_div(k_per_group, BLOCK_N) * BLOCK_N
    reduction_k = kt * kh * kw * c_padded
    fold_xc = min(c_padded, BLOCK_K)
    fold_kw = min(kw, BLOCK_K // fold_xc)
    fold_kh = min(kh, BLOCK_K // fold_xc // fold_kw)
    loop_xc = _ceil_div(c_padded, fold_xc)
    loop_kw = _ceil_div(kw, fold_kw)
    loop_kh = _ceil_div(kh, fold_kh)
    k_tiles = kt * loop_xc * loop_kw * loop_kh
    reduction_k_padded = k_tiles * BLOCK_K
    m = n * do * ho * wo
    m_padded = _ceil_div(m, BLOCK_M) * BLOCK_M

    x_grouped = x.reshape(
        n, groups, c_per_group, d, h, w
    ).permute(0, 3, 4, 5, 1, 2)
    x_packed_6d = torch.zeros(
        (n, d, h, w, groups, c_padded),
        dtype=torch.int8,
        device=x.device,
    )
    x_packed_6d[..., :c_per_group] = x_grouped
    x_packed = x_packed_6d.reshape(
        n, d, h, w, groups * c_padded
    ).contiguous()

    weight_grouped = weight.reshape(
        groups,
        k_per_group,
        c_per_group,
        kt,
        kh,
        kw,
    ).permute(0, 1, 3, 4, 5, 2)
    weight_channels_padded = torch.zeros(
        (groups, n_padded, kt, kh, kw, c_padded),
        dtype=torch.int8,
        device=weight.device,
    )
    weight_channels_padded[:, :k_per_group, ..., :c_per_group] = weight_grouped
    weight_packed = torch.zeros(
        (groups, n_padded, k_tiles, BLOCK_K),
        dtype=torch.int8,
        device=weight.device,
    )
    # Match ixInfer PadFilter<FoldC>: low four channel bytes are distributed
    # across the four 16-element K quarters consumed after SwizzleB8.
    tile_id = 0
    dwords_per_xc = fold_xc // PACK_I8
    for kernel_d in range(kt):
        for loop_c_id in range(loop_xc):
            for loop_w_id in range(loop_kw):
                for loop_h_id in range(loop_kh):
                    for fold_id in range(BLOCK_K):
                        channel_high = fold_id // (BLOCK_K // PACK_I8)
                        fold_low = fold_id % (BLOCK_K // PACK_I8)
                        if fold_c:
                            channel_low = fold_low % dwords_per_xc
                            fold_w_id = (
                                fold_low // dwords_per_xc
                            ) % fold_kw
                            fold_h_id = (
                                fold_low // dwords_per_xc // fold_kw
                            )
                            kernel_w = (
                                loop_w_id * fold_kw + fold_w_id
                            )
                            kernel_h = (
                                loop_h_id * fold_kh + fold_h_id
                            )
                            channel = (
                                loop_c_id * fold_xc
                                + channel_low * PACK_I8
                                + channel_high
                            )
                        else:
                            fold_h_id = 0
                            kernel_w = loop_w_id
                            kernel_h = loop_h_id
                            channel = (
                                loop_c_id * BLOCK_K
                                + fold_low * PACK_I8
                                + channel_high
                            )
                        if (
                            kernel_h < kh
                            and kernel_w < kw
                            and channel < c_padded
                            and fold_h_id < fold_kh
                        ):
                            weight_packed[:, :, tile_id, fold_id] = (
                                weight_channels_padded[
                                    :,
                                    :,
                                    kernel_d,
                                    kernel_h,
                                    kernel_w,
                                    channel,
                                ]
                            )
                    tile_id += 1
    weight_packed = weight_packed.reshape(
        groups,
        n_padded,
        reduction_k_padded,
    )
    y_workspace = torch.empty(
        (m_padded, groups * n_padded),
        dtype=torch.int8,
        device=x.device,
    )
    meta = {
        "shape": (n, c_per_group, d, h, w, k_per_group, kt, kh, kw),
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
        "output_shape": (do, ho, wo),
        "m": m,
        "m_padded": m_padded,
        "n_padded": n_padded,
        "reduction_k": reduction_k,
        "reduction_k_padded": reduction_k_padded,
    }
    return x_packed, weight_packed, y_workspace, meta


def unpack_conv3d_implicit_bypass_slb_output(y_workspace, meta):
    """Convert the padded matrix output to contiguous NCDHW."""

    n, _, _, _, _, k_per_group, _, _, _ = meta["shape"]
    do, ho, wo = meta["output_shape"]
    groups = meta["groups"]
    m = meta["m"]
    n_padded = meta["n_padded"]
    y = y_workspace[:m].view(n, do, ho, wo, groups, n_padded)
    y = y[..., :k_per_group].reshape(
        n, do, ho, wo, groups * k_per_group
    )
    return y.permute(0, 4, 1, 2, 3).contiguous()


def conv3d_implicit_bypass_slb(
    x,
    weight,
    *,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    stream=None,
):
    """Run BypassSlb int8 Conv3D and return saturated int8 NCDHW output."""

    x_packed, weight_packed, y_workspace, meta = (
        prepare_conv3d_implicit_bypass_slb(
            x,
            weight,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
    )
    n, c_per_group, d, h, w, k_per_group, kt, kh, kw = meta["shape"]
    st, sh, sw = meta["stride"]
    pt, ph, pw = meta["padding"]
    dt, dh, dw = meta["dilation"]
    launch = compile_conv3d_implicit_bypass_slb(
        n,
        c_per_group,
        d,
        h,
        w,
        k_per_group,
        kt,
        kh,
        kw,
        st,
        sh,
        sw,
        pt,
        ph,
        pw,
        dt,
        dh,
        dw,
        groups,
    )
    launch(
        x_packed,
        weight_packed,
        y_workspace,
        torch.cuda.current_stream() if stream is None else stream,
    )
    return unpack_conv3d_implicit_bypass_slb_output(y_workspace, meta)


__all__ = [
    "BLOCK_K",
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_THREADS",
    "compile_conv3d_implicit_bypass_slb",
    "conv3d_implicit_bypass_slb",
    "prepare_conv3d_implicit_bypass_slb",
    "unpack_conv3d_implicit_bypass_slb_output",
]
