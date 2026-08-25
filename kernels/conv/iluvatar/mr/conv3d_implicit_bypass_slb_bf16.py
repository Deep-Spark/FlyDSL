# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar MR BF16 implicit-GEMM Conv3D with activation SLB bypass.

The implementation follows ixInfer's BF16 BypassSlb data flow:

* packed NDHWC activations are loaded directly from global memory into MMA
  register fragments;
* the packed filter tile is staged in SME-swizzled shared memory with async
  SME global-to-shared copies;
* 16x16x16 BF16 MR MMA accumulates in FP32;
* the epilogue converts FP32 accumulators to BF16.

The public wrapper accepts NCDHW input and KCTRS weights and implements
cross-correlation with groups, stride, padding, and dilation.
"""

# FlyDSL inspects constexpr annotations while building kernels.

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.ixdl as ixdl
from flydsl.expr import arith
from flydsl.expr.typing import Vector as Vec
from kernels.gemm.iluvatar.common import WARP_SIZE
from kernels.gemm.iluvatar.mr.common import (
    ATOM_K_B16,
    ATOM_M,
    ATOM_N,
    MR_GEMM_GEOM,
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


BLOCK_M = 128
BLOCK_N = 64
BLOCK_K = 32
WARP_M = 32
PACK_BF16 = 2
K_ATOMS = BLOCK_K // ATOM_K_B16

assert K_ATOMS == 2


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _triple(value, name: str) -> tuple[int, int, int]:
    if isinstance(value, int):
        return value, value, value
    if len(value) != 3:
        raise ValueError(f"{name} must be an int or a length-3 sequence")
    return tuple(int(v) for v in value)


def _output_shape(d, h, w, kt, kh, kw, stride, padding, dilation):
    # Input spatial [D, H, W] --> output spatial [Do, Ho, Wo].
    # kt/kh/kw are the filter T/R/S extents; st/pt/dt are the corresponding
    # stride/padding/dilation values along depth (and similarly for H/W).
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
    """Build the shape-specialized MR BypassSlb BF16 Conv3D launcher.

    Shape notation:
      input  x: [N, G*Cpg, D, H, W]
      weight w: [G*Kpg, Cpg, T, R, S]
      output y: [N, G*Kpg, Do, Ho, Wo]

    The compiled kernel receives the packed tensors created by
    ``prepare_conv3d_implicit_bypass_slb``, not these public NCDHW/KCTRS
    layouts.
    """

    if min(n, c_per_group, d, h, w, k_per_group, kt, kh, kw, groups) <= 0:
        raise ValueError("all Conv3D extents and groups must be positive")
    if min(st, sh, sw, dt, dh, dw) <= 0:
        raise ValueError("stride and dilation must be positive")

    stride = (st, sh, sw)
    padding = (pt, ph, pw)
    dilation = (dt, dh, dw)
    do, ho, wo = _output_shape(
        d, h, w, kt, kh, kw, stride, padding, dilation
    )

    block_n = 32 if k_per_group <= 32 else BLOCK_N
    warp_n = block_n
    warps_m = BLOCK_M // WARP_M
    block_threads = warps_m * WARP_SIZE
    warp_atoms_m = WARP_M // ATOM_M
    warp_atoms_n = warp_n // ATOM_N

    # Per-group Conv3D --> implicit GEMM:
    #   A: [M, K_reduction] @ B.T: [K_reduction, Kpg] --> C: [M, Kpg]
    #   M = N*Do*Ho*Wo, K_reduction = T*R*S*Cpg.
    # Cpg is padded to a BF16 pair because each activation load is one i32.
    c_padded = _ceil_div(c_per_group, PACK_BF16) * PACK_BF16
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

    # Logical GEMM dimensions --> physical padded workspace dimensions:
    #   [M, G*Kpg] --> [M_padded, G*N_padded].
    # Here N_padded means padded GEMM-N/output channels per group; it is not
    # the Conv3D batch dimension ``n``.
    m = n * do * ho * wo
    m_padded = _ceil_div(m, BLOCK_M) * BLOCK_M
    n_padded = _ceil_div(k_per_group, block_n) * block_n
    input_channels_padded = groups * c_padded
    output_channels_padded = groups * n_padded
    output_hw = ho * wo
    output_dhw = do * output_hw

    # B is the only shared-memory operand. One SME issue moves a 16x32
    # BF16 chunk; BN=64 therefore has four chunks, BN=32 has two.
    b_elems = block_n * BLOCK_K
    b_atoms_total = b_elems // MR_GEMM_GEOM.cta_chunk_elems
    b_loader_warps = min(warps_m, b_atoms_total)
    b_per_warp = b_atoms_total // b_loader_warps
    assert b_atoms_total % b_loader_warps == 0

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def conv3d_bypass_slb_bf16_kernel(
        x: fx.Tensor,
        weight: fx.Tensor,
        y: fx.Tensor,
    ):
        # Physical kernel arguments (all BF16):
        #   x:      [N, D, H, W, G*C_padded]       (packed NDHWC)
        #   weight: [G, N_padded, K_reduction_pad] (packed GEMM B)
        #   y:      [M_padded, G*N_padded]         (GEMM C workspace)
        tid = fx.thread_idx.x
        #import pdb; pdb.set_trace()
        # Grid [ceil(Kpg/block_n), G, ceil(M/BLOCK_M)] maps x/y/z to
        # output-channel tile, group, and flattened output-position tile.
        block_n_idx, group_id, block_m = fx.block_idx
        warp_id = tid // WARP_SIZE
        lane_id = tid % WARP_SIZE
        lane_row = lane_id // fx.Int32(ATOM_N)
        lane_col = lane_id % fx.Int32(ATOM_N)

        m_base = block_m * fx.Int32(BLOCK_M)
        n_base = block_n_idx * fx.Int32(block_n)
        warp_m_base = warp_id * fx.Int32(WARP_M)

        @fx.struct
        class ConvSmem:
            weight: fx.Array[fx.BFloat16, b_elems]

        smem_bf16 = (
            fx.SharedAllocator(static=True).allocate(ConvSmem).peek().weight.ptr
        )
        x_ptr = fx.get_iter(x)
        x_i32 = fx.recast_iter(
            fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global),
            x_ptr,
        )

        mma_atom = fx.make_mma_atom(
            ixdl.MRMma(
                ATOM_M,
                ATOM_N,
                ATOM_K_B16,
                fx.BFloat16,
                fx.BFloat16,
                fx.Float32,
            )
        )
        tiled_mma = fx.make_tiled_mma(
            mma_atom,
            fx.make_layout((1, 1, 1), (1, 1, 1)),
        )
        thr_mma = tiled_mma.thr_slice(lane_id)
        copy_atom_b = fx.make_copy_atom(
            fx.UniversalCopy32b(),
            fx.BFloat16,
        )
        thr_copy_b = fx.make_tiled_copy_B(
            copy_atom_b,
            tiled_mma,
        ).get_slice(lane_id)
        g2s_sme = mr_g2s_sme_config(
            a_mn_major=False,
            b_mn_major=False,
            elem_dtype=fx.BFloat16,
            row_atom=ixdl.MRAsyncCpRow16b,
            row_swizzle=ixdl.SMESwizzle.Row16b,
        )
        tile_smem_b = fx.make_tile(
            SMEM_ROWS,
            MR_GEMM_GEOM.values_per_sme_row,
        )

        # Each C fragment addresses an [ATOM_M, ATOM_N] tile in the physical
        # matrix y[M_padded, G*N_padded].
        accs = []
        for mma_m in fx.range_constexpr(warp_atoms_m):
            row_accs = []
            for mma_n in fx.range_constexpr(warp_atoms_n):
                c_ptr = fx.add_offset(
                    fx.get_iter(y),
                    fx.make_int_tuple(
                        (
                            m_base
                            + warp_m_base
                            + fx.Int32(mma_m * ATOM_M)
                        )
                        * fx.Int32(output_channels_padded)
                        + group_id * fx.Int32(n_padded)
                        + n_base
                        + fx.Int32(mma_n * ATOM_N)
                    ),
                )
                c_tile = fx.make_view(
                    c_ptr,
                    fx.make_layout(
                        (ATOM_M, ATOM_N),
                        (output_channels_padded, 1),
                    ),
                )
                acc = thr_mma.make_fragment_C(c_tile)
                acc.fill(0)
                row_accs.append(acc)
            accs.append(row_accs)

        for k_tile, _ in fx.range(k_tiles, init=[]):
            k_tile_i32 = fx.Int32(k_tile)
            k_base = k_tile_i32 * fx.Int32(BLOCK_K)
            # Packed K tile id --> (filter_d, channel tile, filter_w,
            # filter_h tile). This order exactly matches host weight packing.
            tile_h = k_tile_i32 % fx.Int32(loop_kh)
            tile_rem_h = k_tile_i32 // fx.Int32(loop_kh)
            tile_w = tile_rem_h % fx.Int32(loop_kw)
            tile_rem_w = tile_rem_h // fx.Int32(loop_kw)
            tile_c = tile_rem_w % fx.Int32(loop_xc)
            tile_d = tile_rem_w // fx.Int32(loop_xc)

            # B: async SME G2S, matching ixInfer's LoaderG2SB
            # (TRANSPOSE=true, ASYNC_G2S=true). Packed weights are K-major:
            # logical B(n,k) has stride (reduction_k_padded, 1), so use the
            # Col SME path selected by b_mn_major=False.
            weight_elem = (
                (
                    group_id * fx.Int32(n_padded)
                    + n_base
                )
                * fx.Int32(reduction_k_padded)
                + k_base
            )
            b_global_tile = fx.make_view(
                fx.add_offset(
                    fx.get_iter(weight),
                    fx.make_int_tuple(weight_elem),
                ),
                fx.make_layout(
                    (block_n, BLOCK_K),
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

            if fx.const_expr(b_loader_warps == warps_m):
                mr_gemm_g2s_issue_b_warp(
                    a_mn_major=False,
                    b_mn_major=False,
                    warp_id=warp_id,
                    b_per_warp=b_per_warp,
                    b_cta_gmem_view=b_cta_gmem_view,
                    g2s_sme=g2s_sme,
                    smem_b=smem_bf16,
                    elem_dtype=fx.BFloat16,
                    bm=BLOCK_M,
                    bn=block_n,
                    bk=BLOCK_K,
                    geom=MR_GEMM_GEOM,
                )
            else:
                # BN=32 has two SME chunks but the CTA still has four M
                # warps. Only warp 0/1 issue one B chunk each.
                if warp_id < fx.Int32(b_loader_warps):
                    mr_gemm_g2s_issue_b_warp(
                        a_mn_major=False,
                        b_mn_major=False,
                        warp_id=warp_id,
                        b_per_warp=b_per_warp,
                        b_cta_gmem_view=b_cta_gmem_view,
                        g2s_sme=g2s_sme,
                        smem_b=smem_bf16,
                        elem_dtype=fx.BFloat16,
                        bm=BLOCK_M,
                        bn=block_n,
                        bk=BLOCK_K,
                        geom=MR_GEMM_GEOM,
                    )
            ixdl.cp_async_commit_group()

            fx.gpu.barrier()

            # Four i32 loads contain eight BF16 values. The register reorder is
            # the BF16 branch of ixInfer XReuseLoadG2SInfo::LoadG2V.
            a_frags_lo = []
            a_frags_hi = []
            for mma_m in fx.range_constexpr(warp_atoms_m):
                packed_rows = []
                for row_i in fx.range_constexpr(4):
                    # BF16 LoadG2V uses offsets {0,1,8,9}; together with
                    # lane_row*2 these cover one 16-row MMA atom.
                    if fx.const_expr(row_i < 2):
                        row_offset = row_i
                    else:
                        row_offset = 6 + row_i
                    row = (
                        m_base
                        + warp_m_base
                        + fx.Int32(mma_m * ATOM_M)
                        + lane_row * fx.Int32(PACK_BF16)
                        + fx.Int32(row_offset)
                    )

                    fold_linear = lane_col * fx.Int32(PACK_BF16)
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

                    # GEMM row m --> Conv3D output coordinate [n, do, ho, wo].
                    # wo is fastest: m = ((n*Do + do)*Ho + ho)*Wo + wo.
                    out_n = row // fx.Int32(output_dhw)
                    out_spatial = row % fx.Int32(output_dhw)
                    out_d = out_spatial // fx.Int32(output_hw)
                    out_hw_rem = out_spatial % fx.Int32(output_hw)
                    out_h = out_hw_rem // fx.Int32(wo)
                    out_w = out_hw_rem % fx.Int32(wo)

                    # Output coordinate + filter coordinate --> input
                    # coordinate. This is cross-correlation (PyTorch Conv3D):
                    # the T/R/S filter axes are not reversed.
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
                    # Packed x index for layout [N, D, H, W, G*C_padded].
                    # The current group selects its contiguous C_padded slice.
                    x_elem = (
                        (
                            (
                                (
                                    out_n * fx.Int32(d) + in_d
                                )
                                * fx.Int32(h)
                                + in_h
                            )
                            * fx.Int32(w)
                            + in_w
                        )
                        * fx.Int32(input_channels_padded)
                        + group_id * fx.Int32(c_padded)
                        + channel
                    )
                    safe_pack = arith.select(
                        valid,
                        x_elem // fx.Int32(PACK_BF16),
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
                sr0 = (
                    (r0 & fx.Int32(0xFFFF))
                    | ((r1 & fx.Int32(0xFFFF)) << fx.Int32(16))
                )
                sr1 = (
                    (r2 & fx.Int32(0xFFFF))
                    | ((r3 & fx.Int32(0xFFFF)) << fx.Int32(16))
                )
                sr2 = (
                    (r0 >> fx.Int32(16))
                    | (r1 & fx.Int32(-0x10000))
                )
                sr3 = (
                    (r2 >> fx.Int32(16))
                    | (r3 & fx.Int32(-0x10000))
                )

                a_anchor = fx.add_offset(
                    x_ptr,
                    fx.make_int_tuple(fx.Int32(0)),
                )
                a_tile = fx.make_view(
                    a_anchor,
                    fx.make_layout(
                        (ATOM_M, ATOM_K_B16),
                        (ATOM_K_B16, 1),
                    ),
                )
                frag_a_lo = thr_mma.make_fragment_A(a_tile)
                frag_a_hi = thr_mma.make_fragment_A(a_tile)
                frag_a_lo.store(
                    Vec.from_elements(
                        [sr0, sr1],
                        fx.Int32,
                    ).bitcast(fx.BFloat16)
                )
                frag_a_hi.store(
                    Vec.from_elements(
                        [sr2, sr3],
                        fx.Int32,
                    ).bitcast(fx.BFloat16)
                )
                a_frags_lo.append(frag_a_lo)
                a_frags_hi.append(frag_a_hi)

            for mma_k in fx.range_constexpr(K_ATOMS):
                b_frags = []
                for mma_n in fx.range_constexpr(warp_atoms_n):
                    b_tile = mr_gemm_s2r_b_tile(
                        a_mn_major=False,
                        b_mn_major=False,
                        mma_n=mma_n,
                        mma_k=mma_k,
                        g2s_sme=g2s_sme,
                        smem_b=smem_bf16,
                        elem_dtype=fx.BFloat16,
                        warp_n_id=fx.Int32(0),
                        warp_atoms_n=warp_atoms_n,
                        bm=BLOCK_M,
                        bn=block_n,
                        bk=BLOCK_K,
                        geom=MR_GEMM_GEOM,
                    )
                    frag_b = mr_gemm_s2r_copy_b(
                        copy_atom=copy_atom_b,
                        thr_copy_b=thr_copy_b,
                        thr_mma=thr_mma,
                        smem_b_tile=b_tile,
                    )
                    b_frags.append(frag_b)

                for mma_m in fx.range_constexpr(warp_atoms_m):
                    if fx.const_expr(mma_k == 0):
                        frag_a = a_frags_lo[mma_m]
                    else:
                        frag_a = a_frags_hi[mma_m]
                    for mma_n in fx.range_constexpr(warp_atoms_n):
                        fx.gemm(
                            mma_atom,
                            accs[mma_m][mma_n],
                            frag_a,
                            b_frags[mma_n],
                            accs[mma_m][mma_n],
                        )

            fx.gpu.barrier()

        # FP32 accumulator --> BF16 GEMM workspace
        # [M_padded, G*N_padded]. Host unpacking later removes both paddings
        # and converts this matrix back to public NCDHW.
        y_ptr = fx.get_iter(y)
        for mma_m in fx.range_constexpr(warp_atoms_m):
            for mma_n in fx.range_constexpr(warp_atoms_n):
                values = Vec(accs[mma_m][mma_n].load())
                for elem_i in fx.range_constexpr(4):
                    out_row = (
                        m_base
                        + warp_m_base
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
                        values[elem_i].to(fx.BFloat16),
                        fx.add_offset(y_ptr, fx.make_int_tuple(out_offset)),
                    )

    # grid.x --> per-group output-channel tiles
    # grid.y --> groups
    # grid.z --> flattened [N, Do, Ho, Wo] output-position tiles
    grid = (
        n_padded // block_n,
        groups,
        m_padded // BLOCK_M,
    )
    block = (block_threads, 1, 1)

    @flyc.jit
    def launch(
        x: fx.Tensor,
        weight: fx.Tensor,
        y: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        conv3d_bypass_slb_bf16_kernel(x, weight, y).launch(
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
    """Pack public Conv3D tensors and allocate the GEMM output workspace.

    Public layouts:
      x:      [N, C, D, H, W], where C = groups*C_per_group
      weight: [K, C_per_group, T, R, S], where K = groups*K_per_group
      output: [N, K, Do, Ho, Wo]
    """

    if x.ndim != 5 or weight.ndim != 5:
        raise ValueError("x and weight must be 5D NCDHW/KCTRS tensors")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("x and weight must have torch.bfloat16 dtype")
    if x.device != weight.device:
        raise ValueError("x and weight must be on the same device")

    # Public tensor dimensions:
    #   x      [N, C, D, H, W]
    #   weight [K, C_per_group, T, R, S]
    n, c_total, d, h, w = (int(v) for v in x.shape)
    k_total, c_per_group, kt, kh, kw = (int(v) for v in weight.shape)
    if groups <= 0 or c_total % groups or k_total % groups:
        raise ValueError("input/output channels must be divisible by groups")
    if c_total // groups != c_per_group:
        raise ValueError("weight.shape[1] must equal input_channels/groups")

    stride = _triple(stride, "stride")
    padding = _triple(padding, "padding")
    dilation = _triple(dilation, "dilation")
    do, ho, wo = _output_shape(
        d, h, w, kt, kh, kw, stride, padding, dilation
    )

    # G is represented explicitly during packing. Cpg/Kpg remain the channel
    # extents within one group.
    k_per_group = k_total // groups
    block_n = 32 if k_per_group <= 32 else BLOCK_N
    block_threads = (BLOCK_M // WARP_M) * WARP_SIZE
    c_padded = _ceil_div(c_per_group, PACK_BF16) * PACK_BF16
    fold_c = c_padded <= BLOCK_K // 2
    n_padded = _ceil_div(k_per_group, block_n) * block_n
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

    # Activation layout transitions:
    #   [N, C, D, H, W]
    #   --> [N, G, Cpg, D, H, W]
    #   --> [N, D, H, W, G, Cpg]
    #   --> [N, D, H, W, G, C_padded]
    #   --> [N, D, H, W, G*C_padded].
    x_grouped = x.reshape(
        n, groups, c_per_group, d, h, w
    ).permute(0, 3, 4, 5, 1, 2)
    x_packed_6d = torch.zeros(
        (n, d, h, w, groups, c_padded),
        dtype=torch.bfloat16,
        device=x.device,
    )
    x_packed_6d[..., :c_per_group] = x_grouped
    x_packed = x_packed_6d.reshape(
        n, d, h, w, groups * c_padded
    ).contiguous()

    # Filter layout transitions:
    #   [K, Cpg, T, R, S]
    #   --> [G, Kpg, Cpg, T, R, S]
    #   --> [G, Kpg, T, R, S, Cpg]
    #   --> [G, N_padded, T, R, S, C_padded].
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
        dtype=torch.bfloat16,
        device=weight.device,
    )
    weight_channels_padded[
        :, :k_per_group, ..., :c_per_group
    ] = weight_grouped

    # Fold/reorder [T, R, S, C_padded] into BLOCK_K-sized reduction
    # tiles. The resulting GEMM-B layout is K-major:
    #   [G, N_padded, T, R, S, C_padded]
    #   --> [G, N_padded, K_tiles, BLOCK_K]
    #   --> [G, N_padded, K_reduction_padded].
    weight_packed = torch.zeros(
        (groups, n_padded, k_tiles, BLOCK_K),
        dtype=torch.bfloat16,
        device=weight.device,
    )
    tile_id = 0
    dwords_per_xc = fold_xc // PACK_BF16
    for kernel_d in range(kt):
        for loop_c_id in range(loop_xc):
            for loop_w_id in range(loop_kw):
                for loop_h_id in range(loop_kh):
                    for fold_id in range(BLOCK_K):
                        channel_high = fold_id // (
                            BLOCK_K // PACK_BF16
                        )
                        fold_low = fold_id % (
                            BLOCK_K // PACK_BF16
                        )
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
                                + channel_low * PACK_BF16
                                + channel_high
                            )
                        else:
                            fold_h_id = 0
                            kernel_w = loop_w_id
                            kernel_h = loop_h_id
                            channel = (
                                loop_c_id * BLOCK_K
                                + fold_low * PACK_BF16
                                + channel_high
                            )
                        if (
                            kernel_h < kh
                            and kernel_w < kw
                            and channel < c_padded
                            and fold_h_id < fold_kh
                        ):
                            weight_packed[
                                :, :, tile_id, fold_id
                            ] = weight_channels_padded[
                                :,
                                :,
                                kernel_d,
                                kernel_h,
                                kernel_w,
                                channel,
                            ]
                    tile_id += 1
    weight_packed = weight_packed.reshape(
        groups,
        n_padded,
        reduction_k_padded,
    )

    # Kernel output is a padded implicit-GEMM matrix, not public NCDHW:
    #   logical [N, Do, Ho, Wo, G, Kpg]
    #   --> flatten [M=N*Do*Ho*Wo, G*Kpg]
    #   --> pad     [M_padded, G*N_padded], where
    #       M_padded = ceil_div(M, BLOCK_M) * BLOCK_M
    #       N_padded = ceil_div(Kpg, block_n) * block_n (per group).
    # Element correspondence:
    #   matrix_row = ((n_idx*Do + do_idx)*Ho + ho_idx)*Wo + wo_idx
    #   matrix_col = group_idx*N_padded + k_within_group
    # Padding exists at rows [M, M_padded) and at the tail of each group's
    # output-channel slice [Kpg, N_padded).
    y_workspace = torch.empty(
        (m_padded, groups * n_padded),
        dtype=torch.bfloat16,
        device=x.device,
    )
    meta = {
        "shape": (
            n,
            c_per_group,
            d,
            h,
            w,
            k_per_group,
            kt,
            kh,
            kw,
        ),
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
        "output_shape": (do, ho, wo),
        "m": m,
        "m_padded": m_padded,
        "n_padded": n_padded,
        "block_n": block_n,
        "block_threads": block_threads,
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
    # Output layout transitions:
    #   [M_padded, G*N_padded]
    #   --> [N, Do, Ho, Wo, G, N_padded]
    #   --> [N, Do, Ho, Wo, K]
    #   --> [N, K, Do, Ho, Wo].
    y = y_workspace[:m].view(
        n, do, ho, wo, groups, n_padded
    )
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
    """Run BF16 Conv3D: NCDHW/KCTRS --> implicit GEMM --> NCDHW.

    Public input/output layouts:
      x:      [N, C, D, H, W], C = G*Cpg
      weight: [K, Cpg, T, R, S], K = G*Kpg
      output: [N, K, Do, Ho, Wo]

    For each group, Conv3D is represented as the conceptual GEMM
      A[M, K_reduction] @ B.T[K_reduction, Kpg] = C_gemm[M, Kpg],
    where M = N*Do*Ho*Wo and K_reduction = T*R*S*Cpg. The implicit
    activation matrix A is never materialized: the kernel decodes each
    GEMM row/reduction coordinate back to x[n, c, d, h, w].
    """

    # Step 1: convert public tensors to the physical kernel layouts.
    #
    # Activation:
    #   x [N, C, D, H, W]
    #   --> group/permute/pad [N, D, H, W, G, C_padded]
    #   --> xp [N, D, H, W, G*C_padded].
    # xp supplies implicit-GEMM A. It is still an activation tensor, not a
    # materialized im3col matrix [M, K_reduction].
    #
    # Filter:
    #   weight [K, Cpg, T, R, S]
    #   --> [G, Kpg, T, R, S, Cpg]
    #   --> pad/reorder [G, N_padded, K_tiles, BLOCK_K]
    #   --> wp [G, N_padded, K_reduction_padded].
    # wp is physical GEMM B stored as [G, padded-GEMM-N, padded-GEMM-K].
    #
    # Output workspace:
    #   conceptual C_gemm [M, G*Kpg]
    #   --> yw [M_padded, G*N_padded].
    xp, wp, yw, meta = prepare_conv3d_implicit_bypass_slb(
        x,
        weight,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    # Step 2: extract compile-time Conv3D extents. The cached compiler uses
    # these scalar values to specialize output-coordinate decoding, packed
    # strides, tile counts, and the launch grid for this exact shape.
    n, c_per_group, d, h, w, k_per_group, kt, kh, kw = meta["shape"]
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
        *meta["stride"],
        *meta["padding"],
        *meta["dilation"],
        groups,
    )

    # Step 3: execute the implicit GEMM:
    # xp: [N, D, H, W, G*C_padded]
    # wp: [G, N_padded, K_reduction_padded] # K_reduction_padded -> TRSC
    # yw: [M_padded, G*N_padded]            # N_padded -> K
    #
    #   xp (implicit A) @ wp.T (packed B)
    #   --> yw [M_padded, G*N_padded].
    # During execution, a GEMM row is decoded as
    #   m --> (n, do, ho, wo),
    # and a reduction position is decoded as
    #   k_reduction --> (filter_d, filter_h, filter_w, channel).
    # These coordinates select xp directly, so no [M, K_reduction] activation
    # matrix is allocated.
    launch(
        xp,
        wp,
        yw,
        torch.cuda.current_stream() if stream is None else stream,
    )

    # Step 4: remove GEMM padding and restore the public output layout:
    #   yw [M_padded, G*N_padded]
    #   --> [N, Do, Ho, Wo, G, N_padded]
    #   --> crop channels [N, Do, Ho, Wo, G, Kpg]
    #   --> merge G*Kpg [N, Do, Ho, Wo, K]
    #   --> permute/contiguous [N, K, Do, Ho, Wo].
    return unpack_conv3d_implicit_bypass_slb_output(yw, meta)


__all__ = [
    "BLOCK_K",
    "BLOCK_M",
    "BLOCK_N",
    "compile_conv3d_implicit_bypass_slb",
    "conv3d_implicit_bypass_slb",
    "prepare_conv3d_implicit_bypass_slb",
    "unpack_conv3d_implicit_bypass_slb_output",
]
