# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar MR BF16 implicit-GEMM Conv3D without SME global-to-shared copies.

The kernel follows ixInfer's ``implConvolutionTcuNoSmeKernel`` BF16 geometry:

* input/output are packed NDHWC internally;
* GEMM is ``[N*Do*Ho*Wo, T*R*S*C] @ [K, T*R*S*C].T``;
* CTA tile is 64x32x32 for K<=32, otherwise 64x64x32;
* every warp computes 32x32 using 16x16x16 BF16 MR MMA atoms;
* global-to-shared copies use ordinary 32-bit loads/stores, not SME;
* accumulation is FP32 and the epilogue converts the result to BF16.

The public ``conv3d_implicit_no_sme`` wrapper accepts PyTorch-style NCDHW input
and KCTRS weights. It implements cross-correlation, matching PyTorch and the
ixInfer NoSme kernel. Groups, stride, padding, and dilation are supported.
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
from kernels.gemm.iluvatar.mr.common import ATOM_K_B16, ATOM_M, ATOM_N


BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32
WARP_M = 32
WARP_N = 32
PACK_BF16 = 2

assert (BLOCK_M, BLOCK_N, BLOCK_K) == (64, 64, 32)


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
def compile_conv3d_implicit_no_sme(
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
    """Build the MR NoSme BF16 Conv3D launcher.

    Runtime tensors use the packed contract produced by
    :func:`prepare_conv3d_implicit_no_sme`:

    * ``x``: contiguous ``[N,D,H,W,groups*Cpad]`` BF16;
    * ``weight``: contiguous ``[groups,Npad,KredPad]`` BF16;
    * ``y``: contiguous ``[Mpad,groups*Npad]`` BF16 workspace.
    """

    if min(n, c_per_group, d, h, w, k_per_group, kt, kh, kw, groups) <= 0:
        raise ValueError("all Conv3D extents and groups must be positive")
    if min(st, sh, sw, dt, dh, dw) <= 0:
        raise ValueError("stride and dilation must be positive")

    stride = (st, sh, sw)
    padding = (pt, ph, pw)
    dilation = (dt, dh, dw)
    do, ho, wo = _output_shape(d, h, w, kt, kh, kw, stride, padding, dilation)

    # ixInfer uses BN=32 for small output-channel counts and BN=64 otherwise.
    block_n = 32 if k_per_group <= 32 else BLOCK_N
    warps_m = BLOCK_M // WARP_M
    warps_n = block_n // WARP_N
    block_threads = warps_m * warps_n * WARP_SIZE
    warp_atoms_m = WARP_M // ATOM_M
    warp_atoms_n = WARP_N // ATOM_N
    k_atoms = BLOCK_K // ATOM_K_B16

    c_padded = _ceil_div(c_per_group, PACK_BF16) * PACK_BF16
    reduction_k = kt * kh * kw * c_padded
    reduction_k_padded = _ceil_div(reduction_k, BLOCK_K) * BLOCK_K
    m = n * do * ho * wo
    m_padded = _ceil_div(m, BLOCK_M) * BLOCK_M
    n_padded = _ceil_div(k_per_group, block_n) * block_n
    input_channels_padded = groups * c_padded
    output_channels_padded = groups * n_padded
    output_hw = ho * wo
    output_dhw = do * output_hw
    k_tiles = reduction_k_padded // BLOCK_K

    a_elems = BLOCK_M * BLOCK_K
    b_elems = block_n * BLOCK_K
    smem_elems = a_elems + b_elems
    a_packs = a_elems // PACK_BF16
    b_packs = b_elems // PACK_BF16
    a_pack_iters = a_packs // block_threads
    b_pack_iters = b_packs // block_threads
    assert a_packs % block_threads == 0
    assert b_packs % block_threads == 0

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def conv3d_no_sme_bf16_kernel(x: fx.Tensor, weight: fx.Tensor, y: fx.Tensor):
        tid = fx.thread_idx.x
        block_n_idx, group_id, block_m = fx.block_idx
        warp_id = tid // WARP_SIZE
        lane_id = tid % WARP_SIZE
        warp_m_id = warp_id // warps_n
        warp_n_id = warp_id % warps_n
        lane_row = lane_id // fx.Int32(ATOM_N)
        lane_col = lane_id % fx.Int32(ATOM_N)

        m_base = block_m * fx.Int32(BLOCK_M)
        n_base = block_n_idx * fx.Int32(block_n)

        @fx.struct
        class ConvSmem:
            buf: fx.Array[fx.BFloat16, smem_elems]

        smem_bf16 = fx.SharedAllocator(static=True).allocate(ConvSmem).peek().buf.ptr
        smem_i32 = fx.recast_iter(
            fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared),
            smem_bf16,
        )
        x_i32 = fx.recast_iter(
            fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global),
            fx.get_iter(x),
        )
        weight_i32 = fx.recast_iter(
            fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global),
            fx.get_iter(weight),
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
        # Plain row-major SLB is not SME-swizzled, so use the scalar BF16 S2R
        # atom from the direct MR MMA device test.
        copy_atom_a = fx.make_copy_atom(fx.UniversalCopy16b(), fx.BFloat16)
        copy_atom_b = fx.make_copy_atom(fx.UniversalCopy16b(), fx.BFloat16)
        thr_copy_a = fx.make_tiled_copy_A(copy_atom_a, tiled_mma).get_slice(lane_id)
        thr_copy_b = fx.make_tiled_copy_B(copy_atom_b, tiled_mma).get_slice(lane_id)

        y_base = (
            (m_base + warp_m_id * fx.Int32(WARP_M)) * fx.Int32(output_channels_padded)
            + group_id * fx.Int32(n_padded)
            + n_base
            + warp_n_id * fx.Int32(WARP_N)
        )
        accs = []
        for mma_m in fx.range_constexpr(warp_atoms_m):
            row_accs = []
            for mma_n in fx.range_constexpr(warp_atoms_n):
                c_ptr = fx.add_offset(
                    fx.get_iter(y),
                    fx.make_int_tuple(
                        y_base
                        + fx.Int32(mma_m * ATOM_M * output_channels_padded)
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

        def _smem_view(elem_offset, shape, stride):
            ptr = fx.add_offset(smem_bf16, fx.make_int_tuple(fx.Int32(elem_offset)))
            return fx.make_view(ptr, fx.make_layout(shape, stride))

        # for k_tile in fx.range_constexpr(k_tiles): # range_constexpr will unroll k_tiles times!

        # Keep the potentially large reduction loop in runtime IR. The inner
        # copy/MMA loops remain constexpr-unrolled because their trip counts are
        # small fixed CTA geometry constants.
        for k_tile, _ in fx.range(k_tiles, init=[]):
            k_base = fx.Int32(k_tile * BLOCK_K)

            # A: implicit im3col. One packed i32 load moves two adjacent BF16s.
            for load_i in fx.range_constexpr(a_pack_iters):
                pack_id = tid + fx.Int32(load_i * block_threads)
                local_m = pack_id // fx.Int32(BLOCK_K // PACK_BF16)
                local_k = (pack_id % fx.Int32(BLOCK_K // PACK_BF16)) * fx.Int32(PACK_BF16)
                row = m_base + local_m
                k_abs = k_base + local_k

                out_n = row // fx.Int32(output_dhw)
                out_spatial = row % fx.Int32(output_dhw)
                out_d = out_spatial // fx.Int32(output_hw)
                out_hw_rem = out_spatial % fx.Int32(output_hw)
                out_h = out_hw_rem // fx.Int32(wo)
                out_w = out_hw_rem % fx.Int32(wo)

                channel = k_abs % fx.Int32(c_padded)
                kernel_linear = k_abs // fx.Int32(c_padded)
                kernel_w = kernel_linear % fx.Int32(kw)
                kernel_hr = kernel_linear // fx.Int32(kw)
                kernel_h = kernel_hr % fx.Int32(kh)
                kernel_d = kernel_hr // fx.Int32(kh)

                in_d = out_d * fx.Int32(st) - fx.Int32(pt) + kernel_d * fx.Int32(dt)
                in_h = out_h * fx.Int32(sh) - fx.Int32(ph) + kernel_h * fx.Int32(dh)
                in_w = out_w * fx.Int32(sw) - fx.Int32(pw) + kernel_w * fx.Int32(dw)
                valid = (
                    (row < fx.Int32(m))
                    & (k_abs < fx.Int32(reduction_k))
                    & (in_d >= fx.Int32(0))
                    & (in_d < fx.Int32(d))
                    & (in_h >= fx.Int32(0))
                    & (in_h < fx.Int32(h))
                    & (in_w >= fx.Int32(0))
                    & (in_w < fx.Int32(w))
                )
                x_elem = (
                    ((((out_n * fx.Int32(d) + in_d) * fx.Int32(h) + in_h) * fx.Int32(w) + in_w)
                    * fx.Int32(input_channels_padded))
                    + group_id * fx.Int32(c_padded)
                    + channel
                )
                safe_pack = arith.select(valid, x_elem // fx.Int32(PACK_BF16), fx.Int32(0))
                raw = fx.Int32(
                    fx.ptr_load(
                        fx.add_offset(x_i32, fx.make_int_tuple(safe_pack)),
                    )
                )
                packed = arith.select(valid, raw, fx.Int32(0))
                fx.ptr_store(
                    packed,
                    fx.add_offset(smem_i32, fx.make_int_tuple(pack_id)),
                )

            # B: packed [group, output-channel, reduction-K].
            for load_i in fx.range_constexpr(b_pack_iters):
                pack_id = tid + fx.Int32(load_i * block_threads)
                local_n = pack_id // fx.Int32(BLOCK_K // PACK_BF16)
                local_k = (pack_id % fx.Int32(BLOCK_K // PACK_BF16)) * fx.Int32(PACK_BF16)
                weight_elem = (
                    ((group_id * fx.Int32(n_padded) + n_base + local_n)
                    * fx.Int32(reduction_k_padded))
                    + k_base
                    + local_k
                )
                raw = fx.Int32(
                    fx.ptr_load(
                        fx.add_offset(
                            weight_i32,
                            fx.make_int_tuple(weight_elem // fx.Int32(PACK_BF16)),
                        )
                    )
                )
                fx.ptr_store(
                    raw,
                    fx.add_offset(
                        smem_i32,
                        fx.make_int_tuple(fx.Int32(a_packs) + pack_id),
                    ),
                )

            fx.gpu.barrier()

            warp_a_offset = warp_m_id * fx.Int32(WARP_M * BLOCK_K)
            warp_b_offset = warp_n_id * fx.Int32(WARP_N * BLOCK_K)
            for mma_k in fx.range_constexpr(k_atoms):
                a_frags = []
                for mma_m in fx.range_constexpr(warp_atoms_m):
                    a_tile = _smem_view(
                        warp_a_offset
                        + fx.Int32(mma_m * ATOM_M * BLOCK_K)
                        + fx.Int32(mma_k * ATOM_K_B16),
                        (ATOM_M, ATOM_K_B16),
                        (BLOCK_K, 1),
                    )
                    frag_a = thr_mma.make_fragment_A(a_tile)
                    fx.copy(
                        copy_atom_a,
                        thr_copy_a.partition_S(a_tile),
                        thr_copy_a.retile(frag_a),
                        pred=None,
                    )
                    a_frags.append(frag_a)

                b_frags = []
                for mma_n in fx.range_constexpr(warp_atoms_n):
                    b_tile = _smem_view(
                        fx.Int32(a_elems)
                        + warp_b_offset
                        + fx.Int32(mma_n * ATOM_N * BLOCK_K)
                        + fx.Int32(mma_k * ATOM_K_B16),
                        (ATOM_N, ATOM_K_B16),
                        (BLOCK_K, 1),
                    )
                    frag_b = thr_mma.make_fragment_B(b_tile)
                    fx.copy(
                        copy_atom_b,
                        thr_copy_b.partition_S(b_tile),
                        thr_copy_b.retile(frag_b),
                        pred=None,
                    )
                    b_frags.append(frag_b)

                for mma_m in fx.range_constexpr(warp_atoms_m):
                    for mma_n in fx.range_constexpr(warp_atoms_n):
                        fx.gemm(
                            mma_atom,
                            accs[mma_m][mma_n],
                            a_frags[mma_m],
                            b_frags[mma_n],
                            accs[mma_m][mma_n],
                        )

            # Stage=1: all warps must finish reading before the next overwrite.
            fx.gpu.barrier()

        # FP32 accumulator -> BF16 output, matching the no-fuse BF16 path.
        y_ptr = fx.get_iter(y)
        for mma_m in fx.range_constexpr(warp_atoms_m):
            for mma_n in fx.range_constexpr(warp_atoms_n):
                values = Vec(accs[mma_m][mma_n].load())
                for elem_i in fx.range_constexpr(4):
                    out_row = (
                        m_base
                        + warp_m_id * fx.Int32(WARP_M)
                        + fx.Int32(mma_m * ATOM_M)
                        + lane_row
                        + fx.Int32(elem_i * 4)
                    )
                    out_col = (
                        group_id * fx.Int32(n_padded)
                        + n_base
                        + warp_n_id * fx.Int32(WARP_N)
                        + fx.Int32(mma_n * ATOM_N)
                        + lane_col
                    )
                    out_offset = out_row * fx.Int32(output_channels_padded) + out_col
                    fx.ptr_store(
                        values[elem_i].to(fx.BFloat16),
                        fx.add_offset(y_ptr, fx.make_int_tuple(out_offset)),
                    )

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
        conv3d_no_sme_bf16_kernel(x, weight, y).launch(
            grid=grid,
            block=block,
            stream=stream,
        )

    return launch


def prepare_conv3d_implicit_no_sme(
    x,
    weight,
    *,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    """Pack NCDHW/KCTRS BF16 tensors and allocate the padded output workspace."""

    if x.ndim != 5 or weight.ndim != 5:
        raise ValueError("x and weight must be 5D NCDHW/KCTRS tensors")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("x and weight must have torch.bfloat16 dtype")
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
    block_n = 32 if k_per_group <= 32 else BLOCK_N
    block_threads = (BLOCK_M // WARP_M) * (block_n // WARP_N) * WARP_SIZE
    c_padded = _ceil_div(c_per_group, PACK_BF16) * PACK_BF16
    n_padded = _ceil_div(k_per_group, block_n) * block_n
    reduction_k = kt * kh * kw * c_padded
    reduction_k_padded = _ceil_div(reduction_k, BLOCK_K) * BLOCK_K
    m = n * do * ho * wo
    m_padded = _ceil_div(m, BLOCK_M) * BLOCK_M

    x_grouped = x.reshape(n, groups, c_per_group, d, h, w).permute(0, 3, 4, 5, 1, 2)
    x_packed_6d = torch.zeros(
        (n, d, h, w, groups, c_padded),
        dtype=torch.bfloat16,
        device=x.device,
    )
    x_packed_6d[..., :c_per_group] = x_grouped
    x_packed = x_packed_6d.reshape(n, d, h, w, groups * c_padded).contiguous()

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
    weight_channels_padded[:, :k_per_group, ..., :c_per_group] = weight_grouped
    weight_packed = torch.zeros(
        (groups, n_padded, reduction_k_padded),
        dtype=torch.bfloat16,
        device=weight.device,
    )
    weight_packed[..., :reduction_k] = weight_channels_padded.reshape(
        groups,
        n_padded,
        reduction_k,
    )
    y_workspace = torch.empty(
        (m_padded, groups * n_padded),
        dtype=torch.bfloat16,
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
        "block_n": block_n,
        "block_threads": block_threads,
        "reduction_k": reduction_k,
        "reduction_k_padded": reduction_k_padded,
    }
    return x_packed, weight_packed, y_workspace, meta


def unpack_conv3d_implicit_no_sme_output(y_workspace, meta):
    """Convert the padded matrix output to contiguous NCDHW."""

    n, _, _, _, _, k_per_group, _, _, _ = meta["shape"]
    do, ho, wo = meta["output_shape"]
    groups = meta["groups"]
    m = meta["m"]
    n_padded = meta["n_padded"]
    y = y_workspace[:m].view(n, do, ho, wo, groups, n_padded)
    y = y[..., :k_per_group].reshape(n, do, ho, wo, groups * k_per_group)
    return y.permute(0, 4, 1, 2, 3).contiguous()


def conv3d_implicit_no_sme(
    x,
    weight,
    *,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    stream=None,
):
    """Run BF16 Conv3D with FP32 accumulation and return BF16 NCDHW."""

    x_packed, weight_packed, y_workspace, meta = prepare_conv3d_implicit_no_sme(
        x,
        weight,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    n, c_per_group, d, h, w, k_per_group, kt, kh, kw = meta["shape"]
    st, sh, sw = meta["stride"]
    pt, ph, pw = meta["padding"]
    dt, dh, dw = meta["dilation"]
    launch = compile_conv3d_implicit_no_sme(
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
    return unpack_conv3d_implicit_no_sme_output(y_workspace, meta)


__all__ = [
    "BLOCK_K",
    "BLOCK_M",
    "BLOCK_N",
    "compile_conv3d_implicit_no_sme",
    "conv3d_implicit_no_sme",
    "prepare_conv3d_implicit_no_sme",
    "unpack_conv3d_implicit_no_sme_output",
]
