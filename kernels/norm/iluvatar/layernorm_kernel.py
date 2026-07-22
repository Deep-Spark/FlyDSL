# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar ivcore11 LayerNorm kernels (forward).

LayerNorm(x) = (x - mean) / sqrt(var + eps) * gamma + beta

Variants:
  - basic / fused-add
  - dynamicquant / smoothquant (i8 output + per-row YScale)
  - fused-add + quant

I/O uses UniversalCopy. Affine uses math.fma. Warp size is 64.

Vector path triggers when N % (BLOCK_THREADS * vec_width) == 0.
Default vec_width is 4 for f16/bf16 (best on ivcore11 vs 8); f32 stays scalar.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.vector import ReductionOp, full
from kernels.gemm.iluvatar.common import WARP_SIZE

KERNEL_NAME = "iluvatar_layernorm"

EPS = 1e-5
BLOCK_THREADS = 256
# Empirically on ivcore11, f16/bf16 prefer 64-bit loads (vec=4) over 128-bit (vec=8).
# f32 vector path is currently slower than scalar, so default keeps scalar there.
VEC_WIDTH_F16 = 4


def _default_vec_width(elem_bits: int) -> int:
    if elem_bits <= 16:
        return VEC_WIDTH_F16
    return 0


def _vec_num_tiles(N: int, vec_width: int) -> int:
    """Tiles per row when every thread always has a valid vector (no mask)."""
    if vec_width <= 0:
        return 0
    tile = BLOCK_THREADS * vec_width
    if N % tile != 0:
        return 0
    return N // tile


def _dtype_to_elem_type(dtype_str: str):
    # Local map: avoid kernels_common (imports ROCDL buffer_ops).
    if dtype_str == "f32":
        return fx.Float32
    if dtype_str == "f16":
        return fx.Float16
    if dtype_str == "bf16":
        return fx.BFloat16
    raise ValueError(f"unsupported dtype: {dtype_str!r} (expected 'f32', 'f16', or 'bf16')")


def _quant_dtype_to_elem_type(dtype_str: str):
    if dtype_str in ("i8", "int8"):
        return fx.Int8
    raise ValueError(f"unsupported quant dtype: {dtype_str!r} (expected 'i8' or 'int8')")


def _quant_dtype_max(dtype_str: str) -> float:
    if dtype_str in ("i8", "int8"):
        return 127.0
    raise ValueError(f"unsupported quant dtype: {dtype_str!r} (expected 'i8' or 'int8')")


def _make_reduction_storage(num_warps: int):
    @fx.struct
    class SharedStorage:
        s_sum: fx.Array[fx.Float32, num_warps, 16]
        s_sumsq: fx.Array[fx.Float32, num_warps, 16]

    return SharedStorage


def _copy_atom_for_bits(elem_bits: int, vec_elems: int = 1):
    bit_size = elem_bits * vec_elems
    if bit_size == 128:
        return fx.make_copy_atom(fx.UniversalCopy128b(), elem_bits)
    if bit_size == 64:
        return fx.make_copy_atom(fx.UniversalCopy64b(), elem_bits)
    if bit_size == 32:
        return fx.make_copy_atom(fx.UniversalCopy32b(), elem_bits)
    if bit_size == 16:
        return fx.make_copy_atom(fx.UniversalCopy16b(), elem_bits)
    if bit_size == 8:
        return fx.make_copy_atom(fx.UniversalCopy8b(), elem_bits)
    raise ValueError(f"unsupported copy bit width: {bit_size}")


def _load_scalar(copy_atom, elem_dtype, divided_tensor, index):
    view = fx.slice(divided_tensor, (None, index))
    r = fx.make_rmem_tensor(1, elem_dtype)
    fx.copy_atom_call(copy_atom, view, r)
    return fx.memref_load_vec(r)[0]


def _store_scalar(copy_atom, elem_dtype, divided_tensor, index, val):
    r = fx.make_rmem_tensor(1, elem_dtype)
    ts = full(1, elem_dtype(val), elem_dtype)
    fx.memref_store_vec(ts, r)
    view = fx.slice(divided_tensor, (None, index))
    fx.copy_atom_call(copy_atom, r, view)


def _load_vec(copy_atom, vec_width, elem_dtype, div_tensor, idx):
    r = fx.make_rmem_tensor(vec_width, elem_dtype)
    fx.copy_atom_call(copy_atom, fx.slice(div_tensor, (None, idx)), r)
    return fx.memref_load_vec(r)


def _store_vec(copy_atom, vec_width, elem_dtype, val, div_tensor, idx):
    r = fx.make_rmem_tensor(vec_width, elem_dtype)
    fx.memref_store_vec(val, r)
    fx.copy_atom_call(copy_atom, r, fx.slice(div_tensor, (None, idx)))


def _store_yscale(scale_copy_atom, yscale_div, index, val):
    r = fx.make_rmem_tensor(1, fx.Float32)
    ts = full(1, fx.Float32(val), fx.Float32)
    fx.memref_store_vec(ts, r)
    fx.copy_atom_call(scale_copy_atom, r, fx.slice(yscale_div, (None, index)))


def _to_elem_scalar(dtype_str: str, elem_dtype, y):
    if const_expr(dtype_str == "f32"):
        return y
    return y.to(elem_dtype)


def _to_elem_vec(dtype_str: str, elem_dtype, y):
    if const_expr(dtype_str == "f32"):
        return y
    return y.to(elem_dtype)


def _affine(x, mean, rstd, g, b, fm_fast):
    y = (x - mean) * rstd
    return fmath.fma(y, g, b, fastmath=fm_fast)


def _check_dtype_n(N: int, dtype_str: str):
    if dtype_str not in ("f32", "f16", "bf16"):
        raise ValueError(f"unsupported dtype: {dtype_str!r} (expected 'f32', 'f16', or 'bf16')")
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}")


def build_layernorm_module(N: int, dtype_str: str, vec_width: int | None = None):
    """Build an Iluvatar LayerNorm launcher for fixed hidden size ``N``."""
    _check_dtype_n(N, dtype_str)

    NUM_WARPS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16
    VEC_WIDTH = int(vec_width) if vec_width is not None else _default_vec_width(elem_bits)
    if VEC_WIDTH not in (0, 2, 4, 8):
        raise ValueError(f"unsupported vec_width={VEC_WIDTH} (expected 0, 2, 4, or 8)")
    if VEC_WIDTH > 0 and elem_bits * VEC_WIDTH > 128:
        raise ValueError(f"vec_width={VEC_WIDTH} exceeds 128-bit copy for elem_bits={elem_bits}")
    NUM_TILES = _vec_num_tiles(N, VEC_WIDTH)
    SharedStorage = _make_reduction_storage(NUM_WARPS)

    @flyc.kernel
    def layernorm_kernel(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Beta: fx.Tensor,
        Output: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = _dtype_to_elem_type(dtype_str)
        fm_fast = arith.FastMathFlags.fast
        eps_c = EPS

        smem = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_sum = smem.s_sum.view(fx.make_layout(NUM_WARPS, 1))
        s_sumsq = smem.s_sumsq.view(fx.make_layout(NUM_WARPS, 1))

        def warp_reduce_add(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def block_reduce_add2(val0, val1):
            if const_expr(NUM_WARPS == 1):
                return warp_reduce_add(val0), warp_reduce_add(val1)

            lane = tid % WARP_SIZE
            warp_id = tid // WARP_SIZE

            w0 = warp_reduce_add(val0)
            w1 = warp_reduce_add(val1)

            if lane == 0:
                fx.memref_store(w0, s_sum, warp_id)
                fx.memref_store(w1, s_sumsq, warp_id)
            gpu.barrier()

            if warp_id == 0:
                in_range = lane < NUM_WARPS
                lane_safe = in_range.select(lane, 0)
                v0 = fx.memref_load(s_sum, lane_safe)
                v1 = fx.memref_load(s_sumsq, lane_safe)
                ww0 = in_range.select(v0, 0.0)
                ww1 = in_range.select(v1, 0.0)
                ww0 = warp_reduce_add(ww0)
                ww1 = warp_reduce_add(ww1)

                if lane == 0:
                    fx.memref_store(ww0, s_sum, 0)
                    fx.memref_store(ww1, s_sumsq, 0)
            gpu.barrier()

            return fx.memref_load(s_sum, 0), fx.memref_load(s_sumsq, 0)

        def compute_mean_rstd(sum_val, sumsq_val):
            inv_n = 1.0 / float(N)
            mean = sum_val * inv_n
            mean_sq = sumsq_val * inv_n
            var = mean_sq - mean * mean
            var = (var < 0.0).select(0.0, var)
            return mean, fmath.rsqrt(var + eps_c, fastmath=fm_fast)

        if const_expr(NUM_TILES > 0):
            num_tiles_py = NUM_TILES
            c_zero_f = fx.Float32(0.0)
            thread_sum = c_zero_f
            thread_sumsq = c_zero_f
            in_local = []

            row_in = fx.slice(Input, (bid, None))
            row_out = fx.slice(Output, (bid, None))

            in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(VEC_WIDTH, 1))
            gamma_div = fx.logical_divide(Gamma, fx.make_layout(VEC_WIDTH, 1))
            beta_div = fx.logical_divide(Beta, fx.make_layout(VEC_WIDTH, 1))

            copy_atom = _copy_atom_for_bits(elem_bits, VEC_WIDTH)

            for tile_i in range_constexpr(num_tiles_py):
                idx = tid + tile_i * BLOCK_THREADS
                vec = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx)
                in_local.append(vec)
                x = vec.to(fx.Float32) if const_expr(dtype_str != "f32") else vec
                x2 = x * x
                thread_sum = thread_sum + x.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sumsq = thread_sumsq + x2.reduce(ReductionOp.ADD, fastmath=fm_fast)

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean, rstd = compute_mean_rstd(sum_val, sumsq_val)

            for tile_i in range_constexpr(num_tiles_py):
                idx = tid + tile_i * BLOCK_THREADS
                x = in_local[tile_i].to(fx.Float32) if const_expr(dtype_str != "f32") else in_local[tile_i]
                g = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, gamma_div, idx)
                b = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, beta_div, idx)
                if const_expr(dtype_str != "f32"):
                    g = g.to(fx.Float32)
                    b = b.to(fx.Float32)
                y = _affine(x, mean, rstd, g, b, fm_fast)
                _store_vec(copy_atom, VEC_WIDTH, elem_dtype, _to_elem_vec(dtype_str, elem_dtype, y), out_div, idx)

        else:
            c_zero_f = fx.Float32(0.0)
            thread_sum = c_zero_f
            thread_sumsq = c_zero_f

            copy_atom_s = _copy_atom_for_bits(elem_bits, 1)

            row_in = fx.slice(Input, (bid, None))
            row_out = fx.slice(Output, (bid, None))
            row_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
            gamma_div = fx.logical_divide(Gamma, fx.make_layout(1, 1))
            beta_div = fx.logical_divide(Beta, fx.make_layout(1, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(1, 1))

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                x2 = x * x
                thread_sum = thread_sum + is_valid.select(x, c_zero_f)
                thread_sumsq = thread_sumsq + is_valid.select(x2, c_zero_f)

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean, rstd = compute_mean_rstd(sum_val, sumsq_val)

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                if idx < N:
                    x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx)
                    g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx)
                    b_e = _load_scalar(copy_atom_s, elem_dtype, beta_div, idx)
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    b = b_e if dtype_str == "f32" else b_e.to(fx.Float32)
                    y = _affine(x, mean, rstd, g, b, fm_fast)
                    _store_scalar(copy_atom_s, elem_dtype, out_div, idx, _to_elem_scalar(dtype_str, elem_dtype, y))

    @flyc.jit
    def launch_layernorm(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Beta: fx.Tensor,
        Output: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = layernorm_kernel(Input, Gamma, Beta, Output)
        launcher.launch(
            grid=(m_in, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_layernorm


def build_fused_add_layernorm_module(N: int, dtype_str: str, vec_width: int | None = None):
    """Fused residual-add + LayerNorm. Writes ResidualOut = Input + ResidualIn."""
    _check_dtype_n(N, dtype_str)

    NUM_WARPS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16
    VEC_WIDTH = int(vec_width) if vec_width is not None else _default_vec_width(elem_bits)
    if VEC_WIDTH not in (0, 2, 4, 8):
        raise ValueError(f"unsupported vec_width={VEC_WIDTH} (expected 0, 2, 4, or 8)")
    if VEC_WIDTH > 0 and elem_bits * VEC_WIDTH > 128:
        raise ValueError(f"vec_width={VEC_WIDTH} exceeds 128-bit copy for elem_bits={elem_bits}")
    NUM_TILES = _vec_num_tiles(N, VEC_WIDTH)
    SharedStorage = _make_reduction_storage(NUM_WARPS)

    @flyc.kernel
    def fused_add_layernorm_kernel(
        Input: fx.Tensor,
        ResidualIn: fx.Tensor,
        Gamma: fx.Tensor,
        Beta: fx.Tensor,
        Output: fx.Tensor,
        ResidualOut: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = _dtype_to_elem_type(dtype_str)
        fm_fast = arith.FastMathFlags.fast
        eps_c = EPS

        smem = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_sum = smem.s_sum.view(fx.make_layout(NUM_WARPS, 1))
        s_sumsq = smem.s_sumsq.view(fx.make_layout(NUM_WARPS, 1))

        def warp_reduce_add(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def block_reduce_add2(val0, val1):
            if const_expr(NUM_WARPS == 1):
                return warp_reduce_add(val0), warp_reduce_add(val1)

            lane = tid % WARP_SIZE
            warp_id = tid // WARP_SIZE
            w0 = warp_reduce_add(val0)
            w1 = warp_reduce_add(val1)

            if lane == 0:
                fx.memref_store(w0, s_sum, warp_id)
                fx.memref_store(w1, s_sumsq, warp_id)
            gpu.barrier()

            if warp_id == 0:
                in_range = lane < NUM_WARPS
                lane_safe = in_range.select(lane, 0)
                v0 = fx.memref_load(s_sum, lane_safe)
                v1 = fx.memref_load(s_sumsq, lane_safe)
                ww0 = in_range.select(v0, 0.0)
                ww1 = in_range.select(v1, 0.0)
                ww0 = warp_reduce_add(ww0)
                ww1 = warp_reduce_add(ww1)
                if lane == 0:
                    fx.memref_store(ww0, s_sum, 0)
                    fx.memref_store(ww1, s_sumsq, 0)
            gpu.barrier()

            return fx.memref_load(s_sum, 0), fx.memref_load(s_sumsq, 0)

        def compute_mean_rstd(sum_val, sumsq_val):
            inv_n = 1.0 / float(N)
            mean = sum_val * inv_n
            mean_sq = sumsq_val * inv_n
            var = mean_sq - mean * mean
            var = (var < 0.0).select(0.0, var)
            return mean, fmath.rsqrt(var + eps_c, fastmath=fm_fast)

        if const_expr(NUM_TILES > 0):
            num_tiles_py = NUM_TILES
            c_zero_f = fx.Float32(0.0)
            thread_sum = c_zero_f
            thread_sumsq = c_zero_f
            added_local = []

            row_in = fx.slice(Input, (bid, None))
            row_residual_in = fx.slice(ResidualIn, (bid, None))
            row_out = fx.slice(Output, (bid, None))
            row_residual_out = fx.slice(ResidualOut, (bid, None))

            in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))
            residual_in_div = fx.logical_divide(row_residual_in, fx.make_layout(VEC_WIDTH, 1))
            gamma_div = fx.logical_divide(Gamma, fx.make_layout(VEC_WIDTH, 1))
            beta_div = fx.logical_divide(Beta, fx.make_layout(VEC_WIDTH, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(VEC_WIDTH, 1))
            residual_out_div = fx.logical_divide(row_residual_out, fx.make_layout(VEC_WIDTH, 1))

            copy_atom = _copy_atom_for_bits(elem_bits, VEC_WIDTH)

            for tile_i in range_constexpr(num_tiles_py):
                idx = tid + tile_i * BLOCK_THREADS
                x = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx)
                residual = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, residual_in_div, idx)
                if const_expr(dtype_str != "f32"):
                    x = x.to(fx.Float32)
                    residual = residual.to(fx.Float32)
                added_e = _to_elem_vec(dtype_str, elem_dtype, x + residual)
                added_local.append(added_e)
                added = added_e if const_expr(dtype_str == "f32") else added_e.to(fx.Float32)
                added2 = added * added
                thread_sum = thread_sum + added.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sumsq = thread_sumsq + added2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                _store_vec(copy_atom, VEC_WIDTH, elem_dtype, added_e, residual_out_div, idx)

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean, rstd = compute_mean_rstd(sum_val, sumsq_val)

            for tile_i in range_constexpr(num_tiles_py):
                idx = tid + tile_i * BLOCK_THREADS
                added = added_local[tile_i] if const_expr(dtype_str == "f32") else added_local[tile_i].to(fx.Float32)
                g = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, gamma_div, idx)
                b = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, beta_div, idx)
                if const_expr(dtype_str != "f32"):
                    g = g.to(fx.Float32)
                    b = b.to(fx.Float32)
                y = _affine(added, mean, rstd, g, b, fm_fast)
                _store_vec(copy_atom, VEC_WIDTH, elem_dtype, _to_elem_vec(dtype_str, elem_dtype, y), out_div, idx)

        else:
            c_zero_f = fx.Float32(0.0)
            thread_sum = c_zero_f
            thread_sumsq = c_zero_f

            copy_atom_s = _copy_atom_for_bits(elem_bits, 1)

            row_in = fx.slice(Input, (bid, None))
            row_residual_in = fx.slice(ResidualIn, (bid, None))
            row_out = fx.slice(Output, (bid, None))
            row_residual_out = fx.slice(ResidualOut, (bid, None))

            in_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
            residual_in_div = fx.logical_divide(row_residual_in, fx.make_layout(1, 1))
            gamma_div = fx.logical_divide(Gamma, fx.make_layout(1, 1))
            beta_div = fx.logical_divide(Beta, fx.make_layout(1, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(1, 1))
            residual_out_div = fx.logical_divide(row_residual_out, fx.make_layout(1, 1))

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = _load_scalar(copy_atom_s, elem_dtype, in_div, idx_safe)
                r_e = _load_scalar(copy_atom_s, elem_dtype, residual_in_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                residual = r_e if dtype_str == "f32" else r_e.to(fx.Float32)
                added_e = _to_elem_scalar(dtype_str, elem_dtype, x + residual)
                added = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                thread_sum = thread_sum + is_valid.select(added, c_zero_f)
                thread_sumsq = thread_sumsq + is_valid.select(added * added, c_zero_f)
                if idx < N:
                    _store_scalar(copy_atom_s, elem_dtype, residual_out_div, idx, added_e)

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean, rstd = compute_mean_rstd(sum_val, sumsq_val)

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                if idx < N:
                    added_e = _load_scalar(copy_atom_s, elem_dtype, residual_out_div, idx)
                    g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx)
                    b_e = _load_scalar(copy_atom_s, elem_dtype, beta_div, idx)
                    added = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    b = b_e if dtype_str == "f32" else b_e.to(fx.Float32)
                    y = _affine(added, mean, rstd, g, b, fm_fast)
                    _store_scalar(copy_atom_s, elem_dtype, out_div, idx, _to_elem_scalar(dtype_str, elem_dtype, y))

    @flyc.jit
    def launch_fused_add_layernorm(
        Input: fx.Tensor,
        ResidualIn: fx.Tensor,
        Gamma: fx.Tensor,
        Beta: fx.Tensor,
        Output: fx.Tensor,
        ResidualOut: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = fused_add_layernorm_kernel(Input, ResidualIn, Gamma, Beta, Output, ResidualOut)
        launcher.launch(
            grid=(m_in, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_fused_add_layernorm


def _build_layernorm_quant_module(
    N: int,
    dtype_str: str,
    *,
    is_smooth: bool,
    quant_dtype_str: str = "i8",
    vec_width: int | None = None,
):
    _check_dtype_n(N, dtype_str)
    if quant_dtype_str not in ("i8", "int8"):
        raise ValueError(f"unsupported quant dtype: {quant_dtype_str!r}")

    NUM_WARPS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16
    VEC_WIDTH = int(vec_width) if vec_width is not None else _default_vec_width(elem_bits)
    if VEC_WIDTH not in (0, 2, 4, 8):
        raise ValueError(f"unsupported vec_width={VEC_WIDTH} (expected 0, 2, 4, or 8)")
    if VEC_WIDTH > 0 and elem_bits * VEC_WIDTH > 128:
        raise ValueError(f"vec_width={VEC_WIDTH} exceeds 128-bit copy for elem_bits={elem_bits}")
    # i8 packing stores VEC_WIDTH/2 elems per transaction when using the vector quant path.
    if VEC_WIDTH > 0 and VEC_WIDTH % 2 != 0:
        raise ValueError(f"quant path requires even vec_width, got {VEC_WIDTH}")
    NUM_TILES = _vec_num_tiles(N, VEC_WIDTH)
    quant_dtype_max = _quant_dtype_max(quant_dtype_str)
    SharedStorage = _make_reduction_storage(NUM_WARPS)

    @flyc.kernel
    def layernorm_quant_kernel(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Beta: fx.Tensor,
        XScale: fx.Tensor,
        YScale: fx.Tensor,
        Output: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = _dtype_to_elem_type(dtype_str)
        quant_dtype = _quant_dtype_to_elem_type(quant_dtype_str)

        fm_fast = arith.FastMathFlags.fast
        eps_c = EPS
        n_float = float(N)
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_neg_inf = fx.Float32(float("-inf"))
        c_dtype_max = fx.Float32(quant_dtype_max)

        smem = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_sum = smem.s_sum.view(fx.make_layout(NUM_WARPS, 1))
        s_sumsq = smem.s_sumsq.view(fx.make_layout(NUM_WARPS, 1))

        yscale_div = fx.logical_divide(YScale, fx.make_layout(1, 1))
        scale_copy_atom = _copy_atom_for_bits(32, 1)

        def warp_reduce_add(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def warp_reduce_max(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.maximumf(peer)
            return w

        def block_reduce_add2(val0, val1):
            if const_expr(NUM_WARPS == 1):
                return warp_reduce_add(val0), warp_reduce_add(val1)

            lane = tid % WARP_SIZE
            warp_id = tid // WARP_SIZE
            w0 = warp_reduce_add(val0)
            w1 = warp_reduce_add(val1)

            if lane == 0:
                fx.memref_store(w0, s_sum, warp_id)
                fx.memref_store(w1, s_sumsq, warp_id)
            gpu.barrier()

            if warp_id == 0:
                in_range = lane < NUM_WARPS
                lane_safe = in_range.select(lane, 0)
                v0 = fx.memref_load(s_sum, lane_safe)
                v1 = fx.memref_load(s_sumsq, lane_safe)
                ww0 = in_range.select(v0, c_zero_f)
                ww1 = in_range.select(v1, c_zero_f)
                ww0 = warp_reduce_add(ww0)
                ww1 = warp_reduce_add(ww1)
                if lane == 0:
                    fx.memref_store(ww0, s_sum, 0)
                    fx.memref_store(ww1, s_sumsq, 0)
            gpu.barrier()

            return fx.memref_load(s_sum, 0), fx.memref_load(s_sumsq, 0)

        def block_reduce_max(val):
            if const_expr(NUM_WARPS == 1):
                return warp_reduce_max(val)

            lane = tid % WARP_SIZE
            warp_id = tid // WARP_SIZE
            w = warp_reduce_max(val)
            if lane == 0:
                fx.memref_store(w, s_sum, warp_id)
            gpu.barrier()

            if warp_id == 0:
                in_range = lane < NUM_WARPS
                lane_safe = in_range.select(lane, 0)
                v = fx.memref_load(s_sum, lane_safe)
                ww = in_range.select(v, c_neg_inf)
                ww = warp_reduce_max(ww)
                if lane == 0:
                    fx.memref_store(ww, s_sum, 0)
            gpu.barrier()

            return fx.memref_load(s_sum, 0)

        if const_expr(NUM_TILES > 0 and elem_bits <= 16):
            num_tiles_py = NUM_TILES
            quant_half_width = VEC_WIDTH // 2
            abs_mask = full(VEC_WIDTH, fx.Uint32(0x7FFFFFFF), fx.Uint32)

            row_in = fx.slice(Input, (bid, None))
            row_out = fx.slice(Output, (bid, None))

            in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))
            gamma_div = fx.logical_divide(Gamma, fx.make_layout(VEC_WIDTH, 1))
            beta_div = fx.logical_divide(Beta, fx.make_layout(VEC_WIDTH, 1))
            out_div_q = fx.logical_divide(row_out, fx.make_layout(quant_half_width, 1))
            if const_expr(is_smooth):
                xscale_div = fx.logical_divide(XScale, fx.make_layout(VEC_WIDTH, 1))

            copy_atom = _copy_atom_for_bits(elem_bits, VEC_WIDTH)
            copy_atom_q = _copy_atom_for_bits(8, quant_half_width)
            if const_expr(is_smooth):
                copy_atom_xs = _copy_atom_for_bits(elem_bits, VEC_WIDTH)

            thread_sum = c_zero_f
            thread_sumsq = c_zero_f
            norm_input_local = []

            for tile_i in range_constexpr(num_tiles_py):
                idx = tid + tile_i * BLOCK_THREADS
                x_e = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx)
                norm_input_local.append(x_e)
                x_norm = x_e.to(fx.Float32)
                x2 = x_norm * x_norm
                thread_sum = thread_sum + x_norm.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sumsq = thread_sumsq + x2.reduce(ReductionOp.ADD, fastmath=fm_fast)

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean = sum_val / n_float
            var = sumsq_val / n_float - mean * mean
            var = (var < c_zero_f).select(c_zero_f, var)
            rstd = fmath.rsqrt(var + eps_c, fastmath=fm_fast)

            thread_row_max = c_zero_f
            y_local = []

            for tile_i in range_constexpr(num_tiles_py):
                idx = tid + tile_i * BLOCK_THREADS
                x = norm_input_local[tile_i].to(fx.Float32)
                g = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, gamma_div, idx).to(fx.Float32)
                b = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, beta_div, idx).to(fx.Float32)
                y = _affine(x, mean, rstd, g, b, fm_fast)
                if const_expr(is_smooth):
                    s = _load_vec(copy_atom_xs, VEC_WIDTH, elem_dtype, xscale_div, idx).to(fx.Float32)
                    y = y * s
                y_local.append(y)
                y_abs = (y.bitcast(fx.Uint32) & abs_mask).bitcast(fx.Float32)
                thread_row_max = thread_row_max.maximumf(y_abs.reduce(ReductionOp.MAX))

            row_max = block_reduce_max(thread_row_max)
            scale = row_max / c_dtype_max
            final_scale = (scale == c_zero_f).select(c_one_f, scale)

            if tid == 0:
                _store_yscale(scale_copy_atom, yscale_div, bid, final_scale)

            inv_scale = c_one_f / final_scale

            for tile_i in range_constexpr(num_tiles_py):
                q = y_local[tile_i] * inv_scale
                q_i8 = q.to(quant_dtype)
                if const_expr(VEC_WIDTH == 8):
                    q_lo = q_i8.shuffle(q_i8, [0, 1, 2, 3])
                    q_hi = q_i8.shuffle(q_i8, [4, 5, 6, 7])
                    out_idx = tid * 2 + tile_i * BLOCK_THREADS * 2
                    _store_vec(copy_atom_q, quant_half_width, quant_dtype, q_lo, out_div_q, out_idx)
                    _store_vec(copy_atom_q, quant_half_width, quant_dtype, q_hi, out_div_q, out_idx + 1)
                else:
                    q_lo = q_i8.shuffle(q_i8, [0, 1])
                    q_hi = q_i8.shuffle(q_i8, [2, 3])
                    out_idx = tid * 2 + tile_i * BLOCK_THREADS * 2
                    _store_vec(copy_atom_q, quant_half_width, quant_dtype, q_lo, out_div_q, out_idx)
                    _store_vec(copy_atom_q, quant_half_width, quant_dtype, q_hi, out_div_q, out_idx + 1)

        else:
            row_in = fx.slice(Input, (bid, None))
            row_out = fx.slice(Output, (bid, None))

            copy_atom_s = _copy_atom_for_bits(elem_bits, 1)
            copy_atom_qs = _copy_atom_for_bits(8, 1)

            in_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
            gamma_div = fx.logical_divide(Gamma, fx.make_layout(1, 1))
            beta_div = fx.logical_divide(Beta, fx.make_layout(1, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(1, 1))
            if const_expr(is_smooth):
                xscale_div = fx.logical_divide(XScale, fx.make_layout(1, 1))

            def _abs_scalar(val):
                is_neg = val < c_zero_f
                return is_neg.select(c_zero_f - val, val)

            thread_sum = c_zero_f
            thread_sumsq = c_zero_f

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = _load_scalar(copy_atom_s, elem_dtype, in_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                x2 = x * x
                thread_sum = thread_sum + is_valid.select(x, c_zero_f)
                thread_sumsq = thread_sumsq + is_valid.select(x2, c_zero_f)

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean = sum_val / n_float
            var = sumsq_val / n_float - mean * mean
            var = (var < c_zero_f).select(c_zero_f, var)
            rstd = fmath.rsqrt(var + eps_c, fastmath=fm_fast)

            thread_row_max = c_zero_f
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = _load_scalar(copy_atom_s, elem_dtype, in_div, idx_safe)
                g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx_safe)
                b_e = _load_scalar(copy_atom_s, elem_dtype, beta_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                b = b_e if dtype_str == "f32" else b_e.to(fx.Float32)
                y = _affine(x, mean, rstd, g, b, fm_fast)
                if const_expr(is_smooth):
                    s_e = _load_scalar(copy_atom_s, elem_dtype, xscale_div, idx_safe)
                    s = s_e if dtype_str == "f32" else s_e.to(fx.Float32)
                    y = y * s
                thread_row_max = thread_row_max.maximumf(is_valid.select(_abs_scalar(y), c_zero_f))

            row_max = block_reduce_max(thread_row_max)
            scale = row_max / c_dtype_max
            final_scale = (scale == c_zero_f).select(c_one_f, scale)

            if tid == 0:
                _store_yscale(scale_copy_atom, yscale_div, bid, final_scale)

            inv_scale = c_one_f / final_scale

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                if idx < N:
                    x_e = _load_scalar(copy_atom_s, elem_dtype, in_div, idx)
                    g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx)
                    b_e = _load_scalar(copy_atom_s, elem_dtype, beta_div, idx)
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    b = b_e if dtype_str == "f32" else b_e.to(fx.Float32)
                    y = _affine(x, mean, rstd, g, b, fm_fast)
                    if const_expr(is_smooth):
                        s_e = _load_scalar(copy_atom_s, elem_dtype, xscale_div, idx)
                        s = s_e if dtype_str == "f32" else s_e.to(fx.Float32)
                        y = y * s
                    q_i8 = (y * inv_scale).to(quant_dtype)
                    _store_scalar(copy_atom_qs, quant_dtype, out_div, idx, q_i8)

    if is_smooth:

        @flyc.jit
        def launch_layernorm_smoothquant(
            Input: fx.Tensor,
            Gamma: fx.Tensor,
            Beta: fx.Tensor,
            XScale: fx.Tensor,
            Output: fx.Tensor,
            YScale: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            launcher = layernorm_quant_kernel(Input, Gamma, Beta, XScale, YScale, Output)
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

        return launch_layernorm_smoothquant

    @flyc.jit
    def launch_layernorm_dynamicquant(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Beta: fx.Tensor,
        Output: fx.Tensor,
        YScale: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = layernorm_quant_kernel(Input, Gamma, Beta, Gamma, YScale, Output)
        launcher.launch(
            grid=(m_in, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_layernorm_dynamicquant


def _build_fused_add_layernorm_quant_module(
    N: int,
    dtype_str: str,
    *,
    is_smooth: bool,
    quant_dtype_str: str = "i8",
    vec_width: int | None = None,
):
    _check_dtype_n(N, dtype_str)
    if quant_dtype_str not in ("i8", "int8"):
        raise ValueError(f"unsupported quant dtype: {quant_dtype_str!r}")

    NUM_WARPS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16
    VEC_WIDTH = int(vec_width) if vec_width is not None else _default_vec_width(elem_bits)
    if VEC_WIDTH not in (0, 2, 4, 8):
        raise ValueError(f"unsupported vec_width={VEC_WIDTH} (expected 0, 2, 4, or 8)")
    if VEC_WIDTH > 0 and elem_bits * VEC_WIDTH > 128:
        raise ValueError(f"vec_width={VEC_WIDTH} exceeds 128-bit copy for elem_bits={elem_bits}")
    if VEC_WIDTH > 0 and VEC_WIDTH % 2 != 0:
        raise ValueError(f"quant path requires even vec_width, got {VEC_WIDTH}")
    NUM_TILES = _vec_num_tiles(N, VEC_WIDTH)
    quant_dtype_max = _quant_dtype_max(quant_dtype_str)
    SharedStorage = _make_reduction_storage(NUM_WARPS)

    @flyc.kernel
    def fused_add_layernorm_quant_kernel(
        Input: fx.Tensor,
        ResidualIn: fx.Tensor,
        Gamma: fx.Tensor,
        Beta: fx.Tensor,
        XScale: fx.Tensor,
        YScale: fx.Tensor,
        Output: fx.Tensor,
        ResidualOut: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = _dtype_to_elem_type(dtype_str)
        quant_dtype = _quant_dtype_to_elem_type(quant_dtype_str)

        fm_fast = arith.FastMathFlags.fast
        eps_c = EPS
        n_float = float(N)
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_neg_inf = fx.Float32(float("-inf"))
        c_dtype_max = fx.Float32(quant_dtype_max)

        smem = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_sum = smem.s_sum.view(fx.make_layout(NUM_WARPS, 1))
        s_sumsq = smem.s_sumsq.view(fx.make_layout(NUM_WARPS, 1))

        yscale_div = fx.logical_divide(YScale, fx.make_layout(1, 1))
        scale_copy_atom = _copy_atom_for_bits(32, 1)

        def warp_reduce_add(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def warp_reduce_max(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.maximumf(peer)
            return w

        def block_reduce_add2(val0, val1):
            if const_expr(NUM_WARPS == 1):
                return warp_reduce_add(val0), warp_reduce_add(val1)

            lane = tid % WARP_SIZE
            warp_id = tid // WARP_SIZE
            w0 = warp_reduce_add(val0)
            w1 = warp_reduce_add(val1)

            if lane == 0:
                fx.memref_store(w0, s_sum, warp_id)
                fx.memref_store(w1, s_sumsq, warp_id)
            gpu.barrier()

            if warp_id == 0:
                in_range = lane < NUM_WARPS
                lane_safe = in_range.select(lane, 0)
                v0 = fx.memref_load(s_sum, lane_safe)
                v1 = fx.memref_load(s_sumsq, lane_safe)
                ww0 = in_range.select(v0, c_zero_f)
                ww1 = in_range.select(v1, c_zero_f)
                ww0 = warp_reduce_add(ww0)
                ww1 = warp_reduce_add(ww1)
                if lane == 0:
                    fx.memref_store(ww0, s_sum, 0)
                    fx.memref_store(ww1, s_sumsq, 0)
            gpu.barrier()

            return fx.memref_load(s_sum, 0), fx.memref_load(s_sumsq, 0)

        def block_reduce_max(val):
            if const_expr(NUM_WARPS == 1):
                return warp_reduce_max(val)

            lane = tid % WARP_SIZE
            warp_id = tid // WARP_SIZE
            w = warp_reduce_max(val)
            if lane == 0:
                fx.memref_store(w, s_sum, warp_id)
            gpu.barrier()

            if warp_id == 0:
                in_range = lane < NUM_WARPS
                lane_safe = in_range.select(lane, 0)
                v = fx.memref_load(s_sum, lane_safe)
                ww = in_range.select(v, c_neg_inf)
                ww = warp_reduce_max(ww)
                if lane == 0:
                    fx.memref_store(ww, s_sum, 0)
            gpu.barrier()

            return fx.memref_load(s_sum, 0)

        if const_expr(NUM_TILES > 0 and elem_bits <= 16):
            num_tiles_py = NUM_TILES
            quant_half_width = VEC_WIDTH // 2
            abs_mask = full(VEC_WIDTH, fx.Uint32(0x7FFFFFFF), fx.Uint32)

            row_in = fx.slice(Input, (bid, None))
            row_residual_in = fx.slice(ResidualIn, (bid, None))
            row_out = fx.slice(Output, (bid, None))
            row_residual_out = fx.slice(ResidualOut, (bid, None))

            in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))
            residual_in_div = fx.logical_divide(row_residual_in, fx.make_layout(VEC_WIDTH, 1))
            gamma_div = fx.logical_divide(Gamma, fx.make_layout(VEC_WIDTH, 1))
            beta_div = fx.logical_divide(Beta, fx.make_layout(VEC_WIDTH, 1))
            out_div_q = fx.logical_divide(row_out, fx.make_layout(quant_half_width, 1))
            residual_out_div = fx.logical_divide(row_residual_out, fx.make_layout(VEC_WIDTH, 1))
            if const_expr(is_smooth):
                xscale_div = fx.logical_divide(XScale, fx.make_layout(VEC_WIDTH, 1))

            copy_atom = _copy_atom_for_bits(elem_bits, VEC_WIDTH)
            copy_atom_q = _copy_atom_for_bits(8, quant_half_width)
            if const_expr(is_smooth):
                copy_atom_xs = _copy_atom_for_bits(elem_bits, VEC_WIDTH)

            thread_sum = c_zero_f
            thread_sumsq = c_zero_f
            norm_input_local = []

            for tile_i in range_constexpr(num_tiles_py):
                idx = tid + tile_i * BLOCK_THREADS
                x = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx).to(fx.Float32)
                residual = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, residual_in_div, idx).to(fx.Float32)
                added_e = _to_elem_vec(dtype_str, elem_dtype, x + residual)
                norm_input_local.append(added_e)
                x_norm = added_e.to(fx.Float32)
                _store_vec(copy_atom, VEC_WIDTH, elem_dtype, added_e, residual_out_div, idx)
                x2 = x_norm * x_norm
                thread_sum = thread_sum + x_norm.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sumsq = thread_sumsq + x2.reduce(ReductionOp.ADD, fastmath=fm_fast)

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean = sum_val / n_float
            var = sumsq_val / n_float - mean * mean
            var = (var < c_zero_f).select(c_zero_f, var)
            rstd = fmath.rsqrt(var + eps_c, fastmath=fm_fast)

            thread_row_max = c_zero_f
            y_local = []

            for tile_i in range_constexpr(num_tiles_py):
                idx = tid + tile_i * BLOCK_THREADS
                x = norm_input_local[tile_i].to(fx.Float32)
                g = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, gamma_div, idx).to(fx.Float32)
                b = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, beta_div, idx).to(fx.Float32)
                y = _affine(x, mean, rstd, g, b, fm_fast)
                if const_expr(is_smooth):
                    s = _load_vec(copy_atom_xs, VEC_WIDTH, elem_dtype, xscale_div, idx).to(fx.Float32)
                    y = y * s
                y_local.append(y)
                y_abs = (y.bitcast(fx.Uint32) & abs_mask).bitcast(fx.Float32)
                thread_row_max = thread_row_max.maximumf(y_abs.reduce(ReductionOp.MAX))

            row_max = block_reduce_max(thread_row_max)
            scale = row_max / c_dtype_max
            final_scale = (scale == c_zero_f).select(c_one_f, scale)

            if tid == 0:
                _store_yscale(scale_copy_atom, yscale_div, bid, final_scale)

            inv_scale = c_one_f / final_scale

            for tile_i in range_constexpr(num_tiles_py):
                q = y_local[tile_i] * inv_scale
                q_i8 = q.to(quant_dtype)
                if const_expr(VEC_WIDTH == 8):
                    q_lo = q_i8.shuffle(q_i8, [0, 1, 2, 3])
                    q_hi = q_i8.shuffle(q_i8, [4, 5, 6, 7])
                else:
                    q_lo = q_i8.shuffle(q_i8, [0, 1])
                    q_hi = q_i8.shuffle(q_i8, [2, 3])
                out_idx = tid * 2 + tile_i * BLOCK_THREADS * 2
                _store_vec(copy_atom_q, quant_half_width, quant_dtype, q_lo, out_div_q, out_idx)
                _store_vec(copy_atom_q, quant_half_width, quant_dtype, q_hi, out_div_q, out_idx + 1)

        else:
            row_in = fx.slice(Input, (bid, None))
            row_residual_in = fx.slice(ResidualIn, (bid, None))
            row_out = fx.slice(Output, (bid, None))
            row_residual_out = fx.slice(ResidualOut, (bid, None))

            copy_atom_s = _copy_atom_for_bits(elem_bits, 1)
            copy_atom_qs = _copy_atom_for_bits(8, 1)

            in_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
            residual_in_div = fx.logical_divide(row_residual_in, fx.make_layout(1, 1))
            gamma_div = fx.logical_divide(Gamma, fx.make_layout(1, 1))
            beta_div = fx.logical_divide(Beta, fx.make_layout(1, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(1, 1))
            residual_out_div = fx.logical_divide(row_residual_out, fx.make_layout(1, 1))
            if const_expr(is_smooth):
                xscale_div = fx.logical_divide(XScale, fx.make_layout(1, 1))

            def _abs_scalar(val):
                is_neg = val < c_zero_f
                return is_neg.select(c_zero_f - val, val)

            thread_sum = c_zero_f
            thread_sumsq = c_zero_f

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = _load_scalar(copy_atom_s, elem_dtype, in_div, idx_safe)
                r_e = _load_scalar(copy_atom_s, elem_dtype, residual_in_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                residual = r_e if dtype_str == "f32" else r_e.to(fx.Float32)
                added_e = _to_elem_scalar(dtype_str, elem_dtype, x + residual)
                if idx < N:
                    _store_scalar(copy_atom_s, elem_dtype, residual_out_div, idx, added_e)
                x = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                x2 = x * x
                thread_sum = thread_sum + is_valid.select(x, c_zero_f)
                thread_sumsq = thread_sumsq + is_valid.select(x2, c_zero_f)

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean = sum_val / n_float
            var = sumsq_val / n_float - mean * mean
            var = (var < c_zero_f).select(c_zero_f, var)
            rstd = fmath.rsqrt(var + eps_c, fastmath=fm_fast)

            thread_row_max = c_zero_f
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = _load_scalar(copy_atom_s, elem_dtype, residual_out_div, idx_safe)
                g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx_safe)
                b_e = _load_scalar(copy_atom_s, elem_dtype, beta_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                b = b_e if dtype_str == "f32" else b_e.to(fx.Float32)
                y = _affine(x, mean, rstd, g, b, fm_fast)
                if const_expr(is_smooth):
                    s_e = _load_scalar(copy_atom_s, elem_dtype, xscale_div, idx_safe)
                    s = s_e if dtype_str == "f32" else s_e.to(fx.Float32)
                    y = y * s
                thread_row_max = thread_row_max.maximumf(is_valid.select(_abs_scalar(y), c_zero_f))

            row_max = block_reduce_max(thread_row_max)
            scale = row_max / c_dtype_max
            final_scale = (scale == c_zero_f).select(c_one_f, scale)

            if tid == 0:
                _store_yscale(scale_copy_atom, yscale_div, bid, final_scale)

            inv_scale = c_one_f / final_scale

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                if idx < N:
                    x_e = _load_scalar(copy_atom_s, elem_dtype, residual_out_div, idx)
                    g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx)
                    b_e = _load_scalar(copy_atom_s, elem_dtype, beta_div, idx)
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    b = b_e if dtype_str == "f32" else b_e.to(fx.Float32)
                    y = _affine(x, mean, rstd, g, b, fm_fast)
                    if const_expr(is_smooth):
                        s_e = _load_scalar(copy_atom_s, elem_dtype, xscale_div, idx)
                        s = s_e if dtype_str == "f32" else s_e.to(fx.Float32)
                        y = y * s
                    q_i8 = (y * inv_scale).to(quant_dtype)
                    _store_scalar(copy_atom_qs, quant_dtype, out_div, idx, q_i8)

    if is_smooth:

        @flyc.jit
        def launch_fused_add_layernorm_smoothquant(
            Input: fx.Tensor,
            ResidualIn: fx.Tensor,
            Gamma: fx.Tensor,
            Beta: fx.Tensor,
            XScale: fx.Tensor,
            Output: fx.Tensor,
            ResidualOut: fx.Tensor,
            YScale: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            launcher = fused_add_layernorm_quant_kernel(
                Input, ResidualIn, Gamma, Beta, XScale, YScale, Output, ResidualOut
            )
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

        return launch_fused_add_layernorm_smoothquant

    @flyc.jit
    def launch_fused_add_layernorm_dynamicquant(
        Input: fx.Tensor,
        ResidualIn: fx.Tensor,
        Gamma: fx.Tensor,
        Beta: fx.Tensor,
        Output: fx.Tensor,
        ResidualOut: fx.Tensor,
        YScale: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = fused_add_layernorm_quant_kernel(Input, ResidualIn, Gamma, Beta, Gamma, YScale, Output, ResidualOut)
        launcher.launch(
            grid=(m_in, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_fused_add_layernorm_dynamicquant


def build_layernorm_dynamicquant_module(
    N: int, dtype_str: str, quant_dtype_str: str = "i8", vec_width: int | None = None
):
    return _build_layernorm_quant_module(
        N, dtype_str, is_smooth=False, quant_dtype_str=quant_dtype_str, vec_width=vec_width
    )


def build_layernorm_smoothquant_module(
    N: int, dtype_str: str, quant_dtype_str: str = "i8", vec_width: int | None = None
):
    return _build_layernorm_quant_module(
        N, dtype_str, is_smooth=True, quant_dtype_str=quant_dtype_str, vec_width=vec_width
    )


def build_fused_add_layernorm_dynamicquant_module(
    N: int, dtype_str: str, quant_dtype_str: str = "i8", vec_width: int | None = None
):
    return _build_fused_add_layernorm_quant_module(
        N, dtype_str, is_smooth=False, quant_dtype_str=quant_dtype_str, vec_width=vec_width
    )


def build_fused_add_layernorm_smoothquant_module(
    N: int, dtype_str: str, quant_dtype_str: str = "i8", vec_width: int | None = None
):
    return _build_fused_add_layernorm_quant_module(
        N, dtype_str, is_smooth=True, quant_dtype_str=quant_dtype_str, vec_width=vec_width
    )
