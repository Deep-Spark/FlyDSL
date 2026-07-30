# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar row-wise softmax forward kernel.

``out[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))`` along the last dim.

Design choices for V1:
- Iluvatar-only; one CTA per row (``grid.x = M``), single-warp block
  (``BLOCK_THREADS = 64``).
- Multi-width vectorized fast paths (``UniversalCopy{128,64,32}b``) plus
  scalar fallback for arbitrary ``N``.
- Supported dtypes: f16 / bf16 / f32. Accumulation is always f32.
- Exponentiation: Schraudolph-style bitcast ``exp2`` approximation.
"""

import math
from collections.abc import Callable

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, range_constexpr, vector
from flydsl.expr.vector import ReductionOp
from kernels.gemm.iluvatar.common import WARP_SIZE

BLOCK_THREADS = 64
SUPPORTED_DTYPES = ("f16", "bf16", "f32")
TORCH_F16_NAME = "torch.float16"
TORCH_BF16_NAME = "torch.bfloat16"
TORCH_F32_NAME = "torch.float32"


def _dtype_to_elem_type(dtype_str: str):
    if dtype_str == "f16":
        return fx.Float16
    if dtype_str == "bf16":
        return fx.BFloat16
    if dtype_str == "f32":
        return fx.Float32
    raise ValueError(f"Iluvatar softmax supports only {SUPPORTED_DTYPES}, got {dtype_str!r}")


def _dtype_name(tensor) -> str:
    return str(tensor.dtype)


def _torch_dtype_name(dtype_str: str) -> str:
    if dtype_str == "f16":
        return TORCH_F16_NAME
    if dtype_str == "bf16":
        return TORCH_BF16_NAME
    if dtype_str == "f32":
        return TORCH_F32_NAME
    raise ValueError(f"Iluvatar softmax supports only {SUPPORTED_DTYPES}, got {dtype_str!r}")


def _byte_range(tensor) -> tuple[int, int]:
    start = int(tensor.data_ptr())
    end = start + int(tensor.numel()) * int(tensor.element_size())
    return start, end


def _tensors_overlap(a, b) -> bool:
    a0, a1 = _byte_range(a)
    b0, b1 = _byte_range(b)
    return max(a0, b0) < min(a1, b1)


def _build_softmax_kernel(*, N: int, dtype_str: str):
    if BLOCK_THREADS != WARP_SIZE:
        raise ValueError(
            f"internal error: V1 softmax expects single-warp block, got "
            f"BLOCK_THREADS={BLOCK_THREADS} WARP_SIZE={WARP_SIZE}"
        )

    elem_bits = 32 if dtype_str == "f32" else 16
    vec128_width = 128 // elem_bits
    vec64_width = 64 // elem_bits
    vec32_width = 32 // elem_bits

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def _softmax_kernel(x: fx.Tensor, out: fx.Tensor):
        bid = fx.block_idx.x
        lane_id = fx.Int32(fx.lane_id)

        elem_dtype = _dtype_to_elem_type(dtype_str)
        fm_fast = arith.FastMathFlags.fast

        c_zero_f = fx.Float32(0.0)
        c_neg_inf = fx.Float32(float("-inf"))
        c_log2e = fx.Float32(1.4426950408889634)
        c_min_exp2 = fx.Float32(-126.0)
        c_max_exp2 = fx.Float32(126.0)
        c_exp2_scale = fx.Float32(8388608.0)  # 2^23
        c_exp2_bias = fx.Float32(1065353216.0)  # 127 << 23

        def _warp_reduce(val, mode):
            w = val
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                if const_expr(mode == "max"):
                    w = w.maximumf(peer)
                else:
                    w = w.addf(peer, fastmath=fm_fast)
            return w

        def exp2_approx_f32(v):
            # Schraudolph-style exp2 approximation:
            #   exp2(x) ~= bitcast_i32_to_f32(int(x * 2^23 + (127 << 23)))
            # Clamp x so exponent bits stay in a normal representable range.
            x_hi = (v > c_max_exp2).select(c_max_exp2, v)
            x_clamped = (x_hi < c_min_exp2).select(c_min_exp2, x_hi)
            y = x_clamped * c_exp2_scale + c_exp2_bias
            yi = fx.Int32(y)
            return yi.bitcast(fx.Float32)

        def _run_vectorized(vec_width: int, copy_atom):
            tile_cols = BLOCK_THREADS * vec_width
            num_tiles = N // tile_cols
            row_buffer = []
            thread_max = c_neg_inf

            row_x = fx.slice(x, (bid, None))
            row_out = fx.slice(out, (bid, None))
            x_div = fx.logical_divide(row_x, fx.make_layout(vec_width, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(vec_width, 1))

            def _load_vec(div_tensor, index):
                reg = fx.make_rmem_tensor(vec_width, elem_dtype)
                fx.copy_atom_call(copy_atom, fx.slice(div_tensor, (None, index)), reg)
                return fx.memref_load_vec(reg)

            def _store_vec(val, div_tensor, index):
                reg = fx.make_rmem_tensor(vec_width, elem_dtype)
                fx.memref_store_vec(val, reg)
                fx.copy_atom_call(copy_atom, reg, fx.slice(div_tensor, (None, index)))

            # 1) Load + local max
            for tile_i in range_constexpr(num_tiles):
                idx = lane_id + tile_i * BLOCK_THREADS
                vec_e = _load_vec(x_div, idx)
                xv = vec_e.to(fx.Float32)
                row_buffer.append(xv)
                red_max = xv.reduce(ReductionOp.MAX)
                thread_max = thread_max.maximumf(red_max)

            global_max = _warp_reduce(thread_max, "max")

            # 2) Exp + local sum
            thread_sum = c_zero_f
            for i in range_constexpr(num_tiles):
                xv = row_buffer[i]
                scaled = (xv - global_max) * c_log2e
                exp_elems = []
                for vi in range_constexpr(vec_width):
                    s_i = vector.extract(scaled, static_position=[vi], dynamic_position=[])
                    exp_elems.append(exp2_approx_f32(s_i))
                exp_val = fx.Vector.from_elements(exp_elems, dtype=fx.Float32)
                row_buffer[i] = exp_val
                red_sum = exp_val.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sum = thread_sum + red_sum

            global_sum = _warp_reduce(thread_sum, "sum")
            inv_sum = 1.0 / global_sum

            # 3) Normalize + store
            for tile_i in range_constexpr(num_tiles):
                norm_vec = row_buffer[tile_i] * inv_sum
                out_e = norm_vec.to(elem_dtype)
                out_idx = lane_id + tile_i * BLOCK_THREADS
                _store_vec(out_e, out_div, out_idx)

        # Fast paths: aligned vectorized copy via UniversalCopy{128,64,32}b.
        if const_expr(N >= BLOCK_THREADS * vec128_width and N % (BLOCK_THREADS * vec128_width) == 0):
            _run_vectorized(vec128_width, fx.make_copy_atom(fx.UniversalCopy128b(), elem_dtype))
        elif const_expr(N >= BLOCK_THREADS * vec64_width and N % (BLOCK_THREADS * vec64_width) == 0):
            _run_vectorized(vec64_width, fx.make_copy_atom(fx.UniversalCopy64b(), elem_dtype))
        elif const_expr(N >= BLOCK_THREADS * vec32_width and N % (BLOCK_THREADS * vec32_width) == 0):
            _run_vectorized(vec32_width, fx.make_copy_atom(fx.UniversalCopy32b(), elem_dtype))
        else:
            # Generic path: scalar for arbitrary N.
            row_buffer = []
            thread_max = c_neg_inf
            for base in range_constexpr(0, N, BLOCK_THREADS):
                idx = lane_id + base
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                val_e = x[bid, idx_safe]
                val = val_e.to(fx.Float32)
                safe_val = is_valid.select(val, c_neg_inf)
                row_buffer.append((safe_val, is_valid))
                thread_max = thread_max.maximumf(safe_val)

            global_max = _warp_reduce(thread_max, "max")

            thread_sum = c_zero_f
            exp_buffer = []
            for safe_val, is_valid in row_buffer:
                scaled = (safe_val - global_max) * c_log2e
                exp_val = exp2_approx_f32(scaled)
                safe_exp = is_valid.select(exp_val, c_zero_f)
                thread_sum = thread_sum + safe_exp
                exp_buffer.append((exp_val, is_valid))

            global_sum = _warp_reduce(thread_sum, "sum")
            inv_sum = 1.0 / global_sum

            buf_idx = 0
            for base in range_constexpr(0, N, BLOCK_THREADS):
                idx = lane_id + base
                exp_val, _is_valid = exp_buffer[buf_idx]
                buf_idx += 1
                if idx < N:
                    norm_val = fx.Float32(exp_val) * inv_sum
                    out[bid, idx] = norm_val.to(elem_dtype)

    return _softmax_kernel


def compile_iluvatar_softmax(*, N: int, dtype: str = "f16") -> Callable:
    """Build Iluvatar row-wise softmax launcher.

    Args:
        N: Hidden size. Compile-time constant and must be ``> 0``.
        dtype: ``"f16"``, ``"bf16"``, or ``"f32"``.

    Returns:
        ``launch_softmax(x, out, M, stream=None) -> out``.
    """
    if N <= 0:
        raise ValueError(f"N must be > 0, got {N}")
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"dtype must be one of {SUPPORTED_DTYPES}, got {dtype!r}")

    kernel = _build_softmax_kernel(N=N, dtype_str=dtype)
    expected_torch_dtype = _torch_dtype_name(dtype)

    @flyc.jit
    def _launch_kernel(
        x: fx.Tensor,
        out: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        kernel(x, out).launch(
            grid=(m_in, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    def launch_softmax(x, out, M: int, stream=None):
        if not isinstance(M, int):
            raise ValueError(f"M must be int, got {type(M).__name__}")
        if M < 0:
            raise ValueError(f"M must be >= 0, got {M}")

        if x.dim() != 2:
            raise ValueError(f"expected x shape (M,N), got dim={x.dim()} shape={tuple(x.shape)}")
        if out.dim() != 2:
            raise ValueError(f"expected out shape (M,N), got dim={out.dim()} shape={tuple(out.shape)}")

        if tuple(x.shape) != (M, N):
            raise ValueError(f"expected x shape (M,N)=({M},{N}), got {tuple(x.shape)}")
        if tuple(out.shape) != (M, N):
            raise ValueError(f"expected out shape (M,N)=({M},{N}), got {tuple(out.shape)}")

        if not x.is_contiguous():
            raise ValueError("x must be contiguous")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")

        x_dtype = _dtype_name(x)
        out_dtype = _dtype_name(out)
        if x_dtype != expected_torch_dtype:
            raise ValueError(f"x dtype must be {expected_torch_dtype}, got {x_dtype}")
        if out_dtype != expected_torch_dtype:
            raise ValueError(f"out dtype must be {expected_torch_dtype}, got {out_dtype}")

        if x.device != out.device:
            raise ValueError(f"x/out must be on same device, got {x.device}/{out.device}")

        if _tensors_overlap(x, out):
            raise ValueError("out must not overlap with x")

        if M == 0:
            return out

        if stream is None:
            _launch_kernel(x, out, M)
        else:
            _launch_kernel(x, out, M, stream=stream)
        return out

    return launch_softmax


__all__ = [
    "BLOCK_THREADS",
    "SUPPORTED_DTYPES",
    "compile_iluvatar_softmax",
]
