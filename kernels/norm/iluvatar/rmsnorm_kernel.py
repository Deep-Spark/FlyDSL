# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar RMSNorm V1 (fp32-only, forward-only).

V1 scope is intentionally narrow:
- 2D contiguous input/output: ``x, out`` are ``(M, N)``
- contiguous ``gamma``: ``(N,)``
- dtype is compile-time/runtime restricted to ``f32`` only
- one CTA per row: ``grid=(M,1,1)``, ``block=(256,1,1)``
- generic scalar two-pass algorithm for arbitrary ``N`` (tail-safe)

Formula:
    ``out[row, col] = x[row, col] * rsqrt(mean(x[row, :]^2) + eps) * gamma[col]``
"""

import math
from collections.abc import Callable

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from kernels.gemm.iluvatar.common import WARP_SIZE

BLOCK_THREADS = 256
RED_SLOTS = BLOCK_THREADS // WARP_SIZE
SUPPORTED_DTYPE = "f32"
TORCH_F32_NAME = "torch.float32"


@fx.struct
class _RmsNormSmem:
    s_red: fx.Array[fx.Float32, RED_SLOTS]


def _dtype_name(tensor) -> str:
    return str(tensor.dtype)


def _byte_range(tensor) -> tuple[int, int]:
    start = int(tensor.data_ptr())
    end = start + int(tensor.numel()) * int(tensor.element_size())
    return start, end


def _tensors_overlap(a, b) -> bool:
    a0, a1 = _byte_range(a)
    b0, b1 = _byte_range(b)
    return max(a0, b0) < min(a1, b1)


def _build_rmsnorm_kernel(*, N: int, eps: float):
    if RED_SLOTS <= 0:
        raise ValueError(f"internal error: RED_SLOTS must be positive, got {RED_SLOTS}")

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def _rmsnorm_kernel(x: fx.Tensor, gamma: fx.Tensor, out: fx.Tensor):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x

        fm_fast = arith.FastMathFlags.fast
        c_zero_f = fx.Float32(0.0)
        n_float = float(N)
        eps_c = eps

        lds = fx.SharedAllocator().allocate(_RmsNormSmem).peek()
        s_red = lds.s_red.view(fx.make_layout(RED_SLOTS, 1))

        def _wave_reduce_add(xf):
            w = xf
            for sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def _block_reduce_add(xf):
            if const_expr(RED_SLOTS == 1):
                return _wave_reduce_add(xf)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE

            wsum = _wave_reduce_add(xf)
            if lane == 0:
                fx.memref_store(wsum, s_red, wave)
            gpu.barrier()

            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v = fx.memref_load(s_red, lane_safe)
                ww = in_range.select(v, c_zero_f)
                ww = _wave_reduce_add(ww)
                if lane == 0:
                    fx.memref_store(ww, s_red, 0)
            gpu.barrier()

            return fx.memref_load(s_red, 0)

        row = fx.Int32(bid)
        thread_sumsq = c_zero_f
        for base_idx in range_constexpr(0, N, BLOCK_THREADS):
            idx = tid + base_idx
            valid = idx < N
            idx_safe = valid.select(idx, 0)
            x_val = fx.Float32(x[row, idx_safe])
            x2 = x_val * x_val
            thread_sumsq = thread_sumsq + valid.select(x2, c_zero_f)

        sum_sq = _block_reduce_add(thread_sumsq)
        rrms = fmath.rsqrt((sum_sq / n_float) + eps_c, fastmath=fm_fast)

        for base_idx in range_constexpr(0, N, BLOCK_THREADS):
            idx = tid + base_idx
            if idx < N:
                x_val = fx.Float32(x[row, idx])
                g_val = fx.Float32(gamma[idx])
                out[row, idx] = x_val * rrms * g_val

    return _rmsnorm_kernel


def compile_iluvatar_rmsnorm(*, N: int, eps: float, dtype: str = SUPPORTED_DTYPE) -> Callable:
    """Build Iluvatar RMSNorm V1 launcher.

    Args:
        N: Hidden size. Compile-time constant and must be ``> 0``.
        eps: Compile-time epsilon and must be ``> 0``.
        dtype: Exposed for future extension; V1 only accepts ``"f32"``.

    Returns:
        ``launch_rmsnorm(x, gamma, out, M, stream=None)``
    """
    if N <= 0:
        raise ValueError(f"N must be > 0, got {N}")
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")
    if dtype != SUPPORTED_DTYPE:
        raise ValueError(f"dtype must be '{SUPPORTED_DTYPE}', got {dtype!r}")

    kernel = _build_rmsnorm_kernel(N=N, eps=float(eps))

    @flyc.jit
    def _launch_kernel(
        x: fx.Tensor,
        gamma: fx.Tensor,
        out: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        kernel(x, gamma, out).launch(
            grid=(m_in, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    def launch_rmsnorm(x, gamma, out, M: int, stream=None):
        if not isinstance(M, int):
            raise ValueError(f"M must be int, got {type(M).__name__}")
        if M < 0:
            raise ValueError(f"M must be >= 0, got {M}")

        if x.dim() != 2:
            raise ValueError(f"expected x shape (M,N), got dim={x.dim()} shape={tuple(x.shape)}")
        if out.dim() != 2:
            raise ValueError(f"expected out shape (M,N), got dim={out.dim()} shape={tuple(out.shape)}")
        if gamma.dim() != 1:
            raise ValueError(f"expected gamma shape (N,), got dim={gamma.dim()} shape={tuple(gamma.shape)}")

        if tuple(x.shape) != (M, N):
            raise ValueError(f"expected x shape (M,N)=({M},{N}), got {tuple(x.shape)}")
        if tuple(out.shape) != (M, N):
            raise ValueError(f"expected out shape (M,N)=({M},{N}), got {tuple(out.shape)}")
        if tuple(gamma.shape) != (N,):
            raise ValueError(f"expected gamma shape (N,)=({N},), got {tuple(gamma.shape)}")

        if not x.is_contiguous():
            raise ValueError("x must be contiguous")
        if not gamma.is_contiguous():
            raise ValueError("gamma must be contiguous")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")

        x_dtype = _dtype_name(x)
        g_dtype = _dtype_name(gamma)
        out_dtype = _dtype_name(out)
        if x_dtype != TORCH_F32_NAME:
            raise ValueError(f"x dtype must be {TORCH_F32_NAME}, got {x_dtype}")
        if g_dtype != TORCH_F32_NAME:
            raise ValueError(f"gamma dtype must be {TORCH_F32_NAME}, got {g_dtype}")
        if out_dtype != TORCH_F32_NAME:
            raise ValueError(f"out dtype must be {TORCH_F32_NAME}, got {out_dtype}")

        if x.device != gamma.device or x.device != out.device:
            raise ValueError(f"x/gamma/out must be on same device, got {x.device}/{gamma.device}/{out.device}")

        if _tensors_overlap(x, out):
            raise ValueError("out must not overlap with x")

        if M == 0:
            return out

        if stream is None:
            _launch_kernel(x, gamma, out, M)
        else:
            _launch_kernel(x, gamma, out, M, stream=stream)
        return out

    return launch_rmsnorm


__all__ = [
    "BLOCK_THREADS",
    "SUPPORTED_DTYPE",
    "compile_iluvatar_rmsnorm",
]
