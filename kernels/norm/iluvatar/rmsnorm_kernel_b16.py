# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar FP16/BF16 RMSNorm forward kernel.

Each block normalizes one row. Odd-width rows use one element per thread, while
even-width rows use two adjacent elements and two FP32 accumulators per thread.
FP16 and BF16 share one builder and are specialized at compile time.
"""

import math
from collections.abc import Callable

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from kernels.gemm.iluvatar.common import WARP_SIZE

MAX_THREADS_PER_BLOCK = 4096
SUPPORTED_DTYPES = ("f16", "bf16")
_TORCH_DTYPE_NAMES = {
    "f16": "torch.float16",
    "bf16": "torch.bfloat16",
}


def _dtype_name(tensor) -> str:
    return str(tensor.dtype)


def _byte_range(tensor) -> tuple[int, int]:
    """Return the half-open byte range occupied by a contiguous tensor."""
    start = int(tensor.data_ptr())
    return start, start + int(tensor.numel()) * int(tensor.element_size())


def _tensors_overlap(a, b) -> bool:
    """Check whether two tensor storage ranges overlap."""
    a0, a1 = _byte_range(a)
    b0, b1 = _byte_range(b)
    return max(a0, b0) < min(a1, b1)


def _normalize_dtype(dtype: str) -> str:
    """Normalize the public FP16 alias and reject unsupported dtypes."""
    if dtype == "fp16":
        return "f16"
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"dtype must be one of {SUPPORTED_DTYPES}, got {dtype!r}")
    return dtype


def _build_rmsnorm_b16_kernel(*, N: int, eps: float, dtype: str):
    """Build a kernel specialized for the row width, epsilon, and dtype."""
    elem_dtype = fx.Float16 if dtype == "f16" else fx.BFloat16

    # This is the launch policy. For even N each
    # thread consumes two elements, hence the division by two.
    launch_warp_count = (N + WARP_SIZE - 1) // WARP_SIZE
    num_threads = min(launch_warp_count * WARP_SIZE, MAX_THREADS_PER_BLOCK)
    elements_per_load = 1 if N & 1 else 2
    block_threads = num_threads if elements_per_load == 1 else num_threads // 2

    reduction_warps = (block_threads + WARP_SIZE - 1) // WARP_SIZE
    partial_count = reduction_warps * 2

    # Two FP32 partial sums are stored for each warp.
    @fx.struct
    class _RmsNormB16Smem:
        partials: fx.Array[fx.Float32, partial_count]

    # Tensor shapes and element types:
    #   x:      (M, N), elem_dtype (Float16 or BFloat16)
    #   gamma:  (N,),   elem_dtype (Float16 or BFloat16)
    #   out:    (M, N), elem_dtype (Float16 or BFloat16)
    #   rsigma: (M,),   Float32
    #
    # Work distribution:
    #   grid.x = M: block bid processes row x[bid, :].
    #   Odd N:  thread tid processes columns tid + k * block_threads.
    #   Even N: thread tid processes adjacent columns
    #           2 * tid + k * (2 * block_threads) and the following column.
    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def _rmsnorm_b16_kernel(
        x: fx.Tensor,
        gamma: fx.Tensor,
        out: fx.Tensor,
        rsigma: fx.Tensor,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % WARP_SIZE
        warp = tid // WARP_SIZE

        fm_fast = arith.FastMathFlags.fast
        c_zero_f = fx.Float32(0.0)
        n_float = float(N)
        eps_c = eps

        smem = fx.SharedAllocator().allocate(_RmsNormB16Smem).peek()
        partials = smem.partials.view(fx.make_layout(partial_count, 1))

        def _warp_reduce_add(value):
            """Reduce one FP32 value across a warp64 using XOR shuffles."""
            reduced = value
            for sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                offset = WARP_SIZE // (2 << sh_exp)
                peer = reduced.shuffle_xor(offset, WARP_SIZE)
                reduced = reduced.addf(peer, fastmath=fm_fast)
            return reduced

        row = fx.Int32(bid)
        sum0 = c_zero_f
        sum1 = c_zero_f

        # Pass 1: load the row and accumulate squares in FP32. The two-element
        # path keeps even and odd elements in separate accumulators.
        for base in range_constexpr(0, N, block_threads * elements_per_load):
            offset = tid + base // elements_per_load
            idx0 = offset * elements_per_load
            valid0 = idx0 < N
            idx0_safe = valid0.select(idx0, 0)
            x0 = fx.Float32(x[row, idx0_safe])
            sum0 = sum0 + valid0.select(x0 * x0, c_zero_f)

            if const_expr(elements_per_load == 2):
                idx1 = idx0 + 1
                valid1 = idx1 < N
                idx1_safe = valid1.select(idx1, 0)
                x1 = fx.Float32(x[row, idx1_safe])
                sum1 = sum1 + valid1.select(x1 * x1, c_zero_f)

        warp_sum0 = _warp_reduce_add(sum0)
        warp_sum1 = _warp_reduce_add(sum1)

        # First reduction level: each warp publishes its two partial sums.
        if lane == 0:
            fx.memref_store(warp_sum0, partials, warp * 2)
        if lane == 1:
            fx.memref_store(warp_sum1, partials, warp * 2 + 1)
        gpu.barrier()

        if warp == 0:
            mean = c_zero_f
            # Second reduction level: warp 0 combines all shared partials.
            for base in range_constexpr(0, partial_count, WARP_SIZE):
                partial_idx = lane + base
                valid = partial_idx < partial_count
                partial_safe = valid.select(partial_idx, 0)
                partial = fx.memref_load(partials, partial_safe)
                chunk_sum = _warp_reduce_add(valid.select(partial, c_zero_f))
                mean = mean + chunk_sum
            if lane == 0:
                fx.memref_store(mean, partials, 0)
        gpu.barrier()

        # Convert the row sum of squares into reciprocal RMS.
        mean = fx.memref_load(partials, 0)
        rrms = fmath.rsqrt((mean / n_float) + eps_c, fastmath=fm_fast)

        # Pass 2: apply the learned scale and reciprocal RMS, then convert back
        # to the input element type.
        for base in range_constexpr(0, N, block_threads * elements_per_load):
            offset = tid + base // elements_per_load
            idx0 = offset * elements_per_load
            if idx0 < N:
                x0 = fx.Float32(x[row, idx0])
                g0 = fx.Float32(gamma[idx0])
                if const_expr(elements_per_load == 2):
                    # Preserve the element-type rounding between the two
                    # multiplications in the paired path.
                    product0 = (x0 * g0).to(elem_dtype)
                    y0 = fx.Float32(product0) * rrms
                else:
                    y0 = x0 * g0 * rrms
                out[row, idx0] = y0.to(elem_dtype)

            if const_expr(elements_per_load == 2):
                idx1 = idx0 + 1
                if idx1 < N:
                    x1 = fx.Float32(x[row, idx1])
                    g1 = fx.Float32(gamma[idx1])
                    product1 = (x1 * g1).to(elem_dtype)
                    y1 = fx.Float32(product1) * rrms
                    out[row, idx1] = y1.to(elem_dtype)

        # One FP32 reciprocal RMS value is produced per row.
        if tid == 0:
            rsigma[row] = rrms

    return _rmsnorm_b16_kernel, block_threads


def compile_iluvatar_rmsnorm_b16(
    *,
    N: int,
    eps: float,
    dtype: str,
) -> Callable:
    """Build an RMSNorm launcher specialized for FP16 or BF16.

    Returns:
        ``launch_rmsnorm_b16(x, gamma, out, rsigma, M, stream=None)``.
    """

    if N <= 0:
        raise ValueError(f"N must be > 0, got {N}")
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")
    dtype = _normalize_dtype(dtype)
    torch_dtype_name = _TORCH_DTYPE_NAMES[dtype]

    kernel, block_threads = _build_rmsnorm_b16_kernel(N=N, eps=float(eps), dtype=dtype)

    @flyc.jit
    def _launch_kernel(
        x: fx.Tensor,
        gamma: fx.Tensor,
        out: fx.Tensor,
        rsigma: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        kernel(x, gamma, out, rsigma).launch(
            grid=(m_in, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    def launch_rmsnorm_b16(x, gamma, out, rsigma, M: int, stream=None):
        """Validate tensors, launch one block per row, and return the outputs."""
        if not isinstance(M, int):
            raise ValueError(f"M must be int, got {type(M).__name__}")
        if M < 0:
            raise ValueError(f"M must be >= 0, got {M}")

        if x.dim() != 2:
            raise ValueError(f"expected x shape (M,N), got dim={x.dim()} shape={tuple(x.shape)}")
        if gamma.dim() != 1:
            raise ValueError(f"expected gamma shape (N,), got dim={gamma.dim()} shape={tuple(gamma.shape)}")
        if out.dim() != 2:
            raise ValueError(f"expected out shape (M,N), got dim={out.dim()} shape={tuple(out.shape)}")
        if rsigma.dim() != 1:
            raise ValueError(f"expected rsigma shape (M,), got dim={rsigma.dim()} shape={tuple(rsigma.shape)}")

        if tuple(x.shape) != (M, N):
            raise ValueError(f"expected x shape (M,N)=({M},{N}), got {tuple(x.shape)}")
        if tuple(gamma.shape) != (N,):
            raise ValueError(f"expected gamma shape (N,)=({N},), got {tuple(gamma.shape)}")
        if tuple(out.shape) != (M, N):
            raise ValueError(f"expected out shape (M,N)=({M},{N}), got {tuple(out.shape)}")
        if tuple(rsigma.shape) != (M,):
            raise ValueError(f"expected rsigma shape (M,)=({M},), got {tuple(rsigma.shape)}")

        if not x.is_contiguous():
            raise ValueError("x must be contiguous")
        if not gamma.is_contiguous():
            raise ValueError("gamma must be contiguous")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
        if not rsigma.is_contiguous():
            raise ValueError("rsigma must be contiguous")

        for name, tensor in (("x", x), ("gamma", gamma), ("out", out)):
            actual_dtype = _dtype_name(tensor)
            if actual_dtype != torch_dtype_name:
                raise ValueError(f"{name} dtype must be {torch_dtype_name}, got {actual_dtype}")
        if _dtype_name(rsigma) != "torch.float32":
            raise ValueError(f"rsigma dtype must be torch.float32, got {_dtype_name(rsigma)}")

        if not (x.device == gamma.device == out.device == rsigma.device):
            raise ValueError(
                f"x/gamma/out/rsigma must be on same device, got "
                f"{x.device}/{gamma.device}/{out.device}/{rsigma.device}"
            )

        if _tensors_overlap(x, out):
            raise ValueError("out must not overlap with x")
        if _tensors_overlap(gamma, out):
            raise ValueError("out must not overlap with gamma")
        if _tensors_overlap(rsigma, x) or _tensors_overlap(rsigma, gamma) or _tensors_overlap(rsigma, out):
            raise ValueError("rsigma must not overlap with x, gamma, or out")

        if M == 0:
            return out, rsigma

        if stream is None:
            _launch_kernel(x, gamma, out, rsigma, M)
        else:
            _launch_kernel(x, gamma, out, rsigma, M, stream=stream)
        return out, rsigma

    return launch_rmsnorm_b16


__all__ = [
    "MAX_THREADS_PER_BLOCK",
    "SUPPORTED_DTYPES",
    "compile_iluvatar_rmsnorm_b16",
]
