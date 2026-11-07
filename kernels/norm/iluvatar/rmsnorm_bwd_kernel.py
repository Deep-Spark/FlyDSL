# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar RMSNorm backward kernel.

The kernel follows the generic AMD implementation: one CTA processes one row,
computes ``dx`` in the input dtype, and atomically accumulates an FP32
``dweight``. The caller must zero ``dweight`` before launching.
"""

import math
from collections.abc import Callable

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from kernels.gemm.iluvatar.common import WARP_SIZE

BLOCK_THREADS = 256
RED_SLOTS = BLOCK_THREADS // WARP_SIZE
SUPPORTED_DTYPES = ("f32", "bf16", "f16")
_TORCH_DTYPE_NAMES = {
    "f32": "torch.float32",
    "bf16": "torch.bfloat16",
    "f16": "torch.float16",
}


@fx.struct
class _RmsNormBwdSmem:
    partials: fx.Array[fx.Float32, RED_SLOTS]


def _normalize_dtype(dtype: str) -> str:
    if dtype == "fp16":
        return "f16"
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"dtype must be one of {SUPPORTED_DTYPES}, got {dtype!r}")
    return dtype


def _dtype_name(tensor) -> str:
    return str(tensor.dtype)


def _byte_range(tensor) -> tuple[int, int]:
    start = int(tensor.data_ptr())
    return start, start + int(tensor.numel()) * int(tensor.element_size())


def _tensors_overlap(a, b) -> bool:
    a0, a1 = _byte_range(a)
    b0, b1 = _byte_range(b)
    return max(a0, b0) < min(a1, b1)


def _build_rmsnorm_bwd_kernel(*, N: int, dtype: str):
    elem_dtype = {
        "f32": fx.Float32,
        "bf16": fx.BFloat16,
        "f16": fx.Float16,
    }[dtype]

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def _rmsnorm_bwd_kernel(
        x: fx.Tensor,
        gamma: fx.Tensor,
        dy: fx.Tensor,
        rstd: fx.Tensor,
        dx: fx.Tensor,
        dweight: fx.Tensor,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane_id = fx.Int32(fx.lane_id)
        warp_id = tid // WARP_SIZE

        fm_fast = arith.FastMathFlags.fast
        c_zero_f = fx.Float32(0.0)
        n_float = float(N)

        smem = fx.SharedAllocator().allocate(_RmsNormBwdSmem).peek()
        partials = smem.partials.view(fx.make_layout(RED_SLOTS, 1))

        def _warp_reduce_add(value):
            reduced = value
            for sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                offset = WARP_SIZE // (2 << sh_exp)
                peer = reduced.shuffle_xor(offset, WARP_SIZE)
                reduced = reduced.addf(peer, fastmath=fm_fast)
            return reduced

        def _block_reduce_add(value):
            warp_sum = _warp_reduce_add(value)
            if lane_id == 0:
                fx.memref_store(warp_sum, partials, warp_id)
            gpu.barrier()

            if warp_id == 0:
                valid = lane_id < RED_SLOTS
                lane_safe = valid.select(lane_id, 0)
                partial = fx.memref_load(partials, lane_safe)
                block_sum = _warp_reduce_add(valid.select(partial, c_zero_f))
                if lane_id == 0:
                    fx.memref_store(block_sum, partials, 0)
            gpu.barrier()
            return fx.memref_load(partials, 0)

        row = fx.Int32(bid)
        rrms = fx.Float32(rstd[row])
        dweight_div = fx.logical_divide(dweight, fx.make_layout(1, 1))
        atomic_add_f32 = fx.make_copy_atom(fx.UniversalAtomicAdd(fx.Float32), fx.Float32)
        dweight_value = fx.make_rmem_tensor(1, fx.Float32)

        # c1 = mean(x_hat * wdy), where x_hat = x*rstd and wdy = dy*gamma.
        thread_sum = c_zero_f
        for base in range_constexpr(0, N, BLOCK_THREADS):
            idx = tid + base
            valid = idx < N
            idx_safe = valid.select(idx, 0)
            x_val = fx.Float32(x[row, idx_safe])
            dy_val = fx.Float32(dy[row, idx_safe])
            gamma_val = fx.Float32(gamma[idx_safe])
            product = (x_val * rrms) * (dy_val * gamma_val)
            thread_sum = thread_sum + valid.select(product, c_zero_f)

        c1 = _block_reduce_add(thread_sum) / n_float

        # dx = (wdy - x_hat*c1)*rstd; dweight += dy*x_hat.
        for base in range_constexpr(0, N, BLOCK_THREADS):
            idx = tid + base
            if idx < N:
                x_val = fx.Float32(x[row, idx])
                dy_val = fx.Float32(dy[row, idx])
                gamma_val = fx.Float32(gamma[idx])
                x_hat = x_val * rrms
                wdy = dy_val * gamma_val
                dx_val = (wdy - x_hat * c1) * rrms
                dx[row, idx] = dx_val if const_expr(dtype == "f32") else dx_val.to(elem_dtype)
                fx.memref_store(dy_val * x_hat, dweight_value, 0)
                fx.copy_atom_call(atomic_add_f32, dweight_value, fx.slice(dweight_div, (None, idx)))

    return _rmsnorm_bwd_kernel


def compile_iluvatar_rmsnorm_bwd(*, N: int, dtype: str) -> Callable:
    """Build an Iluvatar RMSNorm backward launcher.

    Returns ``launch(x, gamma, dy, rstd, dx, dweight, M, stream=None)``.
    ``x``, ``gamma``, ``dy`` and ``dx`` use *dtype*; ``rstd`` and ``dweight``
    are FP32. ``dweight`` is accumulated atomically and must be zero-initialized.
    """

    if N <= 0:
        raise ValueError(f"N must be > 0, got {N}")
    dtype = _normalize_dtype(dtype)
    kernel = _build_rmsnorm_bwd_kernel(N=N, dtype=dtype)
    expected_dtype = _TORCH_DTYPE_NAMES[dtype]

    @flyc.jit
    def _launch_kernel(
        x: fx.Tensor,
        gamma: fx.Tensor,
        dy: fx.Tensor,
        rstd: fx.Tensor,
        dx: fx.Tensor,
        dweight: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        kernel(x, gamma, dy, rstd, dx, dweight).launch(
            grid=(m_in, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    def launch_rmsnorm_bwd(x, gamma, dy, rstd, dx, dweight, M: int, stream=None):
        if not isinstance(M, int):
            raise ValueError(f"M must be int, got {type(M).__name__}")
        if M < 0:
            raise ValueError(f"M must be >= 0, got {M}")

        expected_shapes = {
            "x": (M, N),
            "gamma": (N,),
            "dy": (M, N),
            "rstd": (M,),
            "dx": (M, N),
            "dweight": (N,),
        }
        tensors = {
            "x": x,
            "gamma": gamma,
            "dy": dy,
            "rstd": rstd,
            "dx": dx,
            "dweight": dweight,
        }
        for name, tensor in tensors.items():
            if tuple(tensor.shape) != expected_shapes[name]:
                raise ValueError(f"expected {name} shape {expected_shapes[name]}, got {tuple(tensor.shape)}")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")

        for name in ("x", "gamma", "dy", "dx"):
            actual = _dtype_name(tensors[name])
            if actual != expected_dtype:
                raise ValueError(f"{name} dtype must be {expected_dtype}, got {actual}")
        for name in ("rstd", "dweight"):
            actual = _dtype_name(tensors[name])
            if actual != _TORCH_DTYPE_NAMES["f32"]:
                raise ValueError(f"{name} dtype must be torch.float32, got {actual}")

        device = x.device
        if any(tensor.device != device for tensor in tensors.values()):
            raise ValueError("all tensors must be on the same device")
        for output_name in ("dx", "dweight"):
            for input_name in ("x", "gamma", "dy", "rstd"):
                if _tensors_overlap(tensors[output_name], tensors[input_name]):
                    raise ValueError(f"{output_name} must not overlap with {input_name}")
        if _tensors_overlap(dx, dweight):
            raise ValueError("dx must not overlap with dweight")

        if M == 0:
            return dx, dweight
        if stream is None:
            _launch_kernel(x, gamma, dy, rstd, dx, dweight, M)
        else:
            _launch_kernel(x, gamma, dy, rstd, dx, dweight, M, stream=stream)
        return dx, dweight

    return launch_rmsnorm_bwd
