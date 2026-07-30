# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar RMSNorm forward kernels.

Entries:
- ``compile_iluvatar_rmsnorm`` (V1): fp32 input/output, ``out = x * rrms * gamma``.
- ``compile_iluvatar_rmsnorm_dynamicquant`` (V2): fp32 input, i8 output with
  per-row dynamic symmetric quantization.
  ``y = x * rrms * gamma``; ``y_scale = amax(|y|) / 127`` (0 -> 1 protected);
  ``out = truncate_to_i8(y / y_scale)``.
- ``compile_iluvatar_rmsnorm_smoothquant`` (V3): V2 plus per-channel
  ``x_scale[N]`` applied after gamma:
  ``y = x * rrms * gamma * x_scale``.

All kernels use one CTA per row (``grid=(M, 1, 1)``, ``block=(256, 1, 1)``)
and a scalar generic algorithm over arbitrary ``N`` (tail-safe).
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
TORCH_I8_NAME = "torch.int8"

# i8 symmetric-quant max magnitude. Uses 127 (not 128) so ``scale = amax/qmax``
# guarantees ``|q| <= 127`` and truncate-to-i8 never lands on -128.
QUANT_I8_MAX = 127.0


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

        smem = fx.SharedAllocator().allocate(_RmsNormSmem).peek()
        s_red = smem.s_red.view(fx.make_layout(RED_SLOTS, 1))

        def _warp_reduce_add(xf):
            w = xf
            for sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def _block_reduce_add(xf):
            if const_expr(RED_SLOTS == 1):
                return _warp_reduce_add(xf)

            lane_id = fx.Int32(fx.lane_id)
            warp_id = tid // WARP_SIZE

            w_red = _warp_reduce_add(xf)
            if lane_id == 0:
                fx.memref_store(w_red, s_red, warp_id)
            gpu.barrier()

            if warp_id == 0:
                in_range = lane_id < RED_SLOTS
                lane_safe = in_range.select(lane_id, 0)
                v = fx.memref_load(s_red, lane_safe)
                ww = in_range.select(v, c_zero_f)
                ww = _warp_reduce_add(ww)
                if lane_id == 0:
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


def _build_rmsnorm_quant_kernel(*, N: int, eps: float, is_smooth: bool):
    """Shared V2/V3 quant kernel factory.

    Two ``@flyc.kernel`` signatures (no dummy ``x_scale`` on the DQ path):
    DQ keeps ``(x, gamma, out, y_scale)``; SQ is
    ``(x, gamma, x_scale, out, y_scale)``. Bodies are duplicated inside each
    kernel so nested reducers stay visible to the AST rewriter (module-level /
    outer-factory helpers would skip it -- same constraint as V1/V2).
    """
    if RED_SLOTS <= 0:
        raise ValueError(f"internal error: RED_SLOTS must be positive, got {RED_SLOTS}")

    if is_smooth:

        @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
        def _rmsnorm_sq_kernel(
            x: fx.Tensor,
            gamma: fx.Tensor,
            x_scale: fx.Tensor,
            out: fx.Tensor,
            y_scale: fx.Tensor,
        ):
            tid = fx.thread_idx.x
            bid = fx.block_idx.x

            fm_fast = arith.FastMathFlags.fast
            c_zero_f = fx.Float32(0.0)
            c_one_f = fx.Float32(1.0)
            c_neg_inf = fx.Float32(float("-inf"))
            c_qmax = fx.Float32(QUANT_I8_MAX)
            c_abs_mask = fx.Uint32(0x7FFFFFFF)
            n_float = float(N)
            eps_c = eps

            smem = fx.SharedAllocator().allocate(_RmsNormSmem).peek()
            s_red = smem.s_red.view(fx.make_layout(RED_SLOTS, 1))

            def _warp_reduce_add(xf):
                w = xf
                for sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                    off = WARP_SIZE // (2 << sh_exp)
                    peer = w.shuffle_xor(off, WARP_SIZE)
                    w = w.addf(peer, fastmath=fm_fast)
                return w

            def _warp_reduce_max(xf):
                w = xf
                for sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                    off = WARP_SIZE // (2 << sh_exp)
                    peer = w.shuffle_xor(off, WARP_SIZE)
                    w = w.maximumf(peer)
                return w

            def _block_reduce_add(xf):
                if const_expr(RED_SLOTS == 1):
                    return _warp_reduce_add(xf)

                lane_id = fx.Int32(fx.lane_id)
                warp_id = tid // WARP_SIZE

                w_red = _warp_reduce_add(xf)
                if lane_id == 0:
                    fx.memref_store(w_red, s_red, warp_id)
                gpu.barrier()

                if warp_id == 0:
                    in_range = lane_id < RED_SLOTS
                    lane_safe = in_range.select(lane_id, 0)
                    v = fx.memref_load(s_red, lane_safe)
                    ww = in_range.select(v, c_zero_f)
                    ww = _warp_reduce_add(ww)
                    if lane_id == 0:
                        fx.memref_store(ww, s_red, 0)
                gpu.barrier()

                return fx.memref_load(s_red, 0)

            def _block_reduce_max(xf):
                if const_expr(RED_SLOTS == 1):
                    return _warp_reduce_max(xf)

                lane_id = fx.Int32(fx.lane_id)
                warp_id = tid // WARP_SIZE

                w_red = _warp_reduce_max(xf)
                if lane_id == 0:
                    fx.memref_store(w_red, s_red, warp_id)
                gpu.barrier()

                if warp_id == 0:
                    in_range = lane_id < RED_SLOTS
                    lane_safe = in_range.select(lane_id, 0)
                    v = fx.memref_load(s_red, lane_safe)
                    ww = in_range.select(v, c_neg_inf)
                    ww = _warp_reduce_max(ww)
                    if lane_id == 0:
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

            thread_amax = c_zero_f
            for base_idx in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx
                valid = idx < N
                idx_safe = valid.select(idx, 0)
                x_val = fx.Float32(x[row, idx_safe])
                g_val = fx.Float32(gamma[idx_safe])
                s_val = fx.Float32(x_scale[idx_safe])
                y = x_val * rrms * g_val * s_val
                y_abs_bits = y.bitcast(fx.Uint32) & c_abs_mask
                y_abs = y_abs_bits.bitcast(fx.Float32)
                thread_amax = thread_amax.maximumf(valid.select(y_abs, c_zero_f))

            row_max = _block_reduce_max(thread_amax)
            scale = row_max / c_qmax
            final_scale = (scale == c_zero_f).select(c_one_f, scale)
            if tid == 0:
                y_scale[row] = final_scale
            inv_scale = c_one_f / final_scale

            for base_idx in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx
                if idx < N:
                    x_val = fx.Float32(x[row, idx])
                    g_val = fx.Float32(gamma[idx])
                    s_val = fx.Float32(x_scale[idx])
                    y = x_val * rrms * g_val * s_val
                    q = y * inv_scale
                    out[row, idx] = q.to(fx.Int8)

        return _rmsnorm_sq_kernel

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def _rmsnorm_dq_kernel(x: fx.Tensor, gamma: fx.Tensor, out: fx.Tensor, y_scale: fx.Tensor):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x

        fm_fast = arith.FastMathFlags.fast
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_neg_inf = fx.Float32(float("-inf"))
        c_qmax = fx.Float32(QUANT_I8_MAX)
        c_abs_mask = fx.Uint32(0x7FFFFFFF)
        n_float = float(N)
        eps_c = eps

        smem = fx.SharedAllocator().allocate(_RmsNormSmem).peek()
        s_red = smem.s_red.view(fx.make_layout(RED_SLOTS, 1))

        def _warp_reduce_add(xf):
            w = xf
            for sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def _warp_reduce_max(xf):
            w = xf
            for sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.maximumf(peer)
            return w

        def _block_reduce_add(xf):
            if const_expr(RED_SLOTS == 1):
                return _warp_reduce_add(xf)

            lane_id = fx.Int32(fx.lane_id)
            warp_id = tid // WARP_SIZE

            w_red = _warp_reduce_add(xf)
            if lane_id == 0:
                fx.memref_store(w_red, s_red, warp_id)
            gpu.barrier()

            if warp_id == 0:
                in_range = lane_id < RED_SLOTS
                lane_safe = in_range.select(lane_id, 0)
                v = fx.memref_load(s_red, lane_safe)
                ww = in_range.select(v, c_zero_f)
                ww = _warp_reduce_add(ww)
                if lane_id == 0:
                    fx.memref_store(ww, s_red, 0)
            gpu.barrier()

            return fx.memref_load(s_red, 0)

        def _block_reduce_max(xf):
            if const_expr(RED_SLOTS == 1):
                return _warp_reduce_max(xf)

            lane_id = fx.Int32(fx.lane_id)
            warp_id = tid // WARP_SIZE

            w_red = _warp_reduce_max(xf)
            if lane_id == 0:
                fx.memref_store(w_red, s_red, warp_id)
            gpu.barrier()

            if warp_id == 0:
                in_range = lane_id < RED_SLOTS
                lane_safe = in_range.select(lane_id, 0)
                v = fx.memref_load(s_red, lane_safe)
                ww = in_range.select(v, c_neg_inf)
                ww = _warp_reduce_max(ww)
                if lane_id == 0:
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

        thread_amax = c_zero_f
        for base_idx in range_constexpr(0, N, BLOCK_THREADS):
            idx = tid + base_idx
            valid = idx < N
            idx_safe = valid.select(idx, 0)
            x_val = fx.Float32(x[row, idx_safe])
            g_val = fx.Float32(gamma[idx_safe])
            y = x_val * rrms * g_val
            y_abs_bits = y.bitcast(fx.Uint32) & c_abs_mask
            y_abs = y_abs_bits.bitcast(fx.Float32)
            thread_amax = thread_amax.maximumf(valid.select(y_abs, c_zero_f))

        row_max = _block_reduce_max(thread_amax)
        scale = row_max / c_qmax
        final_scale = (scale == c_zero_f).select(c_one_f, scale)
        if tid == 0:
            y_scale[row] = final_scale
        inv_scale = c_one_f / final_scale

        for base_idx in range_constexpr(0, N, BLOCK_THREADS):
            idx = tid + base_idx
            if idx < N:
                x_val = fx.Float32(x[row, idx])
                g_val = fx.Float32(gamma[idx])
                y = x_val * rrms * g_val
                q = y * inv_scale
                out[row, idx] = q.to(fx.Int8)

    return _rmsnorm_dq_kernel


def _validate_quant_launch_args(x, gamma, out, y_scale, M: int, N: int, *, x_scale=None):
    """Shared shape/dtype/contig/device/overlap guards for V2/V3 launchers."""
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
    if y_scale.dim() != 1:
        raise ValueError(f"expected y_scale shape (M,), got dim={y_scale.dim()} shape={tuple(y_scale.shape)}")

    if tuple(x.shape) != (M, N):
        raise ValueError(f"expected x shape (M,N)=({M},{N}), got {tuple(x.shape)}")
    if tuple(out.shape) != (M, N):
        raise ValueError(f"expected out shape (M,N)=({M},{N}), got {tuple(out.shape)}")
    if tuple(gamma.shape) != (N,):
        raise ValueError(f"expected gamma shape (N,)=({N},), got {tuple(gamma.shape)}")
    if tuple(y_scale.shape) != (M,):
        raise ValueError(f"expected y_scale shape (M,)=({M},), got {tuple(y_scale.shape)}")

    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if not gamma.is_contiguous():
        raise ValueError("gamma must be contiguous")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    if not y_scale.is_contiguous():
        raise ValueError("y_scale must be contiguous")

    x_dtype = _dtype_name(x)
    g_dtype = _dtype_name(gamma)
    out_dtype = _dtype_name(out)
    ys_dtype = _dtype_name(y_scale)
    if x_dtype != TORCH_F32_NAME:
        raise ValueError(f"x dtype must be {TORCH_F32_NAME}, got {x_dtype}")
    if g_dtype != TORCH_F32_NAME:
        raise ValueError(f"gamma dtype must be {TORCH_F32_NAME}, got {g_dtype}")
    if out_dtype != TORCH_I8_NAME:
        raise ValueError(f"out dtype must be {TORCH_I8_NAME}, got {out_dtype}")
    if ys_dtype != TORCH_F32_NAME:
        raise ValueError(f"y_scale dtype must be {TORCH_F32_NAME}, got {ys_dtype}")

    if x_scale is not None:
        if x_scale.dim() != 1:
            raise ValueError(f"expected x_scale shape (N,), got dim={x_scale.dim()} shape={tuple(x_scale.shape)}")
        if tuple(x_scale.shape) != (N,):
            raise ValueError(f"expected x_scale shape (N,)=({N},), got {tuple(x_scale.shape)}")
        if not x_scale.is_contiguous():
            raise ValueError("x_scale must be contiguous")
        xs_dtype = _dtype_name(x_scale)
        if xs_dtype != TORCH_F32_NAME:
            raise ValueError(f"x_scale dtype must be {TORCH_F32_NAME}, got {xs_dtype}")

    devices = [x.device, gamma.device, out.device, y_scale.device]
    names = ["x", "gamma", "out", "y_scale"]
    if x_scale is not None:
        devices.append(x_scale.device)
        names.append("x_scale")
    if any(d != devices[0] for d in devices[1:]):
        joined = "/".join(str(d) for d in devices)
        raise ValueError(f"{'/'.join(names)} must be on same device, got {joined}")

    if _tensors_overlap(x, out):
        raise ValueError("out must not overlap with x")
    if _tensors_overlap(x, y_scale):
        raise ValueError("y_scale must not overlap with x")
    if _tensors_overlap(gamma, y_scale):
        raise ValueError("y_scale must not overlap with gamma")
    if _tensors_overlap(out, y_scale):
        raise ValueError("y_scale must not overlap with out")

    if x_scale is not None:
        if _tensors_overlap(x_scale, x):
            raise ValueError("x_scale must not overlap with x")
        if _tensors_overlap(x_scale, gamma):
            raise ValueError("x_scale must not overlap with gamma")
        if _tensors_overlap(x_scale, out):
            raise ValueError("x_scale must not overlap with out")
        if _tensors_overlap(x_scale, y_scale):
            raise ValueError("x_scale must not overlap with y_scale")


def compile_iluvatar_rmsnorm_dynamicquant(*, N: int, eps: float) -> Callable:
    """Build Iluvatar RMSNorm dynamic-quant launcher (V2).

    Semantics: ``y = x * rsqrt(mean(x^2) + eps) * gamma``, then per-row symmetric
    dynamic quantization ``y_scale = amax(|y|) / 127`` (0 -> 1 protected),
    ``out = truncate_to_i8(y / y_scale)``.

    Args:
        N: Hidden size. Compile-time constant and must be ``> 0``.
        eps: Compile-time epsilon and must be ``> 0``.

    Returns:
        ``launch_rmsnorm_dq(x, gamma, out, y_scale, M, stream=None)``.
    """
    if N <= 0:
        raise ValueError(f"N must be > 0, got {N}")
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")

    kernel = _build_rmsnorm_quant_kernel(N=N, eps=float(eps), is_smooth=False)

    @flyc.jit
    def _launch_kernel(
        x: fx.Tensor,
        gamma: fx.Tensor,
        out: fx.Tensor,
        y_scale: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        kernel(x, gamma, out, y_scale).launch(
            grid=(m_in, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    def launch_rmsnorm_dq(x, gamma, out, y_scale, M: int, stream=None):
        _validate_quant_launch_args(x, gamma, out, y_scale, M, N)
        if M == 0:
            return out, y_scale
        if stream is None:
            _launch_kernel(x, gamma, out, y_scale, M)
        else:
            _launch_kernel(x, gamma, out, y_scale, M, stream=stream)
        return out, y_scale

    return launch_rmsnorm_dq


def compile_iluvatar_rmsnorm_smoothquant(*, N: int, eps: float) -> Callable:
    """Build Iluvatar RMSNorm SmoothQuant launcher (V3).

    Semantics: ``y = x * rsqrt(mean(x^2) + eps) * gamma * x_scale``, then
    per-row symmetric dynamic quantization ``y_scale = amax(|y|) / 127``
    (0 -> 1 protected), ``out = truncate_to_i8(y / y_scale)``.

    Args:
        N: Hidden size. Compile-time constant and must be ``> 0``.
        eps: Compile-time epsilon and must be ``> 0``.

    Returns:
        ``launch_rmsnorm_sq(x, gamma, x_scale, out, y_scale, M, stream=None)``.
    """
    if N <= 0:
        raise ValueError(f"N must be > 0, got {N}")
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")

    kernel = _build_rmsnorm_quant_kernel(N=N, eps=float(eps), is_smooth=True)

    @flyc.jit
    def _launch_kernel(
        x: fx.Tensor,
        gamma: fx.Tensor,
        x_scale: fx.Tensor,
        out: fx.Tensor,
        y_scale: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        kernel(x, gamma, x_scale, out, y_scale).launch(
            grid=(m_in, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    def launch_rmsnorm_sq(x, gamma, x_scale, out, y_scale, M: int, stream=None):
        _validate_quant_launch_args(x, gamma, out, y_scale, M, N, x_scale=x_scale)
        if M == 0:
            return out, y_scale
        if stream is None:
            _launch_kernel(x, gamma, x_scale, out, y_scale, M)
        else:
            _launch_kernel(x, gamma, x_scale, out, y_scale, M, stream=stream)
        return out, y_scale

    return launch_rmsnorm_sq


__all__ = [
    "BLOCK_THREADS",
    "QUANT_I8_MAX",
    "SUPPORTED_DTYPE",
    "compile_iluvatar_rmsnorm",
    "compile_iluvatar_rmsnorm_dynamicquant",
    "compile_iluvatar_rmsnorm_smoothquant",
]
