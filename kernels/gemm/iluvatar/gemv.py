# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar GEMV V1 kernel (F.linear-compatible semantics for M=1).

V1 scope:
- input shape: [K] or [1, K]
- weight shape: [N, K]
- bias: None or [N] (applied as a host-side post-process)
- dtype: fp16 / bf16 for input and weight, fp32 accumulation
- strict tile divisibility: N % TILE_N == 0 and K % TILE_K == 0
"""

from __future__ import annotations

from collections.abc import Callable

import flydsl.compiler as flyc
import flydsl.expr as fx

TILE_N = 64
TILE_K = 16
SUPPORTED_DTYPES = {"torch.float16", "torch.bfloat16"}


def _dtype_name(tensor) -> str:
    return str(tensor.dtype)


def _is_supported_dtype(tensor) -> bool:
    return _dtype_name(tensor) in SUPPORTED_DTYPES


def _validate_shapes_and_dtypes(x, w, bias, *, N: int, K: int) -> None:
    if x.dim() not in (1, 2):
        raise ValueError(f"x must be 1D or 2D, got dim={x.dim()} shape={tuple(x.shape)}")
    if x.dim() == 1 and tuple(x.shape) != (K,):
        raise ValueError(f"x shape mismatch: expected ({K},), got {tuple(x.shape)}")
    if x.dim() == 2 and tuple(x.shape) != (1, K):
        raise ValueError(f"x shape mismatch: expected (1, {K}), got {tuple(x.shape)}")

    if tuple(w.shape) != (N, K):
        raise ValueError(f"w shape mismatch: expected ({N}, {K}), got {tuple(w.shape)}")

    if bias is not None and tuple(bias.shape) != (N,):
        raise ValueError(f"bias shape mismatch: expected ({N},), got {tuple(bias.shape)}")

    if not _is_supported_dtype(x):
        raise ValueError(f"x dtype must be fp16/bf16, got {_dtype_name(x)}")
    if not _is_supported_dtype(w):
        raise ValueError(f"w dtype must be fp16/bf16, got {_dtype_name(w)}")
    if _dtype_name(x) != _dtype_name(w):
        raise ValueError(f"x and w dtype must match, got {_dtype_name(x)} vs {_dtype_name(w)}")
    if bias is not None and _dtype_name(bias) != _dtype_name(x):
        raise ValueError(f"bias dtype must match x dtype, got {_dtype_name(bias)} vs {_dtype_name(x)}")

    if N % TILE_N != 0:
        raise ValueError(f"N must be divisible by TILE_N={TILE_N}, got N={N}")
    if K % TILE_K != 0:
        raise ValueError(f"K must be divisible by TILE_K={TILE_K}, got K={K}")


def compile_iluvatar_gemv(*, N: int, K: int) -> Callable:
    """Build a GEMV launcher with F.linear-compatible semantics for M=1.

    Returns:
        A callable `launch_gemv(x, w, bias=None, out=None, stream=None)` where:
        - x: [K] or [1, K]
        - w: [N, K]
        - bias: None or [N]
        - out: optional preallocated output tensor, shape [N]
    """
    if N <= 0 or K <= 0:
        raise ValueError(f"N and K must be positive, got N={N}, K={K}")
    if N % TILE_N != 0:
        raise ValueError(f"N must be divisible by TILE_N={TILE_N}, got N={N}")
    if K % TILE_K != 0:
        raise ValueError(f"K must be divisible by TILE_K={TILE_K}, got K={K}")

    def _build_gemv_kernel(out_elem_dtype):
        @flyc.kernel(known_block_size=[TILE_N, 1, 1])
        def iluvatar_gemv(x_vec: fx.Tensor, w_mat: fx.Tensor, y_vec: fx.Tensor):
            tid = fx.thread_idx.x
            bid = fx.block_idx.x
            n_idx = fx.Int32(bid * TILE_N + tid)

            x = fx.make_view(fx.get_iter(x_vec), fx.make_layout((K,), (1,)))
            w = fx.make_view(fx.get_iter(w_mat), fx.make_layout((N, K), (K, 1)))
            y = fx.make_view(fx.get_iter(y_vec), fx.make_layout((N,), (1,)))

            init_state = [fx.Float32(0.0)]
            results = init_state
            for k_idx, state in fx.range(0, K, 2, init=init_state):
                acc = state[0]
                k0_i32 = fx.Int32(k_idx)
                k1_i32 = fx.Int32(k_idx + 1)
                x0_val = fx.Float32(x[k0_i32])
                x1_val = fx.Float32(x[k1_i32])
                w0_val = fx.Float32(w[n_idx, k0_i32])
                w1_val = fx.Float32(w[n_idx, k1_i32])
                results = yield [acc + (x0_val * w0_val) + (x1_val * w1_val)]

            # Single loop-carried value is returned as scalar ArithValue.
            y[n_idx] = fx.Float32(results).to(out_elem_dtype)

        return iluvatar_gemv

    def _build_launcher(kernel_fn):
        @flyc.jit
        def _launch_kernel(x_vec: fx.Tensor, w_mat: fx.Tensor, y_vec: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
            kernel_fn(x_vec, w_mat, y_vec).launch(
                grid=(N // TILE_N, 1, 1),
                block=(TILE_N, 1, 1),
                stream=stream,
            )

        return _launch_kernel

    _launchers = {
        "torch.float16": _build_launcher(_build_gemv_kernel(fx.Float16)),
        "torch.bfloat16": _build_launcher(_build_gemv_kernel(fx.BFloat16)),
    }

    def launch_gemv(x, w, bias=None, out=None, stream=None):
        """Run GEMV and return tensor shaped like F.linear output rank."""
        _validate_shapes_and_dtypes(x, w, bias, N=N, K=K)

        input_was_2d = x.dim() == 2
        x_vec = x.reshape(K) if input_was_2d else x

        if out is None:
            out_vec = x_vec.new_empty((N,))
        else:
            if tuple(out.shape) != (N,):
                raise ValueError(f"out shape mismatch: expected ({N},), got {tuple(out.shape)}")
            if _dtype_name(out) != _dtype_name(x_vec):
                raise ValueError(f"out dtype must match x dtype, got {_dtype_name(out)} vs {_dtype_name(x_vec)}")
            out_vec = out

        launch_kernel = _launchers[_dtype_name(x_vec)]
        if stream is None:
            launch_kernel(x_vec, w, out_vec)
        else:
            launch_kernel(x_vec, w, out_vec, stream=stream)

        if bias is not None:
            out_vec.add_(bias)

        return out_vec.view(1, N) if input_was_2d else out_vec

    return launch_gemv


__all__ = ["TILE_K", "TILE_N", "compile_iluvatar_gemv"]
