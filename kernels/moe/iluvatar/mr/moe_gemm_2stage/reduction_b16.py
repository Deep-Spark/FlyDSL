# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar b16 top-k reduction with FP32 accumulation."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from kernels.gemm.iluvatar.common import WARP_SIZE

B16_DTYPES = ("f16", "bf16")
_DTYPE_FX = {"f16": fx.Float16, "bf16": fx.BFloat16}

BLOCK_THREADS = 4 * WARP_SIZE
VALUES_PER_THREAD = 4


@functools.lru_cache(maxsize=128)
def compile_iluvatar_mr_moe_reduction_b16(
    *,
    topk: int,
    model_dim: int,
    dtype: str = "bf16",
    use_valid_mask: bool = False,
):
    """Compile ``Y[t,d] = sum_k X[t,k,d]`` for Iluvatar b16 MoE.

    Returned launcher is ``launch(X, Y, valid_mask, tokens, stream=None)``.
    ``X[tokens,topk,model_dim]`` and ``Y[tokens,model_dim]`` use ``dtype``.
    Accumulation is FP32. If ``use_valid_mask`` is true, ``valid_mask`` must be
    contiguous uint8 or int8 ``[tokens,topk]``; any nonzero byte is valid.
    When masking is disabled, ``valid_mask`` may be ``None``.
    """
    if dtype not in B16_DTYPES:
        raise ValueError(f"dtype must be one of {B16_DTYPES}, got {dtype!r}")
    if topk <= 0 or topk > 255:
        raise ValueError(f"topk must be in [1,255], got {topk}")
    if model_dim <= 0:
        raise ValueError(f"model_dim must be positive, got {model_dim}")
    elem_dtype = _DTYPE_FX[dtype]
    cols_per_cta = BLOCK_THREADS * VALUES_PER_THREAD
    col_tiles = (model_dim + cols_per_cta - 1) // cols_per_cta

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def reduction(
        X: fx.Tensor,
        Y: fx.Tensor,
        valid_mask: fx.Tensor,
        tokens_in: fx.Int32,
    ):
        token = fx.Int32(fx.block_idx.x)
        tile = fx.Int32(fx.block_idx.y)
        tid = fx.Int32(fx.thread_idx.x)
        if token < fx.Int32(tokens_in):
            for vi in fx.range_constexpr(VALUES_PER_THREAD):
                col = (
                    tile * fx.Int32(cols_per_cta)
                    + tid
                    + fx.Int32(vi * BLOCK_THREADS)
                )
                if col < fx.Int32(model_dim):
                    acc = fx.Float32(0.0)
                    for slot in fx.range_constexpr(topk):
                        if fx.const_expr(use_valid_mask):
                            valid = fx.Int8(valid_mask[token, slot]) != fx.Int8(0)
                            value = fx.arith.select(
                                valid,
                                X[token, slot, col].to(fx.Float32),
                                fx.Float32(0.0),
                            )
                        else:
                            value = X[token, slot, col].to(fx.Float32)
                        acc = acc + value
                    Y[token, col] = acc.to(elem_dtype)

    @flyc.jit
    def launch_jit(
        X: fx.Tensor,
        Y: fx.Tensor,
        valid_mask: fx.Tensor,
        tokens_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        reduction(X, Y, valid_mask, tokens_in).launch(
            grid=(tokens_in, col_tiles, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    def launch(X, Y, valid_mask, tokens, stream=None):
        if use_valid_mask and valid_mask is None:
            raise ValueError("valid_mask is required when use_valid_mask=True")
        mask = X if valid_mask is None else valid_mask
        args = (X, Y, mask, fx.Int32(int(tokens)))
        launch_jit(*args) if stream is None else launch_jit(*args, stream=stream)
        return Y

    launch.dtype = dtype
    launch.use_valid_mask = use_valid_mask
    return launch


__all__ = ["compile_iluvatar_mr_moe_reduction_b16"]
