# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar fused NeoX RoPE and flash-layout KV-cache writes.

V1 supports bf16/f16 full-dimension NeoX rotation with i32 positions and slot
mapping. K/V cache tensors use flash layout
``[num_blocks, block_size, num_kv_heads, head_dim]``. FP8 cache scaling,
non-flash cache layout, and i64 metadata are intentionally rejected.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from kernels.gemm.iluvatar.common import WARP_SIZE


def build_fused_rope_cache_module(
    head_dim: int = 64,
    rotary_dim: int = -1,
    num_q_heads: int = 8,
    num_kv_heads: int = 1,
    block_size: int = 16,
    is_neox: bool = True,
    flash_layout: bool = True,
    dtype_str: str = "bf16",
    apply_scale: bool = False,
    reuse_freqs_front_part: bool = True,
    pos_dtype: str = "i32",
):
    """Build the baseline Iluvatar fused RoPE/cache launcher.

    The returned launcher's tensor and scalar argument order matches the AMD
    builder. ``KScale`` and ``VScale`` are accepted for ABI compatibility but
    unused because FP8 cache scaling is not supported in this version.
    """
    if rotary_dim == -1:
        rotary_dim = head_dim
    if not is_neox:
        raise NotImplementedError("Iluvatar fused RoPE V1 supports NeoX layout only")
    if rotary_dim != head_dim:
        raise NotImplementedError("Iluvatar fused RoPE V1 does not support partial rotation")
    if dtype_str not in ("bf16", "f16"):
        raise ValueError(f"dtype_str must be 'bf16' or 'f16', got {dtype_str!r}")
    if not flash_layout:
        raise NotImplementedError("Iluvatar fused RoPE V1 supports flash_layout=True only")
    if apply_scale:
        raise NotImplementedError("Iluvatar fused RoPE V1 does not support FP8 KV-cache scaling")
    if pos_dtype != "i32":
        raise NotImplementedError("Iluvatar fused RoPE V1 supports pos_dtype='i32' only")
    if head_dim <= 0 or head_dim % 2:
        raise ValueError(f"head_dim must be a positive even integer, got {head_dim}")
    if head_dim % WARP_SIZE:
        raise ValueError(f"head_dim must be divisible by the Iluvatar warp size {WARP_SIZE}, got {head_dim}")
    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("num_q_heads and num_kv_heads must be positive")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    elem_dtype = fx.BFloat16 if dtype_str == "bf16" else fx.Float16
    elems_per_lane = max(1, (head_dim + WARP_SIZE - 1) // WARP_SIZE)
    if head_dim % elems_per_lane:
        raise ValueError(f"head_dim must be divisible by the Iluvatar per-lane width {elems_per_lane}, got {head_dim}")
    half_dim = head_dim // 2
    if half_dim % elems_per_lane:
        raise ValueError(
            f"head_dim / 2 must be divisible by the Iluvatar per-lane width {elems_per_lane}, got {head_dim}"
        )
    vectors_per_half = half_dim // elems_per_lane
    max_heads = max(num_q_heads, num_kv_heads)

    @flyc.kernel(known_block_size=[WARP_SIZE, 1, 1])
    def fused_qk_rope_reshape_and_cache(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        Positions: fx.Tensor,
        CosCache: fx.Tensor,
        SinCache: fx.Tensor,
        SlotMapping: fx.Tensor,
        KeyCache: fx.Tensor,
        ValueCache: fx.Tensor,
        Q_out: fx.Tensor,
        K_out: fx.Tensor,
        KScale: fx.Tensor,
        VScale: fx.Tensor,
    ):
        head = fx.Int32(fx.block_idx.x)
        token = fx.Int32(fx.block_idx.y)
        lane = fx.Int32(fx.thread_idx.x)
        pair_lane = lane ^ fx.Int32(vectors_per_half)
        width = fx.Int32(WARP_SIZE)
        position = fx.Int32(Positions[token])

        def rotate_elem(src, dim):
            freq_dim = dim % fx.Int32(half_dim) if reuse_freqs_front_part else dim
            cos = fx.Float32(CosCache[position, freq_dim])
            sin = fx.Float32(SinCache[position, freq_dim])
            current = fx.Float32(src[token, head, dim])
            paired = current.shuffle_idx(pair_lane, width)
            rotated = current * cos
            sin_term = paired * sin
            return (dim < fx.Int32(half_dim)).select(rotated - sin_term, rotated + sin_term).to(elem_dtype)

        if head < fx.Int32(num_q_heads):
            for elem_offset in range_constexpr(elems_per_lane):
                dim = lane * fx.Int32(elems_per_lane) + fx.Int32(elem_offset)
                Q_out[token, head, dim] = rotate_elem(Q, dim)

        if head < fx.Int32(num_kv_heads):
            slot = fx.Int32(SlotMapping[token])

            for elem_offset in range_constexpr(elems_per_lane):
                dim = lane * fx.Int32(elems_per_lane) + fx.Int32(elem_offset)
                rotated_elem = rotate_elem(K, dim)
                K_out[token, head, dim] = rotated_elem

                if slot >= fx.Int32(0):
                    cache_block = slot // fx.Int32(block_size)
                    cache_offset = slot % fx.Int32(block_size)
                    KeyCache[cache_block, cache_offset, head, dim] = rotated_elem
                    ValueCache[cache_block, cache_offset, head, dim] = V[token, head, dim]

    @flyc.jit
    def launch_fused_rope_cache(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        Positions: fx.Tensor,
        CosCache: fx.Tensor,
        SinCache: fx.Tensor,
        SlotMapping: fx.Tensor,
        KeyCache: fx.Tensor,
        ValueCache: fx.Tensor,
        Q_out: fx.Tensor,
        K_out: fx.Tensor,
        num_tokens: fx.Int32,
        KScale: fx.Tensor,
        VScale: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        fused_qk_rope_reshape_and_cache(
            Q,
            K,
            V,
            Positions,
            CosCache,
            SinCache,
            SlotMapping,
            KeyCache,
            ValueCache,
            Q_out,
            K_out,
            KScale,
            VScale,
        ).launch(
            grid=(max_heads, num_tokens, 1),
            block=(WARP_SIZE, 1, 1),
            stream=stream,
        )

    return launch_fused_rope_cache


__all__ = ["build_fused_rope_cache_module"]
