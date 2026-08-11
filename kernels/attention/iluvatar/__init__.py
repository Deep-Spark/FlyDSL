# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar-specific attention kernels."""

from .block_mask import (
    FlexBlockMask,
    PackedPagedBlockMask,
    PackedVarlenBlockMask,
    create_block_mask,
    create_block_masks_paged,
    create_block_masks_varlen,
    pack_block_masks_paged,
    pack_block_masks_varlen,
)
from .block_mask import FlexBlockMask, create_block_mask
from .flash_attn_kvcache import flash_attn_with_kvcache
from .flash_attn_varlen import flash_attn_varlen_func
from .flex_attention import compile_iluvatar_flex_attention
from .flex_attention_bwd import compile_iluvatar_flex_attention_bwd
from .flex_attn_interface import autotune_iluvatar_flex_attention_tile, flydsl_flex_attn_func
from .fused_rope_cache_kernel import build_fused_rope_cache_module

__all__ = [
    "build_fused_rope_cache_module",
    "compile_iluvatar_flex_attention",
    "compile_iluvatar_flex_attention_bwd",
    "create_block_mask",
    "create_block_masks_varlen",
    "create_block_masks_paged",
    "pack_block_masks_varlen",
    "pack_block_masks_paged",
    "FlexBlockMask",
    "PackedVarlenBlockMask",
    "PackedPagedBlockMask",
    "flydsl_flex_attn_func",
    "flash_attn_varlen_func",
    "flash_attn_with_kvcache",
    "autotune_iluvatar_flex_attention_tile",
]
