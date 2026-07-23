# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Flash-decoding split heuristics for the MMA KV-cache decode kernel.

Ports library ``FlashDecoding_GQA::getGroupLengthAndKWarps`` (non-heuristic,
``seqs == 1``) from ``library_page_attention_get_groups_num.h``.
"""

from __future__ import annotations

import functools
import math

_MAX_FLYDSL_SPLITS = 32
_V5_CHUNK = 16
_V5_TARGET_CTAS = 128


def _is_split_eligible(num_n_blocks: int, num_splits: int) -> bool:
    if num_splits == 1:
        return True
    ceildiv = lambda a, b: (a + b - 1) // b
    return ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1)


def _calculate_max_splits(num_n_blocks: int, mr_max_blocks: int, batch_nheads_mblocks: int) -> int:
    """Port of library ``CalculateMaxSplits``."""
    max_splits = min(128, mr_max_blocks, num_n_blocks)
    max_efficiency = 0.0
    efficiency: list[float] = []
    for num_splits in range(1, max_splits + 1):
        if not _is_split_eligible(num_n_blocks, num_splits):
            efficiency.append(0.0)
            continue
        n_warps = batch_nheads_mblocks * num_splits / mr_max_blocks
        eff = n_warps / math.ceil(n_warps)
        max_efficiency = max(max_efficiency, eff)
        efficiency.append(eff)
        if eff == 1.0:
            for cur in range(1, num_splits + 1):
                if not _is_split_eligible(num_n_blocks, cur):
                    continue
                if efficiency[cur - 1] >= 0.85 * max_efficiency:
                    return cur
    for cur in range(1, max_splits + 1):
        if not _is_split_eligible(num_n_blocks, cur):
            continue
        if efficiency[cur - 1] >= 0.85 * max_efficiency:
            return cur
    return 1


def _compute_mma_decode_num_splits(
    *,
    batch_size: int,
    seqlen_q: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_seqlen_k: int,
    block_n: int,
) -> int:
    if head_dim != 128:
        return 1
    if block_n <= 0 or max_seqlen_k % block_n != 0:
        return 1

    least_block_size = 32
    mr_max_cu_warps = 16  # head_size=128 row in MR_MAX_CU_WARPS
    nh_warps = 1
    mr_max_blocks = mr_max_cu_warps * 4 * 4 // nh_warps

    share_heads = num_heads // num_kv_heads
    q_warp_heads = 2 if share_heads > 1 else 1
    q_warps = 1 if share_heads == 1 else (share_heads + q_warp_heads - 1) // q_warp_heads
    q_blocks = (share_heads + q_warps * q_warp_heads - 1) // (q_warps * q_warp_heads)

    num_n_blocks = (max_seqlen_k + least_block_size - 1) // least_block_size
    batch_nheads_mblocks = batch_size * seqlen_q * num_kv_heads * q_blocks

    if batch_nheads_mblocks >= 0.8 * mr_max_blocks:
        splits = 1
    else:
        splits = _calculate_max_splits(num_n_blocks, mr_max_blocks, batch_nheads_mblocks)

    least_task_size = 32
    # library can profitably use more groups because its dispatch subsequently
    # folds small group counts into K-warps and uses a specialized group
    # reduction.  FlyDSL currently launches a separate 256-thread reduction
    # kernel whose cost grows with every split; measurements on MR-V100 show
    # that 64 groups regress versus 32 for long-context decode.
    splits = min(
        _MAX_FLYDSL_SPLITS,
        splits,
        (max_seqlen_k + least_task_size - 1) // least_task_size,
    )

    num_kv_tiles = max_seqlen_k // block_n
    splits = min(splits, num_kv_tiles)
    while splits > 1 and num_kv_tiles % splits != 0:
        splits -= 1
    return max(1, splits)


@functools.lru_cache(maxsize=None)
def compute_mma_decode_num_splits(
    *,
    batch_size: int,
    seqlen_q: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_seqlen_k: int,
    block_n: int,
) -> int:
    """Return flash-decoding group count for ``grid=(splits, batch, kv_heads)``."""
    return _compute_mma_decode_num_splits(
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_seqlen_k=max_seqlen_k,
        block_n=block_n,
    )


@functools.lru_cache(maxsize=None)
def compute_v5_decode_config(
    *,
    batch_size: int,
    seqlen_q: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_seqlen_k: int,
) -> tuple[int, int, int]:
    """Return library V5 ``(groups, fix_length, k_warps)``.

    Unlike the legacy FlyDSL MMA path, V5 permits a shorter final group and
    therefore does not decrement the occupancy-selected group count until it
    evenly divides the number of sequence tiles.
    """
    if head_dim != 128 or max_seqlen_k <= 0 or max_seqlen_k % _V5_CHUNK:
        return 1, max_seqlen_k, 1
    if num_heads % num_kv_heads:
        return 1, max_seqlen_k, 1

    share_heads = num_heads // num_kv_heads
    q_blocks = (share_heads + 15) // 16
    mr_max_blocks = 16 * 4 * 4
    num_n_blocks = (max_seqlen_k + 31) // 32
    base_blocks = batch_size * seqlen_q * num_kv_heads * q_blocks
    groups = (
        1 if base_blocks >= 0.8 * mr_max_blocks else _calculate_max_splits(num_n_blocks, mr_max_blocks, base_blocks)
    )
    groups = max(1, min(128, groups, (max_seqlen_k + 31) // 32))
    # The library occupancy model selects 114 groups for B1/Hq16/Hkv2/Sk32K,
    # but FlyDSL's separate group-reduce makes that slower than filling the
    # device once with 128 one-warp CTAs (64 groups * 2 KV heads).
    groups = min(groups, max(1, (_V5_TARGET_CTAS + base_blocks - 1) // base_blocks))
    fix_length = ((max_seqlen_k + groups - 1) // groups + _V5_CHUNK - 1) // _V5_CHUNK * _V5_CHUNK
    return groups, fix_length, 1
