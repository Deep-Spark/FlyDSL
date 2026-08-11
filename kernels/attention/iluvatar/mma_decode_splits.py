# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Flash-decoding split heuristics for the MMA KV-cache decode kernel.

Group / split counts follow the usual flash-decoding occupancy heuristic
(cap splits so CTAs fill the device without over-fragmenting the KV stream).
"""

from __future__ import annotations

import functools
import math

_MAX_FLYDSL_SPLITS = 32
_PIPELINED_PAGE_SIZE = 16
_PIPELINED_TARGET_CTAS = 128


def _is_split_eligible(num_n_blocks: int, num_splits: int) -> bool:
    if num_splits == 1:
        return True
    ceildiv = lambda a, b: (a + b - 1) // b
    return ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1)


def _calculate_max_splits(num_n_blocks: int, mr_max_blocks: int, batch_nheads_mblocks: int) -> int:
    """Pick a split count that keeps CTA occupancy high without empty splits."""
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
    # Cap splits: each extra group needs a separate reduce, so too many groups
    # hurt long-context decode even when occupancy formulas suggest more.
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
def compute_pipelined_mma_decode_config(
    *,
    batch_size: int,
    seqlen_q: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_seqlen_k: int,
) -> tuple[int, int, int]:
    """Return pipelined HND ``(groups, fix_length, k_warps)``.

    Unlike the baseline MMA path, this configuration permits a shorter final
    group and does not require the group count to evenly divide the tiles.
    """
    if head_dim != 128 or max_seqlen_k <= 0 or max_seqlen_k % _PIPELINED_PAGE_SIZE:
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
    # Prefer filling the device once (~128 one-warp CTAs) over a larger split
    # count that then pays more in the separate group-reduce kernel.
    groups = min(groups, max(1, (_PIPELINED_TARGET_CTAS + base_blocks - 1) // base_blocks))
    fix_length = (
        ((max_seqlen_k + groups - 1) // groups + _PIPELINED_PAGE_SIZE - 1)
        // _PIPELINED_PAGE_SIZE
        * _PIPELINED_PAGE_SIZE
    )
    return groups, fix_length, 1


@functools.lru_cache(maxsize=None)
def compute_qwen_simt_decode_config(
    *,
    batch_size: int,
    seqlen_q: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_seqlen_k: int,
) -> tuple[int, int]:
    """Return ``(groups, k_warps)`` for Qwen SIMT decode."""
    if (
        batch_size == 1
        and seqlen_q == 1
        and num_heads == 16
        and num_kv_heads == 4
        and head_dim == 256
        and max_seqlen_k > 0
    ):
        # Schedule D256 GQA4 in 16-token chunks. Each CTA owns two query heads;
        # two sibling CTAs cover the four heads sharing one KV head.
        logical_pages = (max_seqlen_k + 15) // 16
        # D256 needs more than two CTAs to fill the device even at 128 tokens;
        # very fine splitting pays extra workspace/reduce traffic on longer K.
        if logical_pages <= 8:
            groups = logical_pages
        elif logical_pages <= 68:
            groups = 16
        else:
            groups = 32
        return max(2, min(_MAX_FLYDSL_SPLITS, groups, logical_pages)), 1

    if (
        batch_size != 1
        or seqlen_q != 1
        or num_heads != 16
        or num_kv_heads != 8
        or head_dim != 128
        or max_seqlen_k <= 0
        or max_seqlen_k % _PIPELINED_PAGE_SIZE
    ):
        return 1, 1
    pages = (max_seqlen_k + _PIPELINED_PAGE_SIZE - 1) // _PIPELINED_PAGE_SIZE
    if pages <= 2:
        return 1, 1

    # Short contexts keep the CTA-local K-warp tree. For long contexts, split
    # K into 32-token tasks and launch one K warp per split CTA. Do not inflate
    # the split count to a fixed CTA target or require eight pages per split --
    # both force a 16-warp CTA just past 512 tokens, where barrier/LDS merge
    # cost creates a latency cliff.
    k_warps = min(16, 1 << ((pages - 1).bit_length() - 1))
    if pages <= 32:
        # One or two pages per warp do not amortize the two CTA barriers and
        # LDS merge. Four warps keep enough parallelism while cutting that
        # fixed short-context reduction cost.
        return 1, min(k_warps, 4)
    q_blocks = 1
    base_blocks = batch_size * seqlen_q * num_kv_heads * q_blocks
    mr_max_blocks = 16 * 4 * 4
    num_n_blocks = (max_seqlen_k + 31) // 32
    groups = (
        1 if base_blocks >= 0.8 * mr_max_blocks else _calculate_max_splits(num_n_blocks, mr_max_blocks, base_blocks)
    )
    return max(1, min(_MAX_FLYDSL_SPLITS, groups, num_n_blocks)), 1
