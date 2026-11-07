# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Host-side paged-KV layout names and metadata validation."""

import torch

HND = "HND"
NHD = "NHD"
SUPPORTED_KV_CACHE_LAYOUTS = (HND, NHD)


def validate_kv_cache_layout(kv_cache_layout: str) -> bool:
    if kv_cache_layout not in SUPPORTED_KV_CACHE_LAYOUTS:
        raise ValueError("kv_cache_layout must be 'HND' or 'NHD'")
    return kv_cache_layout == NHD


def resolve_kv_cache_layout(*, default_nhd: bool, force_nhd) -> bool:
    """Resolve the optional API override to the internal NHD-layout flag."""
    return default_nhd if force_nhd is None else bool(force_nhd)


def validate_block_table_shape(block_table, *, batch: int, device) -> None:
    if block_table.dtype != torch.int32 or not block_table.is_cuda:
        raise ValueError("block_table must be a CUDA int32 tensor")
    if (
        block_table.ndim != 2
        or block_table.shape[0] != batch
        or block_table.shape[1] <= 0
        or block_table.stride(-1) != 1
        or block_table.device != device
    ):
        raise ValueError(
            f"block_table must be contiguous [batch, max_blocks] on the same device as q; "
            f"got shape={tuple(block_table.shape)}, device={block_table.device}, "
            f"batch={batch}, q.device={device}"
        )
