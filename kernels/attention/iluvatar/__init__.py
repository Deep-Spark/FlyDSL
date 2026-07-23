# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar-specific attention kernels."""

from .fused_rope_cache_kernel import build_fused_rope_cache_module

__all__ = ["build_fused_rope_cache_module"]
