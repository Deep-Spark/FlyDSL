# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar norm kernels."""

from .rmsnorm_kernel import compile_iluvatar_rmsnorm, compile_iluvatar_rmsnorm_dynamicquant

__all__ = ["compile_iluvatar_rmsnorm", "compile_iluvatar_rmsnorm_dynamicquant"]
