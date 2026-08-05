# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""CQ GEMM package: shared tile helpers + CQMma fragment smoke.

Full pipelined ``hgemm`` / ``igemm`` are not here yet (need async copy / load-matrix).
"""

from .common import MMA_TILE_BASE, MMA_TILE_PRESETS, CqMmaTile, parse_mma_tile
from .mma_fragment_smoke import (
    compile_cq_mma_fragment_smoke_f16,
    compile_cq_mma_fragment_smoke_s8,
)

__all__ = [
    "MMA_TILE_BASE",
    "MMA_TILE_PRESETS",
    "CqMmaTile",
    "parse_mma_tile",
    "compile_cq_mma_fragment_smoke_f16",
    "compile_cq_mma_fragment_smoke_s8",
]
