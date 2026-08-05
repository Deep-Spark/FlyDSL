# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""CQ (ivcore30) CQMma (M, N) tile presets.

``mma_tile`` selects the atom (M, N); K is dtype-dependent and unchanged by
FeatureLongMtx (f16/bf16 K=16, i8 K=32).
"""

from typing import NamedTuple

from kernels.gemm.iluvatar.common import ATOM_M, ATOM_N

# Legal ``mma_tile`` strings (FeatureLongMtx MN enlargements + base).
MMA_TILE_BASE = "base"
MMA_TILE_PRESETS = {
    MMA_TILE_BASE: (ATOM_M, ATOM_N),
    "16x16": (ATOM_M, ATOM_N),
    "32x32": (32, 32),
    "16x64": (16, 64),
    "64x16": (64, 16),
}


class CqMmaTile(NamedTuple):
    """Resolved CQ MMA atom extents for one ``mma_tile`` + dtype family."""

    atom_m: int
    atom_n: int
    atom_k: int
    mma_tile: str


def parse_mma_tile(mma_tile: str, *, atom_k: int) -> CqMmaTile:
    """Map ``mma_tile`` to CQMma (M, N, K).

    Args:
        mma_tile: ``\"base\"`` / ``\"16x16\"`` / ``\"32x32\"`` / ``\"16x64\"`` / ``\"64x16\"``.
        atom_k: K extent for the dtype (``ATOM_K_B16`` or ``ATOM_K_B8``).
    """
    key = str(mma_tile).strip().lower()
    if key not in MMA_TILE_PRESETS:
        raise ValueError(f"unsupported mma_tile={mma_tile!r}; expected one of {sorted(MMA_TILE_PRESETS)}")
    atom_m, atom_n = MMA_TILE_PRESETS[key]
    return CqMmaTile(atom_m=atom_m, atom_n=atom_n, atom_k=int(atom_k), mma_tile=key)
