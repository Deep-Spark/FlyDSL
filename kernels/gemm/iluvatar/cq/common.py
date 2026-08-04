# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""CQ (ivcore30) MMA tile helpers + SmexMtx G2S/S2R geometry for HGEMM.

``mma_tile`` selects the CQMma (M, N) enlargement for the fragment bring-up
path (long-mtx is out of scope for the pipelined HGEMM). Pipelined HGEMM uses
base 16x16x16 f16/bf16 atoms with SmexMtx bricks (16 rows x 64 B/row).
"""

from typing import NamedTuple

import flydsl.expr as fx
from kernels.gemm.iluvatar.common import (
    ATOM_K_B16,
    ATOM_M,
    ATOM_N,
    SMEM_B16_PER_ROW,
    SMEM_ROWS,
)

# Legal ``mma_tile`` strings (FeatureLongMtx MN enlargements + base).
MMA_TILE_BASE = "base"
MMA_TILE_PRESETS = {
    MMA_TILE_BASE: (ATOM_M, ATOM_N),
    "16x16": (ATOM_M, ATOM_N),
    "32x32": (32, 32),
    "16x64": (16, 64),
    "64x16": (64, 16),
}

# Per-CTA shared memory cap used for CQ HGEMM compile-time checks.
DEFAULT_SMEM_CAP_BYTES = 131072

# SmexMtx loadn16 G2S: 16 rows x 64 B/row. For f16/bf16 that is 32 elems/row.
SMEX_G2S_BYTES_PER_ROW = 64


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


class CqOperandGeom(NamedTuple):
    """Compile-time operand geometry for CQ SmexMtx G2S / loadn S2R (b16 HGEMM).

    ``values_per_sme_row`` matches MR f16 (32): one ``smex.loadn.16x1b64.mtx``
    brick is ``SMEM_ROWS x vpr`` elements. ``atom_*`` are base CQMma extents.
    """

    elem_bits: int
    atom_m: int
    atom_n: int
    atom_k: int
    values_per_sme_row: int

    @staticmethod
    def b16() -> "CqOperandGeom":
        return CqOperandGeom(16, ATOM_M, ATOM_N, ATOM_K_B16, SMEM_B16_PER_ROW)

    @property
    def cta_chunk_elems(self) -> int:
        """Elements per SmexMtx G2S brick (16 rows x vpr)."""
        return SMEM_ROWS * self.values_per_sme_row

    @property
    def cta_chunk_bytes(self) -> int:
        return self.cta_chunk_elems * (self.elem_bits // 8)

    @property
    def sme_row_k_slices(self) -> int:
        """K slices per SmexMtx brick row (vpr // atom_k); f16 -> 2."""
        return self.values_per_sme_row // self.atom_k


CQ_GEMM_GEOM = CqOperandGeom.b16()


def cq_stage_smem_ab(smem_base, stage_base, a_stage_elems):
    """Per-stage shared A/B base pointers for the CQ HGEMM pipeline."""
    smem_a = fx.add_offset(smem_base, fx.make_int_tuple(stage_base))
    smem_b = fx.add_offset(smem_a, fx.make_int_tuple(fx.Int32(a_stage_elems)))
    return smem_a, smem_b


def cq_smex_mtx_tile_ptr_info(*, role: str, warp_mn_idx: int, warp_k_idx: int, block_k_loop_cnt: int):
    """ixmma Loadn16 B16 shared-tile bp + EmPart (Row A / Col B).

    Mirrors ``computeLoadn16{Row,Col}B16`` in ixcc ``MtxLoadUtils``:
    ``shared_idx = warp_mn * block_k + warp_k``, then
    ``tile_base_bytes = (shared_idx // 2) * 1024``, ``em_part = (shared_idx % 2) * 2``.
    """
    if role not in ("A", "B"):
        raise ValueError(f"role must be 'A' or 'B', got {role!r}")
    shared_idx = int(warp_mn_idx) * int(block_k_loop_cnt) + int(warp_k_idx)
    tile_base_bytes = (shared_idx // 2) * 1024
    em_part = (shared_idx % 2) * 2
    return tile_base_bytes, em_part


__all__ = [
    "ATOM_K_B16",
    "ATOM_M",
    "ATOM_N",
    "CQ_GEMM_GEOM",
    "CqMmaTile",
    "CqOperandGeom",
    "DEFAULT_SMEM_CAP_BYTES",
    "MMA_TILE_BASE",
    "MMA_TILE_PRESETS",
    "SMEM_ROWS",
    "SMEX_G2S_BYTES_PER_ROW",
    "cq_smex_mtx_tile_ptr_info",
    "cq_stage_smem_ab",
    "parse_mma_tile",
]
