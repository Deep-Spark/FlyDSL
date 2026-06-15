# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Shared ivcore11 MR (TCU/SME) hardware constants for Iluvatar GEMM kernels."""

# Warp-collective TCU/SME width on ivcore11 (BI-V150 / MR-100 class).
WARP_SIZE = 64

# TCU MMA atom shape (f16/bf16 and i8 paths use M=N=16; K=16 f16, K=32 i8).
ATOM_M = 16
ATOM_N = 16
ATOM_K = 16

# TCU lane grid: 64 lanes → 4 rows × 16 cols (same as ATOM_M / ATOM_N).
TCU_LANE_COLS = 16

# SME G2S brick: 16 rows x 512 bits/row (= 32 f16 or 64 i8 per row).
SME_BITS_PER_ROW = 512
SMEM_ROWS = 16
SMEM_F16_PER_ROW = SME_BITS_PER_ROW // 16

# Per-CTA dynamic shared memory cap on ivcore11.
DEFAULT_SMEM_CAP_BYTES = 131072

# G2S global layout tags for logical A(m,k) / B(n,k) operands.
# Two letters: A major (n=NoTrans/row SME, t=Trans/col SME), B major (same).
MAJOR_PATTERN_NN = "nn"
MAJOR_PATTERN_TN = "tn"
MAJOR_PATTERN_NT = "nt"
MAJOR_PATTERN_TT = "tt"
MAJOR_PATTERN_CHOICES = (
    MAJOR_PATTERN_NN,
    MAJOR_PATTERN_TN,
    MAJOR_PATTERN_NT,
    MAJOR_PATTERN_TT,
)
DEFAULT_MAJOR_PATTERN = MAJOR_PATTERN_NT
PATTERN_ID = {name: idx for idx, name in enumerate(MAJOR_PATTERN_CHOICES)}


def major_pattern_id(major_pattern: str) -> int:
    return PATTERN_ID[major_pattern]


def sme_values_per_row(elem_bits: int) -> int:
    return SME_BITS_PER_ROW // elem_bits


def mr_swizzle_cta_shape(
    warps_m: int,
    warps_n: int,
    k_rep: int,
    *,
    warp_atoms_m: int,
    warp_atoms_n: int,
    atom_k: int,
    elem_bytes: int,
    stages: int = 2,
) -> tuple[int, int, int, int, int]:
    """Swizzle-mode CTA geometry shared by the f16/i8 MR GEMM kernels.

    Returns ``(bm, bn, bk, threads, pipeline_smem_bytes)``. ``pipeline_smem_bytes``
    is the double-buffered G2S staging size; the i8 packed epilogue may need
    ``max(pipeline_smem, bm * bn)`` and applies that at its call site.
    """
    warp_m = ATOM_M * warp_atoms_m
    warp_n = ATOM_N * warp_atoms_n
    bm = warp_m * warps_m
    bn = warp_n * warps_n
    bk = atom_k * k_rep
    threads = warps_m * warps_n * WARP_SIZE
    smem_bytes = (bm + bn) * bk * elem_bytes * stages
    return bm, bn, bk, threads, smem_bytes


def pattern_sme_atom_counts(
    pattern_id: int,
    bm: int,
    bn: int,
    bk: int,
    *,
    values_per_sme_row: int = SMEM_F16_PER_ROW,
) -> tuple[int, int, int, int]:
    """Return (a_atoms_total, b_atoms_total, a_smem_k_bricks, b_smem_k_bricks)."""
    if pattern_id in (0, 2):
        a_atoms_total = (bm // SMEM_ROWS) * (bk // values_per_sme_row)
        a_smem_k_bricks = bk // values_per_sme_row
    else:
        a_atoms_total = (bm // values_per_sme_row) * (bk // SMEM_ROWS)
        a_smem_k_bricks = bk // SMEM_ROWS
    if pattern_id in (0, 1):
        b_atoms_total = (bn // values_per_sme_row) * (bk // SMEM_ROWS)
        b_smem_k_bricks = bk // SMEM_ROWS
    else:
        b_atoms_total = (bn // SMEM_ROWS) * (bk // values_per_sme_row)
        b_smem_k_bricks = bk // values_per_sme_row
    return a_atoms_total, b_atoms_total, a_smem_k_bricks, b_smem_k_bricks


__all__ = [
    "ATOM_K",
    "ATOM_M",
    "ATOM_N",
    "DEFAULT_MAJOR_PATTERN",
    "DEFAULT_SMEM_CAP_BYTES",
    "MAJOR_PATTERN_CHOICES",
    "MAJOR_PATTERN_NN",
    "MAJOR_PATTERN_NT",
    "MAJOR_PATTERN_TN",
    "MAJOR_PATTERN_TT",
    "PATTERN_ID",
    "SME_BITS_PER_ROW",
    "SMEM_F16_PER_ROW",
    "SMEM_ROWS",
    "TCU_LANE_COLS",
    "WARP_SIZE",
    "major_pattern_id",
    "mr_swizzle_cta_shape",
    "pattern_sme_atom_counts",
    "sme_values_per_row",
]
