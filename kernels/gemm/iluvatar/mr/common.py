# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MR (TCU/SME) hardware constants and CTA smem grid helpers for Iluvatar MR GEMM.

Index naming uses four prefixes (do not mix):

  cta_*     G2S smem chunk grid (MrCtaSmemGrid, cta_lin, cta_m/n/k)
  mma_*     warp loop indices (mma_m, mma_n, mma_k)
  sme_row_* S2R sub-slice inside one 512-bit SME row (MrOperandGeom)
  phys_*    epilogue GMEM byte/element offsets

atom_m/n/k — size (elements) of one MRMma / fx.gemm tile (f16: 16). Not a loop var.
mma_m/n/k  — 0-based index of which atom_* tile the warp uses within the CTA bk step.
values_per_sme_row (vpr) — elements per 512-bit SME row; f16 32, i8 64, f32 16.

G2S chunk grid (cta_*, mr_gemm_g2s_issue_*_warp):
  One G2S issue moves cta_chunk_elems = 16 smem rows x vpr into smem.
  Count chunks on the CTA A(m,k) or B(n,k) tile (sme_atom_counts / MrCtaSmemGrid):
    a_atoms_total = (bm/16)*(bk/vpr) for k-major A; B symmetric on bn.
  Split evenly across warps: a_per_warp = a_atoms_total // num_warps (must divide).
  cta_lin is the linear chunk id 0 .. atoms_total-1, assigned per warp:
    cta_lin = warp_id * per_warp + t   (t = 0 .. per_warp-1; A and B each have own range)
  Decode coords from cta_lin (fast axis = %, slow axis = //), e.g. A k-major tn:
    cta_k = cta_lin % cta_a_k_cnt_k_major;  cta_m = cta_lin // cta_a_k_cnt_k_major
  Smem offset: cta_lin * cta_chunk_elems within the stage's smem_a / smem_b buffer.

  f16 tn, bm=64, bk=32: vpr=32 -> 4 chunks (4x1 along M); warp0..3 get cta_lin 0..3
  when num_warps=4 and a_per_warp=1; all have cta_k=0, cta_m=cta_lin.

SME row sub-slicing (sme_row_*, S2R only):
  Several atom_*-wide operand slices may sit side-by-side along one SME row; S2R picks
  the slice for the current mma_k (or mma_m / mma_n). Not cta_* — a G2S chunk spans
  16 smem rows; sme_row_* slices within a single row.

  sme_row_k_slices  = vpr // atom_k  when K is along the row (k-major)
  sme_row_a_m_atoms = vpr // atom_m  when A is mn-major
  sme_row_b_n_atoms = vpr // atom_n  when B is mn-major

  f16 tn (k-major A/B): vpr=32, atom_k=16 -> sme_row_k_slices=2;
  mma_k=0/1 pick two atom_k-wide K slabs; sme_row_k_sub = mma_k % sme_row_k_slices.

  f16 nt (mn-major A/B): sme_row_a_m_atoms=2, sme_row_b_n_atoms=2; M/N run along the
  row instead of K. A S2R: sme_row_m_sub = cta_m_atom % sme_row_a_m_atoms (from mma_m);
  B S2R: sme_row_n_sub = cta_n_atom % sme_row_b_n_atoms (from mma_n); mma_k is not
  split along the row on nt.

Common pitfalls:
  major_pattern ("tn", etc.) — which operand is k-major on logical A(m,k)/B(n,k);
    not whole-matrix row/column major. See kernels.gemm.iluvatar.common.GemmLayout.
  mma_k — K step within one CTA bk tile; not cta_k (G2S chunk coord) and not the
    outer loop over full problem K.
  MrCtaSmemGrid — cta_lin decode divisors; not a CuTe Layout.

Three unrelated "atom" sizes:
  atom_m/n/k     — MRMma compute tile (MmaAtom)
  S2R copy atom  — UniversalCopy via make_tiled_copy_A/B
  G2S smem chunk — one MRAsyncCp issue (cta_lin); 16 smem rows x vpr elems
"""

from typing import NamedTuple

import flydsl.expr as fx

from kernels.gemm.iluvatar.common import GemmLayout

# Per-CTA dynamic shared memory cap on ivcore11 (BI-V150 / MR-50 / MR-100 class).
DEFAULT_SMEM_CAP_BYTES = 131072

# TCU MMA atom shape: M=N=16 for all dtypes; K per MRMma instruction depends on element width.
ATOM_M = 16
ATOM_N = 16
ATOM_K_B8 = 32  # i8 MRMma K extent
ATOM_K_B16 = 16  # f16/bf16 MRMma K extent (one mma_k slice in the f16 pipeline)
ATOM_K_B32 = 16  # f32 MRMma K extent (one mma_k slice in the b32 pipeline)

# TCU lane grid: 64 lanes -> 4 rows x 16 cols (same as ATOM_M / ATOM_N).
TCU_LANE_COLS = 16

# SME G2S chunk: 16 rows x 512 bits/row (= 32 f16 or 64 i8 per row).
SME_BITS_PER_ROW = 512
SMEM_ROWS = 16
SMEM_B8_PER_ROW = SME_BITS_PER_ROW // 8
SMEM_B16_PER_ROW = SME_BITS_PER_ROW // 16
SMEM_B32_PER_ROW = SME_BITS_PER_ROW // 32


class MrOperandGeom(NamedTuple):
    """Compile-time operand geometry for MR GEMM G2S/S2R (b8 / b16 / b32).

    values_per_sme_row field; abbreviated vpr in module doc and formulas below.

    sme_row_* counts how many atom_*-wide slices fit on one SME row during S2R
    (vpr // atom_*). Axis depends on major mode:

      k-major operand  -> sme_row_k_slices  (= vpr // atom_k)
      A mn-major       -> sme_row_a_m_atoms (= vpr // atom_m)
      B mn-major       -> sme_row_b_n_atoms (= vpr // atom_n)

    major_pattern -> which property S2R uses on A / B:
      tn: k/k   -> sme_row_k_slices on both
      nt: mn/mn -> sme_row_a_m_atoms, sme_row_b_n_atoms
      nn: mn/k  -> sme_row_a_m_atoms; sme_row_k_slices on B
      tt: k/mn  -> sme_row_k_slices on A; sme_row_b_n_atoms on B

    sme_row_* — S2R only (in-row sub-slices from mma_k / mma_m / mma_n).

    G2S reads only vpr (values_per_sme_row) and cta_chunk_elems (= 16 * vpr) via
    mr_cta_smem_grid; smem offset is cta_lin * cta_chunk_elems within smem_a/smem_b.
    atom_m/n/k are not used in G2S address math (MMA / S2R tile shape only).

    Used in mr_gemm_s2r_*_tile for sme_row_* and atom_*; G2S issue passes geom but
    does not read sme_row_* or atom_* for addressing.
    """

    elem_bits: int
    atom_m: int
    atom_n: int
    atom_k: int
    values_per_sme_row: int

    @staticmethod
    def b8() -> "MrOperandGeom":
        return MrOperandGeom(8, ATOM_M, ATOM_N, ATOM_K_B8, SMEM_B8_PER_ROW)

    @staticmethod
    def b16() -> "MrOperandGeom":
        return MrOperandGeom(16, ATOM_M, ATOM_N, ATOM_K_B16, SMEM_B16_PER_ROW)

    @staticmethod
    def b32() -> "MrOperandGeom":
        return MrOperandGeom(32, ATOM_M, ATOM_N, ATOM_K_B32, SMEM_B32_PER_ROW)

    @staticmethod
    def from_elem_bits(elem_bits: int) -> "MrOperandGeom":
        if elem_bits == 8:
            return MrOperandGeom.b8()
        if elem_bits == 16:
            return MrOperandGeom.b16()
        if elem_bits == 32:
            return MrOperandGeom.b32()
        raise ValueError(f"unsupported MR operand elem_bits: {elem_bits}")

    @property
    def sme_row_k_slices(self) -> int:
        """k-major S2R: K slices per SME row (vpr // atom_k)."""
        return self.values_per_sme_row // self.atom_k

    @property
    def sme_row_a_m_atoms(self) -> int:
        """A mn-major S2R: M slices per SME row (vpr // atom_m)."""
        return self.values_per_sme_row // self.atom_m

    @property
    def sme_row_b_n_atoms(self) -> int:
        """B mn-major S2R: N slices per SME row (vpr // atom_n)."""
        return self.values_per_sme_row // self.atom_n

    @property
    def cta_chunk_elems(self) -> int:
        """Elements per G2S smem chunk (16 smem rows x vpr)."""
        return SMEM_ROWS * self.values_per_sme_row


class MrCtaSmemGrid(NamedTuple):
    """cta_* divisors for G2S cta_lin decode on one bm x bn x bk K-step.

    Built by mr_cta_smem_grid in iluvatar_mr_operand_copy. One G2S chunk is 16 smem
    rows x vpr elems (cta_chunk_elems). cta_lin is the linear chunk id over the CTA
    bm x bk / bn x bk grid; G2S uses one field as cta_lin % (fast) and cta_lin // (slow).

    Not the same as sme_row_* in MrOperandGeom (in-row S2R sub-slices for mma_k).

    Fields (vpr = geom.values_per_sme_row; f16: 32, i8: 64):
      cta_chunk_elems — smem element stride per chunk
      cta_a_k_cnt_k_major — A k-major K count (G2S %)
      cta_a_k_cnt — A K-span (S2R A; bk/vpr or bk/16 by major)
      cta_b_k_cnt — B k-major K count
      cta_b_n_cnt — B mn-major N count (G2S %)

    See mr_gemm_g2s_issue_a_warp / mr_gemm_g2s_issue_b_warp for formulas.
    """

    cta_a_k_cnt: int
    cta_b_k_cnt: int
    cta_a_k_cnt_k_major: int
    cta_b_n_cnt: int
    cta_chunk_elems: int


def sme_values_per_row(elem_bits: int) -> int:
    return SME_BITS_PER_ROW // elem_bits


def sme_atom_counts(
    layout: GemmLayout,
    bm: int,
    bn: int,
    bk: int,
    *,
    values_per_sme_row: int = SMEM_B16_PER_ROW,
) -> tuple[int, int, int, int]:
    """G2S chunk counts for a bm x bn x bk CTA tile.

    Returns (a_atoms_total, b_atoms_total, cta_a_k_cnt, cta_b_k_cnt).
    """
    if layout.a_k_major:
        cta_a_k_cnt = bk // values_per_sme_row
        a_atoms_total = (bm // SMEM_ROWS) * cta_a_k_cnt
    else:
        cta_a_k_cnt = bk // SMEM_ROWS
        a_atoms_total = (bm // values_per_sme_row) * cta_a_k_cnt

    if layout.b_k_major:
        cta_b_k_cnt = bk // values_per_sme_row
        b_atoms_total = (bn // SMEM_ROWS) * cta_b_k_cnt
    else:
        cta_b_k_cnt = bk // SMEM_ROWS
        b_atoms_total = (bn // values_per_sme_row) * cta_b_k_cnt

    return a_atoms_total, b_atoms_total, cta_a_k_cnt, cta_b_k_cnt


def mr_stage_smem_ab(smem_base, stage_base, a_stage_elems):
    """Per-stage shared A/B base pointers for the GEMM pipeline.

    A starts at ``stage_base``; B follows immediately after A's tile at
    ``stage_base + a_stage_elems``. ``smem_base`` is the element-typed shared base
    pointer, so all offsets are counted in elements (``a_stage_elems`` = bm * bk
    for a bm x bk A tile). Shared by the f16 (hgemm) and i8 (igemm) pipelines.
    """
    smem_a = fx.add_offset(smem_base, fx.make_int_tuple(stage_base))
    smem_b = fx.add_offset(smem_a, fx.make_int_tuple(fx.Int32(a_stage_elems)))
    return smem_a, smem_b


# Default operand geometry for production MR f16 HGEMM kernels/tests.
MR_GEMM_GEOM = MrOperandGeom.b16()


__all__ = [
    "ATOM_K_B32",
    "ATOM_K_B16",
    "ATOM_K_B8",
    "ATOM_M",
    "ATOM_N",
    "DEFAULT_SMEM_CAP_BYTES",
    "MrCtaSmemGrid",
    "MrOperandGeom",
    "MR_GEMM_GEOM",
    "SME_BITS_PER_ROW",
    "SMEM_B16_PER_ROW",
    "SMEM_B32_PER_ROW",
    "SMEM_B8_PER_ROW",
    "SMEM_ROWS",
    "TCU_LANE_COLS",
    "mr_stage_smem_ab",
    "sme_atom_counts",
    "sme_values_per_row",
]
