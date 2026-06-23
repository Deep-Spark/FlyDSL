# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Shared helpers for Iluvatar MR HGEMM staged device tests."""

from kernels.iluvatar_common import (
    logical_strides,
    parse_major_pattern,
    remap_gemm_tensors,
    WARP_SIZE,
)
from kernels.iluvatar_mr_common import (
    MR_GEMM_GEOM,
    MrOperandGeom,
    SMEM_ROWS,
    sme_atom_counts,
)

# Default staged CTA tile (matches production swizzle-1024 preset at k_atoms=4).
STAGED_BRICK_M = 256
STAGED_BRICK_N = 256
STAGED_BRICK_K_DEFAULT = 64
STAGED_WARPS_M = 4
STAGED_WARPS_N = 4
STAGED_WARP_ATOMS_M = 4
STAGED_WARP_ATOMS_N = 4


def multibrick_position_tensor(torch, shape, dtype):
    rows, cols = shape
    row_idx = torch.arange(rows, device="cuda", dtype=torch.int32).view(rows, 1)
    col_idx = torch.arange(cols, device="cuda", dtype=torch.int32).view(1, cols)
    encoded = row_idx * 257 + col_idx
    if dtype == torch.int8:
        encoded = (encoded * 73 + 19) % 255 - 127
    return encoded.to(dtype)


def expected_multibrick_a_dump(torch, A_logical, A_dev, major_pattern: str, brick_k: int, values_per_sme_row: int):
    vpr = values_per_sme_row
    layout = parse_major_pattern(major_pattern)
    brick_m = A_logical.shape[0]
    chunks = []
    if layout.a_k_major:
        for cta_lin in range((brick_m // SMEM_ROWS) * (brick_k // vpr)):
            cta_m = cta_lin // (brick_k // vpr)
            cta_k = cta_lin % (brick_k // vpr)
            chunks.append(
                A_dev[
                    cta_m * SMEM_ROWS : (cta_m + 1) * SMEM_ROWS,
                    cta_k * vpr : (cta_k + 1) * vpr,
                ].contiguous()
            )
    else:
        for cta_lin in range((brick_m // vpr) * (brick_k // SMEM_ROWS)):
            cta_m = cta_lin // (brick_k // SMEM_ROWS)
            cta_k = cta_lin % (brick_k // SMEM_ROWS)
            chunks.append(
                A_logical[
                    cta_m * vpr : (cta_m + 1) * vpr,
                    cta_k * SMEM_ROWS : (cta_k + 1) * SMEM_ROWS,
                ].contiguous()
            )
    return torch.cat([chunk.reshape(-1) for chunk in chunks])


def expected_multibrick_b_dump(torch, B_dev, major_pattern: str, brick_n: int, brick_k: int, values_per_sme_row: int):
    vpr = values_per_sme_row
    layout = parse_major_pattern(major_pattern)
    chunks = []
    if layout.b_k_major:
        for cta_lin in range((brick_n // SMEM_ROWS) * (brick_k // vpr)):
            cta_n = cta_lin // (brick_k // vpr)
            cta_k = cta_lin % (brick_k // vpr)
            chunks.append(
                B_dev[
                    cta_n * SMEM_ROWS : (cta_n + 1) * SMEM_ROWS,
                    cta_k * vpr : (cta_k + 1) * vpr,
                ].contiguous()
            )
    else:
        for cta_lin in range((brick_n // vpr) * (brick_k // SMEM_ROWS)):
            cta_n = cta_lin % (brick_n // vpr)
            cta_k = cta_lin // (brick_n // vpr)
            chunks.append(
                B_dev[
                    cta_k * SMEM_ROWS : (cta_k + 1) * SMEM_ROWS,
                    cta_n * vpr : (cta_n + 1) * vpr,
                ]
                .t()
                .contiguous()
            )
    return torch.cat([chunk.reshape(-1) for chunk in chunks])


def brick_k_from_k_atoms(k_atoms: int, *, geom: MrOperandGeom = MR_GEMM_GEOM) -> int:
    """CTA K-tile size: ``bk = geom.atom_k * k_atoms``."""
    return geom.atom_k * k_atoms


def staged_cta_config(
    *,
    major_pattern: str,
    brick_k: int,
    brick_m: int = STAGED_BRICK_M,
    brick_n: int = STAGED_BRICK_N,
    warps_m: int = STAGED_WARPS_M,
    warps_n: int = STAGED_WARPS_N,
    elem_bits: int = 16,
) -> dict:
    """Compile-time CTA metadata shared by staged G2S/S2R/MMA test kernels."""
    layout = parse_major_pattern(major_pattern)
    geom = MrOperandGeom.from_elem_bits(elem_bits)
    vpr = geom.values_per_sme_row
    num_warps = warps_m * warps_n
    threads = num_warps * WARP_SIZE
    smem_elems = (brick_m + brick_n) * brick_k
    mma_k_slices = brick_k // geom.atom_k
    a_atoms_total, b_atoms_total, _, _ = sme_atom_counts(
        layout,
        brick_m,
        brick_n,
        brick_k,
        values_per_sme_row=vpr,
    )
    a_logical_stride, b_logical_stride = logical_strides(
        layout,
        m=brick_m,
        k=brick_k,
        n=brick_n,
    )
    cta_chunk_elems = geom.cta_chunk_elems
    return {
        "layout": layout,
        "geom": geom,
        "a_mn_major": layout.a_mn_major,
        "b_mn_major": layout.b_mn_major,
        "a_k_major": layout.a_k_major,
        "b_k_major": layout.b_k_major,
        "brick_m": brick_m,
        "brick_n": brick_n,
        "brick_k": brick_k,
        "values_per_sme_row": vpr,
        "threads": threads,
        "smem_elems": smem_elems,
        "mma_k_slices": mma_k_slices,
        "a_atoms_total": a_atoms_total,
        "b_atoms_total": b_atoms_total,
        "a_per_warp": a_atoms_total // num_warps,
        "b_per_warp": b_atoms_total // num_warps,
        "a_logical_stride": a_logical_stride,
        "b_logical_stride": b_logical_stride,
        "cta_chunk_elems": cta_chunk_elems,
        "cta_b_n_cnt": brick_n // vpr,
    }


def staged_k_atoms_config(*, major_pattern: str, k_atoms: int, **kwargs) -> dict:
    """``staged_cta_config`` with ``brick_k = geom.atom_k * k_atoms`` (default f16 geom)."""
    return staged_cta_config(major_pattern=major_pattern, brick_k=brick_k_from_k_atoms(k_atoms), **kwargs)


def expected_warp00_ab_mma_k_slice(
    A_logical,
    B_logical,
    *,
    mma_k: int,
    atom_m: int = 16,
    atom_n: int = 16,
    atom_k: int = 16,
):
    """Logical top-left warp atom (mma_m=0,mma_n=0) for one ``mma_k`` slice."""
    return (
        A_logical[0:atom_m, mma_k * atom_k : (mma_k + 1) * atom_k].contiguous(),
        B_logical[0:atom_n, mma_k * atom_k : (mma_k + 1) * atom_k].contiguous(),
    )


def expected_warp00_atom_gemm(A_logical, B_logical, *, brick_k: int, atom_m: int = 16, atom_n: int = 16):
    """Reference C for warp-00 atom (mma_m=0,mma_n=0) over the full BK tile."""
    a = A_logical[0:atom_m, :brick_k].to(dtype=A_logical.dtype).float()
    b = B_logical[0:atom_n, :brick_k].to(dtype=B_logical.dtype).float()
    return a @ b.T
