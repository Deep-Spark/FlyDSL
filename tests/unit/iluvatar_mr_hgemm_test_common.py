# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Shared helpers for Iluvatar MR HGEMM staged device tests."""

from kernels.iluvatar_mr_common import (
    ATOM_K,
    SMEM_ROWS,
    WARP_SIZE,
    major_pattern_id,
    pattern_sme_atom_counts,
    sme_values_per_row,
)

# Default staged CTA tile (matches production swizzle-1024 preset at k_rep=4).
STAGED_BRICK_M = 256
STAGED_BRICK_N = 256
STAGED_BRICK_K_DEFAULT = 64
STAGED_WARPS_M = 4
STAGED_WARPS_N = 4
STAGED_WARP_ATOMS_M = 4
STAGED_WARP_ATOMS_N = 4


def remap_hgemm_tensors_for_pattern(A, B, major_pattern: str):
    """Physical layout adapter for logical A(m,k), B(n,k) inputs."""
    if major_pattern == "nn":
        return A, B.t().contiguous()
    if major_pattern == "tn":
        return A.t().contiguous(), B.t().contiguous()
    if major_pattern == "nt":
        return A, B
    if major_pattern == "tt":
        return A.t().contiguous(), B
    raise ValueError(f"unknown major pattern: {major_pattern}")


def multibrick_position_tensor(torch, shape, dtype):
    rows, cols = shape
    row_idx = torch.arange(rows, device="cuda", dtype=torch.int32).view(rows, 1)
    col_idx = torch.arange(cols, device="cuda", dtype=torch.int32).view(1, cols)
    encoded = row_idx * 257 + col_idx
    if dtype == torch.int8:
        encoded = (encoded * 73 + 19) % 255 - 127
    return encoded.to(dtype)


def expected_multibrick_a_dump(torch, A_logical, A_dev, major_pattern: str, brick_k: int, values_per_sme_row: int):
    pattern_id = major_pattern_id(major_pattern)
    brick_m = A_logical.shape[0]
    chunks = []
    if pattern_id in (0, 2):
        for atom_idx in range((brick_m // SMEM_ROWS) * (brick_k // values_per_sme_row)):
            mi = atom_idx // (brick_k // values_per_sme_row)
            ki = atom_idx % (brick_k // values_per_sme_row)
            chunks.append(
                A_dev[
                    mi * SMEM_ROWS : (mi + 1) * SMEM_ROWS,
                    ki * values_per_sme_row : (ki + 1) * values_per_sme_row,
                ].contiguous()
            )
    else:
        for atom_idx in range((brick_m // values_per_sme_row) * (brick_k // SMEM_ROWS)):
            mi = atom_idx // (brick_k // SMEM_ROWS)
            ki = atom_idx % (brick_k // SMEM_ROWS)
            chunks.append(
                A_logical[
                    mi * values_per_sme_row : (mi + 1) * values_per_sme_row,
                    ki * SMEM_ROWS : (ki + 1) * SMEM_ROWS,
                ].contiguous()
            )
    return torch.cat([chunk.reshape(-1) for chunk in chunks])


def expected_multibrick_b_dump(torch, B_dev, major_pattern: str, brick_n: int, brick_k: int, values_per_sme_row: int):
    pattern_id = major_pattern_id(major_pattern)
    chunks = []
    if pattern_id in (0, 1):
        for atom_idx in range((brick_n // values_per_sme_row) * (brick_k // SMEM_ROWS)):
            ni = atom_idx % (brick_n // values_per_sme_row)
            ki = atom_idx // (brick_n // values_per_sme_row)
            chunks.append(
                B_dev[
                    ki * SMEM_ROWS : (ki + 1) * SMEM_ROWS,
                    ni * values_per_sme_row : (ni + 1) * values_per_sme_row,
                ]
                .t()
                .contiguous()
            )
    else:
        for atom_idx in range((brick_n // SMEM_ROWS) * (brick_k // values_per_sme_row)):
            ni = atom_idx // (brick_k // values_per_sme_row)
            ki = atom_idx % (brick_k // values_per_sme_row)
            chunks.append(
                B_dev[
                    ni * SMEM_ROWS : (ni + 1) * SMEM_ROWS,
                    ki * values_per_sme_row : (ki + 1) * values_per_sme_row,
                ].contiguous()
            )
    return torch.cat([chunk.reshape(-1) for chunk in chunks])


def brick_k_from_k_rep(k_rep: int) -> int:
    """CTA K-tile size: BK = 16 * k_rep (k_rep=2 -> BK=32)."""
    return 16 * k_rep


def staged_logical_strides(
    *, pattern_id: int, brick_m: int, brick_n: int, brick_k: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return (a_logical_stride, b_logical_stride) for logical A(m,k) / B(n,k) views."""
    a_logical_stride = (1, brick_m) if pattern_id in (1, 3) else (brick_k, 1)
    b_logical_stride = (1, brick_n) if pattern_id in (0, 1) else (brick_k, 1)
    return a_logical_stride, b_logical_stride


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
    pattern_id = major_pattern_id(major_pattern)
    values_per_sme_row = sme_values_per_row(elem_bits)
    num_warps = warps_m * warps_n
    threads = num_warps * WARP_SIZE
    smem_elems = (brick_m + brick_n) * brick_k
    ki_slices = brick_k // ATOM_K
    a_atoms_total, b_atoms_total, _, _ = pattern_sme_atom_counts(
        pattern_id,
        brick_m,
        brick_n,
        brick_k,
        values_per_sme_row=values_per_sme_row,
    )
    a_logical_stride, b_logical_stride = staged_logical_strides(
        pattern_id=pattern_id,
        brick_m=brick_m,
        brick_n=brick_n,
        brick_k=brick_k,
    )
    brick_elems = SMEM_ROWS * values_per_sme_row
    return {
        "pattern_id": pattern_id,
        "brick_m": brick_m,
        "brick_n": brick_n,
        "brick_k": brick_k,
        "values_per_sme_row": values_per_sme_row,
        "threads": threads,
        "smem_elems": smem_elems,
        "ki_slices": ki_slices,
        "a_atoms_total": a_atoms_total,
        "b_atoms_total": b_atoms_total,
        "a_per_warp": a_atoms_total // num_warps,
        "b_per_warp": b_atoms_total // num_warps,
        "a_logical_stride": a_logical_stride,
        "b_logical_stride": b_logical_stride,
        "brick_elems": brick_elems,
        "b_n_chunks": brick_n // values_per_sme_row,
    }


def staged_k_rep_config(*, major_pattern: str, k_rep: int, **kwargs) -> dict:
    """``staged_cta_config`` with ``brick_k = 16 * k_rep``."""
    return staged_cta_config(major_pattern=major_pattern, brick_k=brick_k_from_k_rep(k_rep), **kwargs)


def expected_warp00_ab_ki_slice(
    A_logical,
    B_logical,
    *,
    ki: int,
    atom_m: int = 16,
    atom_n: int = 16,
    atom_k: int = 16,
):
    """Logical top-left warp atom (im=0,jn=0) for one Ki slice."""
    return (
        A_logical[0:atom_m, ki * atom_k : (ki + 1) * atom_k].contiguous(),
        B_logical[0:atom_n, ki * atom_k : (ki + 1) * atom_k].contiguous(),
    )


def expected_warp00_atom_gemm(A_logical, B_logical, *, brick_k: int, atom_m: int = 16, atom_n: int = 16):
    """Reference C for warp-00 atom (im=0,jn=0) over the full BK tile."""
    a = A_logical[0:atom_m, :brick_k].to(dtype=A_logical.dtype).float()
    b = B_logical[0:atom_n, :brick_k].to(dtype=B_logical.dtype).float()
    return a @ b.T
