# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Shared constants and GEMM layout helpers for Iluvatar kernels."""

from typing import NamedTuple

# Warp width for Iluvatar devices.
WARP_SIZE = 64

# CUTLASS 3.x / CuTe BLAS layout tags for logical A(m,k) @ B(n,k).T (MxK * NxK).
# SME/G2S index naming: kernels.gemm.iluvatar.mr.common (cta_*, mma_*, sme_row_*).
# BLAS layout tags nn/nt/tn/tt -- opaque names on logical A(m,k)/B(n,k); not per-operand M/N/K major letters.
# Host shapes are PyTorch tensor.shape after remap_gemm_tensors:
#   nt: A mn, B mn  -> A (k,m), B (k,n)
#   tn: A k,  B k   -> A (m,k), B (n,k)  [default]
#   nn: A mn, B k   -> A (k,m), B (n,k)
#   tt: A k,  B mn  -> A (m,k), B (k,n)
MAJOR_PATTERN_NN = "nn"
MAJOR_PATTERN_NT = "nt"
MAJOR_PATTERN_TN = "tn"
MAJOR_PATTERN_TT = "tt"
MAJOR_PATTERN_CHOICES = (
    MAJOR_PATTERN_NT,
    MAJOR_PATTERN_TN,
    MAJOR_PATTERN_NN,
    MAJOR_PATTERN_TT,
)
DEFAULT_MAJOR_PATTERN = MAJOR_PATTERN_TN


class GemmLayout(NamedTuple):
    """Per-operand major modes for logical A(m,k) / B(n,k).

    a_mn_major / b_mn_major are authoritative (M on A, N on B when True).
    a_k_major / b_k_major are the complements (K contiguous).
    """

    a_mn_major: bool
    b_mn_major: bool

    @property
    def a_k_major(self) -> bool:
        return not self.a_mn_major

    @property
    def b_k_major(self) -> bool:
        return not self.b_mn_major


_MAJOR_PATTERN_LAYOUT: dict[str, GemmLayout] = {
    MAJOR_PATTERN_NT: GemmLayout(a_mn_major=True, b_mn_major=True),
    MAJOR_PATTERN_TN: GemmLayout(a_mn_major=False, b_mn_major=False),
    MAJOR_PATTERN_NN: GemmLayout(a_mn_major=True, b_mn_major=False),
    MAJOR_PATTERN_TT: GemmLayout(a_mn_major=False, b_mn_major=True),
}


def parse_major_pattern(tag: str) -> GemmLayout:
    """Map a CUTLASS 3.x BLAS layout tag to per-operand major modes."""
    try:
        return _MAJOR_PATTERN_LAYOUT[tag]
    except KeyError as exc:
        raise ValueError(f"unknown major pattern: {tag}") from exc


def remap_gemm_tensors(A, B, layout: GemmLayout | str):
    """Adapt host A/B to the physical layout expected by Iluvatar GEMM kernels.

    Logical A(m,k) and B(n,k). Default major_pattern "tn" (DEFAULT_MAJOR_PATTERN) is
    k-major on both operands, so PyTorch (m,k) and (n,k) need no change. mn-major
    operands need .t().contiguous() so the non-K dim is contiguous in device memory.

    The kernel still uses GemmLayout as logical A(m,k)/B(n,k); this only changes
    physical tensor.shape before launch.

    layout: GemmLayout or a tag string ("nt", "tn", ...) parsed by parse_major_pattern.
    Returns (a_dev, b_dev).
    """
    if isinstance(layout, str):
        layout = parse_major_pattern(layout)
    a_dev = A.t().contiguous() if layout.a_mn_major else A
    b_dev = B.t().contiguous() if layout.b_mn_major else B
    return a_dev, b_dev


__all__ = [
    "DEFAULT_MAJOR_PATTERN",
    "GemmLayout",
    "MAJOR_PATTERN_CHOICES",
    "MAJOR_PATTERN_NT",
    "MAJOR_PATTERN_TN",
    "MAJOR_PATTERN_NN",
    "MAJOR_PATTERN_TT",
    "parse_major_pattern",
    "remap_gemm_tensors",
    "WARP_SIZE",
]
