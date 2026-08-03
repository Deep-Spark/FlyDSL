# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""High-level API for Iluvatar CQ (ivcore30) TCU MMA and SmexMtx S2R atoms.

Exposes ``CQMma`` for CQ TCU MMA and ``CQMtxLoadn`` for shared->register
matrix loads (``loadn16`` / ``loadn64`` → ``ixdl.mtx_loadn_*``).
"""

from ..._mlir import ir
from ..._mlir._mlir_libs._mlirDialectsFlyIXDL import CopyOpCQMtxLoadnType, MmaOpCQMmaType


def _to_ir_type(t) -> "ir.Type":
    """Coerce a FlyDSL numeric type / ir.Type to an ``ir.Type``."""
    if isinstance(t, ir.Type):
        return t
    if hasattr(t, "ir_type"):
        return t.ir_type
    raise TypeError(f"expected a NumericType or ir.Type, got {t!r}")


class CQMtxPattern:
    """SmexMtx shared-layout pattern (pairs with SME mtx G2S).

    Incompatible with LegacySme (``ldmatrix`` / byte swizzle) on the same buffer.
    Swizzle policy for SmexMtx is Bypass — dword/SMEX layout is handled by the
    matrix-load / SMEX path, not MR LegacyByteSwizzle.
    """

    Loadn16 = 0  # pairs with smex_loadn_16x1b64_mtx
    Loadn64 = 1  # pairs with smex_loadn_64x1b64_mtx


class CQMtxDir:
    """S2R gather direction (SoT: Row = A, Col = B)."""

    Row = 0
    Col = 1


def CQMtxLoadn(pattern, direction, bit_width, x2=True):
    """Create a CQ SmexMtx shared->register matrix-load atom.

    Args:
        pattern: ``CQMtxPattern.Loadn16`` or ``Loadn64`` (both lower to
            ``ixdl.mtx_loadn_*``; the value documents G2S pairing / EmPart).
        direction: ``CQMtxDir.Row`` (A) or ``Col`` (B).
        bit_width: multiplicand element width, ``8`` or ``16``.
        x2: if true (default), load 64b / ``vector<2xi32>`` per lane (full
            base-tile MMA fragment half); if false, 32b / ``i32``.

    Thr layouts match CQ MMA base-tile A/B fragments so ``make_tiled_copy_A/B``
    can couple this atom to ``CQMma``.
    """
    return CopyOpCQMtxLoadnType.get(int(pattern), int(direction), int(bit_width), int(bool(x2)))


def CQMma(m, n, k, elem_ty_a, elem_ty_b, elem_ty_acc):
    """Create an Iluvatar CQ (ivcore30) TCU MMA atom (``D = A*B + C``).

    Supported families (``(M,N)`` in {16x16, 32x32, 16x64, 64x16}):

    - f16/bf16 -> f32, ``K=16``, A and B must match
    - i8/ui8 -> i32, ``K=32``, A and B must match (signless i8 -> s8, ui8 -> u8)
    - f8E4M3/f8E5M2 -> f32 or f16; A/B may mix; ``K=32``
    """
    return MmaOpCQMmaType.get(
        int(m),
        int(n),
        int(k),
        _to_ir_type(elem_ty_a),
        _to_ir_type(elem_ty_b),
        _to_ir_type(elem_ty_acc),
    )
