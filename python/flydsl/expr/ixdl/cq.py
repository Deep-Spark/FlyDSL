# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""High-level API for Iluvatar CQ (ivcore30) TCU MMA atoms.

Exposes ``CQMma`` for CQ TCU matrix-multiply-accumulate atoms (base + long-mtx).
"""

from ..._mlir import ir
from ..._mlir._mlir_libs._mlirDialectsFlyIXDL import MmaOpCQMmaType


def _to_ir_type(t) -> "ir.Type":
    """Coerce a FlyDSL numeric type / ir.Type to an ``ir.Type``."""
    if isinstance(t, ir.Type):
        return t
    if hasattr(t, "ir_type"):
        return t.ir_type
    raise TypeError(f"expected a NumericType or ir.Type, got {t!r}")


def CQMma(m, n, k, elem_ty_a, elem_ty_b, elem_ty_acc):
    """Create an Iluvatar CQ (ivcore30) TCU MMA atom (``D = A*B + C``).

    Supported families (``(M,N)`` in {16x16, 32x32, 16x64, 64x16}):

    - f16/bf16 -> f32, ``K=16``, A and B must match
    - i8/ui8 -> i32, ``K=32``, A and B must match signedness
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
