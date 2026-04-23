# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Iluvatar ``ivcore11`` MMA (ixdl.mmad) atom wrappers for flydsl.expr.

Mirrors the ``flydsl.expr.rocdl`` module layout but targets the Iluvatar
``ixdl`` dialect. The only atom family exposed here is ``MMAD`` — a single
parameterized type (``MmaOpIX11_MMADType``) that, depending on its
element-type triple, lowers to one of the five CUTLASS-documented IX11
intrinsics:

  * 16x16x16 f32 = f32 * f32 + f32
  * 16x16x16 f32 = f16 * f16 + f32
  * 16x16x16 f32 = bf16 * bf16 + f32
  * 16x16x32 i32 = i8  * i8  + i32  (signed)
  * 16x16x32 i32 = u8  * u8  + i32  (unsigned — reserved for a future atom)

Usage example (same shape as the existing MFMA helpers):

    from flydsl.expr import ixdl
    from flydsl._mlir.extras import types as T

    mma_op = ixdl.MMAD(16, 16, 16, T.f16(), T.f32())
"""

from .._mlir._mlir_libs._mlirDialectsFlyIXDL import MmaOpIX11_MMADType
from .._mlir.extras import types as T


def _ir(ty):
    """Accept either a raw MLIR type or a flydsl DSL type wrapper."""
    return ty.ir_type if hasattr(ty, "ir_type") else ty


def MMAD(m, n, k, elem_type, elem_type_b=None, elem_type_acc=None):
    """Create an ``ivcore11`` MMAD atom type (``ixdl.mmad`` lowering).

    Args:
        m, n, k: MMAD tile dimensions. Supported triples are
            ``(16, 16, 16)`` for f32/f16/bf16 and ``(16, 16, 32)`` for i8.
        elem_type: Element type for the A operand.
        elem_type_b: Element type for the B operand (defaults to ``elem_type``).
        elem_type_acc: Element type for the accumulator (defaults to
            ``f32`` for 16-bit/32-bit inputs, ``i32`` for i8 inputs).
    """
    ty = _ir(elem_type)
    ty_b = ty if elem_type_b is None else _ir(elem_type_b)

    if elem_type_acc is not None:
        ty_acc = _ir(elem_type_acc)
    else:
        width = getattr(ty, "width", None)
        if width is None:
            ty_acc = T.f32()
        elif ty == T.f16() or ty == T.bf16() or ty == T.f32():
            ty_acc = T.f32()
        else:
            ty_acc = T.i32()
    return MmaOpIX11_MMADType.get(m, n, k, ty, ty_b, ty_acc)


__all__ = [
    "MmaOpIX11_MMADType",
    "MMAD",
]
