# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""IXDL helpers for Iluvatar GPU programming."""

from .._mlir.dialects.fly import MmaAtomIXDLMMADType


def MMAD(m, n, k, elem_type, elem_type_b=None, elem_type_acc=None):
    """Create an IXDL MMAD MMA atom type.

    The current FlyDSL PoC only supports ``m16n16k16, f16/f16 -> f32``.
    """
    from .._mlir import ir

    if isinstance(elem_type, type) and hasattr(elem_type, "ir_type"):
        ty = elem_type.ir_type
    elif isinstance(elem_type, ir.Type):
        ty = elem_type
    else:
        raise TypeError(f"MMAD: unsupported elem_type {elem_type}")

    ty_b = ty if elem_type_b is None else (elem_type_b.ir_type if hasattr(elem_type_b, "ir_type") else elem_type_b)
    ty_acc = (
        ty if elem_type_acc is None else (elem_type_acc.ir_type if hasattr(elem_type_acc, "ir_type") else elem_type_acc)
    )
    return MmaAtomIXDLMMADType.get(m, n, k, ty, ty_b, ty_acc)


__all__ = [
    "MMAD",
    "MmaAtomIXDLMMADType",
]
