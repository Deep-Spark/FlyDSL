# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Iluvatar CQ (`fly_cq`) atom helpers (placeholder MMA / copy lowering via LLVM)."""

from .._mlir.dialects.fly_cq import CopyOpCQ_ScalarMemType, MmaOpCQ_MatmulF32Type
from .._mlir.extras import types as T


def ScalarMem32():
    """`!fly_cq.scalar_mem<32>` — placeholder register copy (LLVM load/store)."""
    return CopyOpCQ_ScalarMemType.get(32)


def MatmulF32(m=16, n=16, k=4, elem_ty_ab=None, elem_ty_acc=None):
    """Return `!fly_cq.matmul_f32<mxnxk, (ty, ty) -> ty_acc>` for use with `make_mma_atom`.

    Defaults match the CQ placeholder verifier (16x16x4, f32).
    """
    if elem_ty_ab is None:
        ty_ab = T.f32()
    else:
        ty_ab = elem_ty_ab.ir_type if hasattr(elem_ty_ab, "ir_type") else elem_ty_ab
    if elem_ty_acc is None:
        ty_acc = T.f32()
    else:
        ty_acc = elem_ty_acc.ir_type if hasattr(elem_ty_acc, "ir_type") else elem_ty_acc
    return MmaOpCQ_MatmulF32Type.get(m, n, k, ty_ab, ty_ab, ty_acc)
