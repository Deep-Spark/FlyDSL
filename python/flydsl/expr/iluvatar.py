# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Iluvatar (ivcore11 / FlyIXDL) DSL extension.

Exposes the MR SME GEMM building blocks:

- Atom payload type constructors: :func:`MMAD`, :func:`SLBLoad`, :func:`AsyncCopy16x64B8Row`.
- SME async-copy synchronization helpers: :func:`cp_async_commit_group`,
  :func:`cp_async_wait_group`, :func:`barrier`.

This module is imported lazily (e.g. ``import flydsl.expr.iluvatar`` or via the
``fx.flyixdl`` alias). It is **not** pulled in by the default ``flydsl.expr``
import so ROCm-only environments are unaffected.
"""

from .._mlir._mlir_libs._mlirDialectsFlyIXDL import (
    CopyOpMRAsyncCopy16x64B8RowType,
    CopyOpMRSLBLoadType,
    MmaOpMR_MMADType,
)

__all__ = [
    "MmaOpMR_MMADType",
    "CopyOpMRSLBLoadType",
    "CopyOpMRAsyncCopy16x64B8RowType",
    "MMAD",
    "SLBLoad",
    "AsyncCopy16x64B8Row",
    "make_smem_tile",
    "sme_g2s_tile",
    "cp_async_commit_group",
    "cp_async_wait_group",
    "barrier",
    "_i32_const",
]


def _as_ir_type(elem_type):
    from .._mlir import ir

    if isinstance(elem_type, type) and hasattr(elem_type, "ir_type"):
        return elem_type.ir_type
    if hasattr(elem_type, "ir_type"):
        return elem_type.ir_type
    if isinstance(elem_type, ir.Type):
        return elem_type
    raise TypeError(f"unsupported element type {elem_type!r}")


def MMAD(m, n, k, elem_type, elem_type_b=None, elem_type_acc=None):
    """Create an ivcore11 SME MMAD atom type.

    Phase-1 supports the fixed FP16 shape ``16x16x16`` with an FP32
    accumulator. ``elem_type`` is used for A (and B/accumulator when the
    others are not given).
    """
    from .._mlir import ir  # noqa: F401

    ty = _as_ir_type(elem_type)
    ty_b = ty if elem_type_b is None else _as_ir_type(elem_type_b)
    if elem_type_acc is None:
        from .typing import T

        ty_acc = T.f32
    else:
        ty_acc = _as_ir_type(elem_type_acc)
    return MmaOpMR_MMADType.get(m, n, k, ty, ty_b, ty_acc)


def SLBLoad(bit_size):
    """Create a shared->register SLB load copy-op (lowers to ``llvm.load``)."""
    return CopyOpMRSLBLoadType.get(bit_size)


def AsyncCopy16x64B8Row(bit_size=128):
    """Create a stateful global->shared SME ``cp.async`` copy-op.

    Despite the legacy ``16x64.b8.row`` mnemonic, this lowers to the f16
    row-major ``rowxfb16`` SME load (``ixdl.cp.async`` shape ``[16,32]``,
    elementSize 16) to match the Row16b SmemAtom built by ``make_smem_tile``.
    """
    return CopyOpMRAsyncCopy16x64B8RowType.get(bit_size)


# ---------------------------------------------------------------------------
# make_smem_tile: ComposedLayout(Swizzle . Offset . Outer) over the kernel's
# dynamic shared-memory buffer. This is a *layout constructor*, not an
# allocator -- it builds the SME vertical-packing `Outer` frame and stacks the
# XOR swizzle so that partition_S/partition_A + fx.copy(SLBLoad) land on the
# correct physical SLB addresses.
#
# Outer formula (see ivcore11 SME GEMM playbook section 5.3):
#   M == 16:
#     shape  = ((2, 8),    (K/2, 2))
#     stride = ((1, K),    (2, 8*K))
#   M == 16*n:
#     shape  = ((2, 8, n),          (K/2, 2))
#     stride = ((1, K, 16*K),       (2, 8*K))
# ---------------------------------------------------------------------------


def _swizzle_from_kwargs(for_mma, swizzle, swizzle_byte, elem_bytes):
    """Resolve the XOR swizzle triple ``(mask, base, shift)`` in element space."""
    import math

    if for_mma is not None:
        # ivcore11 FP16 SME tiles use the canonical (1, 6, 2) element-space swizzle.
        return (1, 6, 2)
    if swizzle is not None:
        return tuple(swizzle)
    if swizzle_byte is not None:
        mask, base_byte, shift = swizzle_byte
        base_elem = base_byte - int(math.log2(elem_bytes))
        return (mask, base_elem, shift)
    return None


def _swizzle_mod_from_kwargs(swizzle_mod, swizzle_mod_byte, elem_bytes):
    """Resolve the modular (rowxfb8) SwizzleMod triple ``(mask, base, shift)``.

    ``swizzle_mod`` is already element-space. ``swizzle_mod_byte`` is byte-space
    and converted with ``base_elem = base_byte - log2(elem_bytes)`` (the FlyDSL
    convention; no smem_ptr_flag specialization). For int8 (1 byte) the rowxfb8
    byte-space ``(2, 6, 2)`` maps to the element-space ``(2, 6, 2)`` used by the
    Row8b SmemAtom (see swizzle_mod_recast.mlir / swizzle_mod_crd2idx.mlir).
    """
    import math

    if swizzle_mod is not None:
        return tuple(swizzle_mod)
    if swizzle_mod_byte is not None:
        mask, base_byte, shift = swizzle_mod_byte
        base_elem = base_byte - int(math.log2(elem_bytes))
        return (mask, base_elem, shift)
    return None


def make_smem_tile(M, K, dtype, *, for_mma=None, swizzle=(1, 6, 2), swizzle_byte=None,
                   swizzle_mod=None, swizzle_mod_byte=None,
                   base_offset_elems=0, loc=None, ip=None):
    """Build a swizzled shared-memory tile view for SME GEMM.

    Returns a ``fly.memref`` view whose layout is
    ``ComposedLayout = Swizzle . Offset(0) . Outer`` over the kernel's dynamic
    shared buffer. ``base_offset_elems`` lets several tiles (e.g. sA/sB) share
    one dynamic buffer by carving out disjoint element ranges.

    The inner swizzle is one of:

    - XOR ``Swizzle`` (default; ``for_mma`` / ``swizzle`` / ``swizzle_byte``) for
      the f16 ``rowxfb16`` Row16b atom.
    - modular ``SwizzleMod`` (``swizzle_mod`` / ``swizzle_mod_byte``) for the
      int8 ``rowxfb8`` Row8b atom. ``swizzle_mod`` takes precedence over the XOR
      kwargs when set.
    """
    from . import primitive as P

    assert M % 16 == 0, "make_smem_tile requires M % 16 == 0"
    assert K % 2 == 0, "make_smem_tile requires K % 2 == 0"

    ir_dtype = _as_ir_type(dtype)
    elem_bytes = max(1, ir_dtype.width // 8) if hasattr(ir_dtype, "width") else 2

    # get_dyn_shared/recast_iter expect a Numeric subclass (e.g. fx.Float16).
    smem_iter = P.get_dyn_shared(dtype, loc=loc, ip=ip)
    if base_offset_elems:
        smem_iter = P.add_offset(smem_iter, int(base_offset_elems), loc=loc, ip=ip)

    if M == 16:
        outer = P.make_layout(((2, 8), (K // 2, 2)),
                              ((1, K), (2, 8 * K)), loc=loc, ip=ip)
    else:
        m_sub = M // 16
        outer = P.make_layout(((2, 8, m_sub), (K // 2, 2)),
                              ((1, K, 16 * K), (2, 8 * K)), loc=loc, ip=ip)

    # Modular (rowxfb8) SwizzleMod takes precedence over the XOR swizzle.
    mod_triple = _swizzle_mod_from_kwargs(swizzle_mod, swizzle_mod_byte, elem_bytes)
    if mod_triple is not None:
        from .._mlir.dialects.fly import SwizzleModType

        mask, base, shift = mod_triple
        swz = P.static(SwizzleModType.get(mask, base, shift), loc=loc, ip=ip)
        offset0 = P.make_int_tuple(0, loc=loc, ip=ip)
        composed = P.make_composed_layout(swz, offset0, outer, loc=loc, ip=ip)
        return P.make_view(smem_iter, composed, loc=loc, ip=ip)

    swz_triple = _swizzle_from_kwargs(for_mma, swizzle, swizzle_byte, elem_bytes)
    if swz_triple is None:
        return P.make_view(smem_iter, outer, loc=loc, ip=ip)

    from .._mlir.dialects.fly import SwizzleType

    mask, base, shift = swz_triple
    swz = P.static(SwizzleType.get(mask, base, shift), loc=loc, ip=ip)
    offset0 = P.make_int_tuple(0, loc=loc, ip=ip)
    composed = P.make_composed_layout(swz, offset0, outer, loc=loc, ip=ip)
    return P.make_view(smem_iter, composed, loc=loc, ip=ip)


def sme_g2s_tile(src, dst, stride=None, *, dtype=None, bit_size=128, loc=None, ip=None):
    """Cooperative global->shared SME copy for a tile.

    Reuses the (FileCheck-verified) ``AsyncCopy16x64B8Row`` copy-atom lowering:
    builds the stateful copy atom and issues ``fx.copy(atom, src, dst)`` which
    lowers to the ``ixdl.cp.async`` descriptor sequence, followed by a commit
    fence. ``stride`` (an i32 byte stride, see :func:`_i32_const`) is accepted
    for the multi-block schedule; the descriptor stride is currently derived
    from the source layout during lowering.

    NOTE: per-thread tile partitioning / exact geometry is finalized together
    with on-device numeric validation; this wrapper keeps the kernel surface
    identical to the playbook (``fx.ixdl.sme_g2s_tile(bA, sA, stride)``).
    """
    from . import primitive as P

    if dtype is None:
        from .typing import T

        dtype = T.f16
    atom = P.make_copy_atom(AsyncCopy16x64B8Row(bit_size), dtype, loc=loc, ip=ip)
    # The SME cp.async is a *cooperative whole-tile* transfer: emit the atom call
    # directly so it survives to convert-fly-to-ixdl as one ixdl.cp.async, instead
    # of going through fx.copy -> ExpandCopyOp (which would decompose the tile).
    P.copy_atom_call(atom, src, dst, pred=None, loc=loc, ip=ip)
    cp_async_commit_group(loc=loc, ip=ip)


def _i32_const(value, *, loc=None, ip=None):
    """Materialize an ``i32`` constant (used for SME G2S byte strides)."""
    from .._mlir import ir
    from .._mlir.dialects import arith

    i32 = ir.IntegerType.get_signless(32)
    return arith.constant(i32, int(value), loc=loc, ip=ip)


# ---------------------------------------------------------------------------
# Synchronization helpers (commit / wait / barrier)
#
# Per the MR async-copy design, commit/wait live OUTSIDE the copy atom as
# explicit kernel helpers. They emit the corresponding `ixdl` ops via the
# generic-op path so no extra Python dialect binding is required.
# ---------------------------------------------------------------------------


def _emit_ixdl(op_name, operands=None, attributes=None, results=None, *, loc=None, ip=None):
    from .._mlir import ir
    from .._mlir.dialects import llvm as _llvm  # noqa: F401  (ensures deps loaded)

    operands = list(operands or [])
    results = list(results or [])
    op = ir.Operation.create(
        op_name,
        results=results,
        operands=operands,
        attributes=attributes or {},
        loc=loc,
        ip=ip,
    )
    return op


def cp_async_commit_group(*, loc=None, ip=None):
    """Emit ``ixdl.cp.async.commit.group`` (batch fence for issued SME loads)."""
    return _emit_ixdl("ixdl.cp.async.commit.group", loc=loc, ip=ip)


def cp_async_wait_group(n=0, *, loc=None, ip=None):
    """Emit ``ixdl.cp.async.wait.group`` (synchronize, allow ``n`` in flight)."""
    from .._mlir import ir

    n_attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), int(n))
    return _emit_ixdl("ixdl.cp.async.wait.group", attributes={"n": n_attr}, loc=loc, ip=ip)


def barrier(*, loc=None, ip=None):
    """Emit ``ixdl.barrier`` (workgroup barrier)."""
    return _emit_ixdl("ixdl.barrier", loc=loc, ip=ip)
