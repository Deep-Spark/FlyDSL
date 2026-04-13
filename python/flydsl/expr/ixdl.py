# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""IXDL helpers for Iluvatar GPU programming."""

from .._mlir.dialects.fly import AddressSpace, CopyOpIXDLSMELoadType, MmaAtomIXDLMMADType, PointerType, SwizzleType


def _coerce_ir_type(elem_type, opname):
    from .._mlir import ir

    if isinstance(elem_type, type) and hasattr(elem_type, "ir_type"):
        return elem_type.ir_type
    if isinstance(elem_type, ir.Type):
        return elem_type
    raise TypeError(f"{opname}: unsupported elem_type {elem_type}")


def _logical_shape_16x512b(elem_type, opname):
    ty = _coerce_ir_type(elem_type, opname)
    elem_bits = int(ty.width)
    if elem_bits <= 0 or 512 % elem_bits != 0:
        raise ValueError(f"{opname}: unsupported element width {elem_bits}")
    return ty, [16, 512 // elem_bits]


def MMAD(m, n, k, elem_type, elem_type_b=None, elem_type_acc=None):
    """Create an IXDL MMAD MMA atom type.

    The current FlyDSL PoC only supports ``m16n16k16, f16/f16 -> f32``.
    """
    ty = _coerce_ir_type(elem_type, "MMAD")

    ty_b = ty if elem_type_b is None else (elem_type_b.ir_type if hasattr(elem_type_b, "ir_type") else elem_type_b)
    ty_acc = (
        ty if elem_type_acc is None else (elem_type_acc.ir_type if hasattr(elem_type_acc, "ir_type") else elem_type_acc)
    )
    return MmaAtomIXDLMMADType.get(m, n, k, ty, ty_b, ty_acc)


def SMELoad(shape, bit_size, transpose=False):
    """Create an IXDL SME load copy-op type.

    This is the minimal FlyDSL surface for lowering ``fly.copy`` to
    ``ixdl.cp.async``. The current PoC treats one copy atom as a single
    SME load with the provided logical ``shape`` and total ``bit_size``.
    """

    shape = list(shape)
    if not shape:
        raise ValueError("SMELoad: shape must be non-empty")
    return CopyOpIXDLSMELoadType.get(shape, bit_size=bit_size, transpose=transpose)


def SMELoad16x512b(elem_type, transpose=False):
    """Create the common 16x512b IXDL SME load copy-op type.

    The logical element shape is inferred from the element width:
    ``(16, 512 / elem_bits)`` for both the row-major and transpose/column
    variants. The transpose flag changes how IXDL interprets the tile, not the
    logical extents exposed through FlyDSL. For example, ``f16`` becomes
    ``(16, 32)`` in both cases.
    """

    _, shape = _logical_shape_16x512b(elem_type, "SMELoad16x512b")
    return SMELoad(shape, 16 * 512, transpose=transpose)


def cp_async_commit_group(loc=None, ip=None):
    """Commit the current IXDL cp.async group."""

    from .primitive import ixdl_cp_async_commit_group

    return ixdl_cp_async_commit_group(loc=loc, ip=ip)


def cp_async_wait_group(num_groups=0, loc=None, ip=None):
    """Wait until at most ``num_groups`` IXDL cp.async groups remain pending."""

    from .primitive import ixdl_cp_async_wait_group

    return ixdl_cp_async_wait_group(num_groups=num_groups, loc=loc, ip=ip)


def SMELayout16x512b(elem_type, transpose=False):
    """Create the CUTLASS-matched shared-memory layout for a 16x512b SME tile."""

    from .primitive import make_composed_layout, make_int_tuple, make_layout, recast_layout, static, tile_to_shape

    ty, shape = _logical_shape_16x512b(elem_type, "SMELayout16x512b")
    if transpose:
        swizzle = static(SwizzleType.get(2, 3, 4))
        order = make_int_tuple((0, 1))
        outer_bits = make_layout(((4, 4), (32, 4, 4)), ((32, 2048), (1, 512, 128)))
    else:
        swizzle = static(SwizzleType.get(1, 6, 2))
        order = make_int_tuple((1, 0))
        outer_bits = make_layout(((2, 8), (16, 16, 2)), ((16, 512), (1, 32, 4096)))

    offset = make_int_tuple(0)
    shape_val = make_int_tuple(tuple(shape))
    outer = tile_to_shape(recast_layout(outer_bits, 1, int(ty.width)), shape_val, order)
    return make_composed_layout(swizzle, offset, outer)


def SMEView16x512b(base_ptr, elem_type, transpose=False, elem_offset=0):
    """Create a swizzled shared-memory view for one 16x512b SME tile."""

    from .primitive import add_offset, make_view, recast_iter

    ty, _ = _logical_shape_16x512b(elem_type, "SMEView16x512b")
    smem_ptr_ty = PointerType.get(ty, AddressSpace.Shared)
    smem_ptr = recast_iter(smem_ptr_ty, base_ptr)
    if elem_offset:
        smem_ptr = add_offset(smem_ptr, elem_offset)
    return make_view(smem_ptr, SMELayout16x512b(ty, transpose=transpose))


__all__ = [
    "MMAD",
    "SMELoad",
    "SMELoad16x512b",
    "cp_async_commit_group",
    "cp_async_wait_group",
    "SMELayout16x512b",
    "SMEView16x512b",
    "CopyOpIXDLSMELoadType",
    "MmaAtomIXDLMMADType",
]
