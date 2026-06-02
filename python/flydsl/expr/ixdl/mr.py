# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""High-level API for the Iluvatar MR async copy (SME Load series).

Mirrors the FlyROCDL ``BufferCopy*`` factories, but targets the Iluvatar SME
``ixdl.cp_async.*`` op family. The copy atom is parameterised by a single
``sme_swizzle`` value. The element dtype and row/column-major selection are
carried elsewhere (``fly.copy_atom<..., elemBits>`` / the kernel layout
factory), not by this parameter.
"""

from ..._mlir._mlir_libs._mlirDialectsFlyIXDL import CopyOpMRAsyncCpType
from ..._mlir.dialects.fly import ModSwizzleType, PointerType, SwizzleType
from ..._mlir.dialects.fly_ixdl import TargetAddressSpace
from ..primitive import (
    get_iter,
    get_layout,
    get_leaves,
    get_stride,
    make_composed_layout,
    make_layout,
    make_ptr,
    make_view,
    static,
)
from ..typing import Int32, Numeric, Tensor


class SMESwizzle:
    """MR SME swizzle encoding."""

    NoSwizzle = 0  # 32b row-major / INTER
    Col = 1  # col-major xor swizzle -> colxfb8 (also bf16/fp16 col)
    Row8b = 2  # 8b row-major mod/add swizzle -> rowxfb8 (Swizzle_Mod)
    Row16b = 3  # 16b row-major xor swizzle -> rowxfb16


class SMEMajor:
    """SME shared-memory layout orientation."""

    MN = "MN"
    K = "K"


def _normalize_major(major):
    if isinstance(major, str):
        major = major.upper()
    if major not in (SMEMajor.MN, SMEMajor.K):
        raise ValueError(f"major must be {SMEMajor.MN!r} or {SMEMajor.K!r}, got {major!r}")
    return major


def _dtype_width(dtype):
    if isinstance(dtype, type) and issubclass(dtype, Numeric):
        return dtype.width
    if hasattr(dtype, "width"):
        return int(dtype.width)
    raise TypeError(f"dtype must be a FlyDSL Numeric type, got {dtype!r}")


def _ceil_div(lhs, rhs):
    return (lhs + rhs - 1) // rhs


def _upcast_shape_stride(shape, stride, factor):
    if isinstance(shape, int):
        if stride == 0:
            return shape, stride

        abs_stride = abs(stride)
        if abs_stride == 0:
            return shape, stride
        if abs_stride % factor != 0 and factor % abs_stride != 0:
            raise ValueError(
                f"SME layout upcast requires divisible stride/factor, got stride={stride}, factor={factor}"
            )

        sign = -1 if stride < 0 else 1
        new_shape = _ceil_div(shape, _ceil_div(factor, abs_stride))
        new_stride = _ceil_div(abs_stride, factor) * sign
        return new_shape, new_stride

    out_shape = []
    out_stride = []
    for child_shape, child_stride in zip(shape, stride, strict=True):
        new_shape, new_stride = _upcast_shape_stride(child_shape, child_stride, factor)
        out_shape.append(new_shape)
        out_stride.append(new_stride)
    return tuple(out_shape), tuple(out_stride)


def _upcast_swizzle(mask, base, shift, factor):
    if mask == 0:
        return 0, 0, 0

    log_factor = factor.bit_length() - 1
    base -= log_factor
    if base < 0:
        raise ValueError(
            f"cannot upcast SME swizzle S<{mask},{base + log_factor},{shift}> "
            f"to {factor}-bit elements without losing sub-element swizzle bits"
        )
    return mask, base, shift


def _sme_shared_layout_bits(sme_swizzle, major):
    if int(sme_swizzle) == SMESwizzle.NoSwizzle:
        swizzle = ("xor", 0, 0, 0)
        if major == SMEMajor.MN:
            return swizzle, (512, 16), (1, 512)
        return swizzle, (16, 512), (512, 1)

    if int(sme_swizzle) == SMESwizzle.Col:
        swizzle = ("xor", 2, 4, 4)
        if major == SMEMajor.MN:
            return swizzle, ((32, 4, 4), (4, 4)), ((1, 512, 128), (32, 2048))
        return swizzle, ((4, 4), (32, 4, 4)), ((32, 2048), (1, 512, 128))

    if int(sme_swizzle) == SMESwizzle.Row8b:
        swizzle = ("mod", 2, 6, 2)
        if major == SMEMajor.MN:
            return swizzle, ((8, 4, 4, 4), (4, 4)), ((1, 32, 128, 2048), (8, 512))
        return swizzle, ((4, 4), (8, 4, 4, 4)), ((8, 512), (1, 32, 128, 2048))

    if int(sme_swizzle) == SMESwizzle.Row16b:
        swizzle = ("xor", 1, 7, 2)
        if major == SMEMajor.MN:
            return swizzle, ((16, 16, 2), (2, 8)), ((1, 32, 4096), (16, 512))
        return swizzle, ((2, 8), (16, 16, 2)), ((16, 512), (1, 32, 4096))

    raise ValueError(f"unsupported SME swizzle = {sme_swizzle!r}")


def make_sme_shared_layout(sme_swizzle, dtype, *, major=SMEMajor.MN):
    """Build the shared-memory layout produced by an MR SME async copy.

    The copy atom describes only the instruction footprint; this layout should
    be used on the destination shared-memory view when later consumers access
    the tile with logical coordinates.
    """
    major = _normalize_major(major)
    val_bits = _dtype_width(dtype)
    if val_bits not in (8, 16, 32):
        raise ValueError(f"MR SME shared layout supports 8/16/32-bit elements, got {val_bits}")
    if val_bits & (val_bits - 1):
        raise ValueError(f"element bit width must be a power of two, got {val_bits}")

    swizzle, bit_shape, bit_stride = _sme_shared_layout_bits(sme_swizzle, major)
    elem_shape, elem_stride = _upcast_shape_stride(bit_shape, bit_stride, val_bits)
    outer = make_layout(elem_shape, elem_stride)

    kind, mask, base, shift = swizzle
    mask, base, shift = _upcast_swizzle(mask, base, shift, val_bits)
    if kind == "mod":
        inner_ty = ModSwizzleType.get(mask, base, shift)
    else:
        inner_ty = SwizzleType.get(mask, base, shift)
    return make_composed_layout(static(inner_ty), 0, outer)


def MRAsyncCp(sme_swizzle):
    """Create an Iluvatar MR async copy atom (SME Load series).

    Args:
        sme_swizzle: MR SME swizzle value, which selects the SME builtin /
            IXDL op:
            0 NoSwizzle (b32 row / INTER), 1 Col (colxfb8),
            2 Row8b (rowxfb8), 3 Row16b (rowxfb16).

    Note:
        Lowering support follows the IXDL op family:
        ``NoSwizzle`` supports 32-bit row, ``Col`` supports 8/16/32-bit col,
        ``Row8b`` supports 8-bit row, and ``Row16b`` supports 16-bit row.
    """
    return CopyOpMRAsyncCpType.get(int(sme_swizzle))


# Convenience aliases (the 4 SMESwizzle states).
MRAsyncCpNoSwizzle = lambda: CopyOpMRAsyncCpType.get(SMESwizzle.NoSwizzle)  # b32 row / INTER
MRAsyncCpCol = lambda: CopyOpMRAsyncCpType.get(SMESwizzle.Col)  # colxfb8 (incl. bf16/fp16 col)
MRAsyncCpRow8b = lambda: CopyOpMRAsyncCpType.get(SMESwizzle.Row8b)  # rowxfb8 (Swizzle_Mod)
MRAsyncCpRow16b = lambda: CopyOpMRAsyncCpType.get(SMESwizzle.Row16b)  # rowxfb16


def make_sme_gmem_tensor(tensor: Tensor, *, leading_stride=None) -> Tensor:
    """Wrap ``tensor`` in an SME global-memory view (``#fly_ixdl.sme_gmem``).

    The SME descriptor needs the leading (outermost) stride in element units;
    ``make_ptr`` stores ``leading_stride * elem_bytes`` into the fat pointer's
    ``stride_byte`` field. By default the leading stride is taken from the
    tensor layout's first stride leaf; pass ``leading_stride`` to override.

    Mirrors :func:`flydsl.expr.rocdl.universal.make_buffer_tensor`.
    """
    elem_ty = tensor.element_type

    ptr = get_iter(tensor)
    layout = get_layout(tensor)

    if leading_stride is None:
        # The leading (outermost) stride leaf, in element units.
        leading_stride = get_leaves(get_stride(layout))[0]

    sme_ptr_ty = PointerType.get(
        elem_ty=elem_ty.ir_type,
        address_space=TargetAddressSpace.SmeGmem,
        alignment=ptr.alignment,
    )
    sme_ptr = make_ptr(sme_ptr_ty, [ptr, Int32(leading_stride).ir_value()])
    return make_view(sme_ptr, layout)
