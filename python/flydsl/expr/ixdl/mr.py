# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""High-level API for the Iluvatar MR async copy (SME Load series).

Copy atoms are parameterised by ``sme_swizzle``, which selects the SME
shared-memory swizzle state. Element dtype and ``Major::K/MN`` axis are carried
elsewhere (``fly.copy_atom`` / kernel layout factories).
"""

from ..._mlir import ir
from ..._mlir._mlir_libs._mlirDialectsFlyIXDL import CopyOpMRAsyncCpType, MmaOpMRMmaType
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
    recast_layout,
    static,
)
from ..typing import Int32, Tensor


class SMESwizzle:
    """Iluvatar SME shared-memory swizzle state (4 states)."""

    NoSwizzle = 0  # 32b row-major / INTER
    Col = 1  # col-major xor swizzle -> colxfb8 (also bf16/fp16 col)
    Row8b = 2  # 8b row-major mod/add swizzle -> rowxfb8 (mod swizzle)
    Row16b = 3  # 16b row-major xor swizzle -> rowxfb16


class SMEMajor:
    """Major axis (K / MN) selection for SME shared layouts."""

    MN = 0
    K = 1


def MRAsyncCp(sme_swizzle):
    """Create an Iluvatar MR async copy atom (SME Load series).

    Args:
        sme_swizzle: SME swizzle-state value (4 states), which alone
            determines the SME builtin / IXDL op:
            0 NoSwizzle (b32 row / INTER), 1 Col (colxfb8),
            2 Row8b (rowxfb8), 3 Row16b (rowxfb16).

    Note:
        Lowering support follows the IXDL op family:
        ``NoSwizzle`` supports 32-bit row, ``Col`` supports 8/16/32-bit col,
        ``Row8b`` supports 8-bit row, and ``Row16b`` supports 16-bit row.
    """
    return CopyOpMRAsyncCpType.get(int(sme_swizzle))


def _to_ir_type(t) -> "ir.Type":
    """Coerce a FlyDSL numeric type / ir.Type to an ``ir.Type``."""
    if isinstance(t, ir.Type):
        return t
    if hasattr(t, "ir_type"):
        return t.ir_type
    raise TypeError(f"expected a NumericType or ir.Type, got {t!r}")


def MRMma(m, n, k, elem_ty_a, elem_ty_b, elem_ty_acc):
    """Create an Iluvatar MR (ivcore11) TCU MMA atom (``D = A*B + C``).

    Supported (M, N, K, A, B, acc):
        (16, 16, 16, f16,  f16,  f32),
        (16, 16, 16, bf16, bf16, f32),
        (16, 16, 16, f32,  f32,  f32),
        (16, 16, 32, i8,   i8,   i32).
    """
    return MmaOpMRMmaType.get(
        int(m),
        int(n),
        int(k),
        _to_ir_type(elem_ty_a),
        _to_ir_type(elem_ty_b),
        _to_ir_type(elem_ty_acc),
    )


# Convenience aliases (the 4 SMESwizzle states).
MRAsyncCpNoSwizzle = lambda: CopyOpMRAsyncCpType.get(SMESwizzle.NoSwizzle)  # b32 row / INTER
MRAsyncCpCol = lambda: CopyOpMRAsyncCpType.get(SMESwizzle.Col)  # colxfb8 (incl. bf16/fp16 col)
MRAsyncCpRow8b = lambda: CopyOpMRAsyncCpType.get(SMESwizzle.Row8b)  # rowxfb8 (Swizzle_Mod)
MRAsyncCpRow16b = lambda: CopyOpMRAsyncCpType.get(SMESwizzle.Row16b)  # rowxfb16


def _elem_bits(elem_type) -> int:
    if hasattr(elem_type, "width"):
        return int(elem_type.width)
    if hasattr(elem_type, "ir_type") and hasattr(elem_type.ir_type, "width"):
        return int(elem_type.ir_type.width)
    raise TypeError(f"elem_type must carry bit width information, got {elem_type!r}")


def _normalize_major(major) -> int:
    if isinstance(major, str):
        text = major.strip().upper()
        if text == "MN":
            return SMEMajor.MN
        if text == "K":
            return SMEMajor.K
    val = int(major)
    if val not in (SMEMajor.MN, SMEMajor.K):
        raise ValueError(f"invalid major={major}, expected MN/K or 0/1")
    return val


def _sme_bit_spec(sme_swizzle, elem_bits, is_mn):
    """Return (outer_shape, outer_stride, swizzle_spec) for one SMESwizzle state.

    The layout shape/stride are in *bit* granularity, while the swizzle base is
    *byte*-granular (an SME swizzle permutes whole bytes and acts on byte
    addresses). ``swizzle_spec`` is ``None`` for NoSwizzle, else
    ``(kind, B, M, S)`` with ``kind in {"xor", "mod"}`` and ``M`` the
    byte-granular swizzle base.
    """
    if sme_swizzle == SMESwizzle.NoSwizzle:
        if elem_bits != 32:
            raise ValueError(f"NoSwizzle shared layout requires 32-bit element type, got {elem_bits}")
        shape, stride = ((512, 16), (1, 512)) if is_mn else ((16, 512), (512, 1))
        return shape, stride, None
    if sme_swizzle == SMESwizzle.Col:
        if elem_bits not in (8, 16, 32):
            raise ValueError(f"Col shared layout requires 8/16/32-bit element type, got {elem_bits}")
        shape, stride = (
            (((32, 4, 4), (4, 4)), ((1, 512, 128), (32, 2048)))
            if is_mn
            else (((4, 4), (32, 4, 4)), ((32, 2048), (1, 512, 128)))
        )
        return shape, stride, ("xor", 2, 4, 4)
    if sme_swizzle == SMESwizzle.Row8b:
        if elem_bits != 8:
            raise ValueError(f"Row8b shared layout requires 8-bit element type, got {elem_bits}")
        shape, stride = (
            (((8, 4, 4, 4), (4, 4)), ((1, 32, 128, 2048), (8, 512)))
            if is_mn
            else (((4, 4), (8, 4, 4, 4)), ((8, 512), (1, 32, 128, 2048)))
        )
        return shape, stride, ("mod", 2, 6, 2)
    if sme_swizzle == SMESwizzle.Row16b:
        if elem_bits != 16:
            raise ValueError(f"Row16b shared layout requires 16-bit element type, got {elem_bits}")
        shape, stride = (
            (((16, 16, 2), (2, 8)), ((1, 32, 4096), (16, 512)))
            if is_mn
            else (((2, 8), (16, 16, 2)), ((16, 512), (1, 32, 4096)))
        )
        return shape, stride, ("xor", 1, 7, 2)
    raise ValueError(f"invalid sme_swizzle={sme_swizzle}, expected 0..3 (NoSwizzle/Col/Row8b/Row16b)")


def _make_swizzle(kind, b, m, s):
    if kind == "xor":
        return static(SwizzleType.get(b, m, s))
    if kind == "mod":
        return static(ModSwizzleType.get(b, m, s))
    raise ValueError(kind)


def make_sme_shared_layout(sme_swizzle, elem_type, *, major=SMEMajor.MN):
    """Build the element-granularity SME shared-memory physical layout.

    This is the physical layout the SME load instruction writes for one
    ``SMESwizzle`` state; readback views built from it recover the logical
    ``(m, n)`` coordinates correctly.

    SME col/row swizzles are *byte*-granular: they permute whole bytes, act on
    byte addresses, and never reorder bits inside a byte. So the physical layout
    is assembled at **byte** granularity (the swizzle composed over the
    byte-recast layout) and then converted to element granularity by a single
    ``recast_layout(.., 8, elem_bits)`` -- a plain byte->element value
    reinterpret applied uniformly to both the swizzle and the layout strides.
    No per-swizzle bit/byte correction is needed.

    (Do NOT instead build the layout in *bit* granularity and recast by the full
    ``elem_bits``: that upcasts the swizzle base by ``log2(elem_bits)`` rather
    than ``log2(elem_bits/8)``, double-counting the byte width and scrambling
    data.)
    """
    sme_swizzle = int(sme_swizzle)
    elem_bits = _elem_bits(elem_type)
    is_mn = _normalize_major(major) == SMEMajor.MN

    shape, stride, swz = _sme_bit_spec(sme_swizzle, elem_bits, is_mn)
    # Assemble at byte granularity: layout strides bit->byte, swizzle as-is
    # (its base is already the byte-granular value).
    byte_layout = recast_layout(make_layout(shape, stride), 1, 8)
    if swz is not None:
        byte_layout = make_composed_layout(_make_swizzle(*swz), byte_layout)
    # Single byte->element value reinterpret of the whole composed layout.
    return recast_layout(byte_layout, 8, elem_bits)


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
