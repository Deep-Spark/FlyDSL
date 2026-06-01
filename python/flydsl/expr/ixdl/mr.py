# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""High-level API for the Iluvatar MR async copy (SME Load series).

Mirrors the FlyROCDL ``BufferCopy*`` factories, but targets the Iluvatar SME
``ixdl.cp_async.*`` op family. The copy atom is parameterised by a single
``sme_swizzle`` value matching CUTLASS ``IX11::SMESwizzle`` (see
``cute/arch/copy_ix11_desc.hpp``); the element dtype and ``Major::K/MN`` axis
are orthogonal and are carried elsewhere (``fly.copy_atom<..., elemBits>`` /
the kernel layout factory), not by this parameter.
"""

from ..._mlir._mlir_libs._mlirDialectsFlyIXDL import CopyOpMRAsyncCpType
from ..._mlir.dialects.fly import PointerType
from ..._mlir.dialects.fly_ixdl import TargetAddressSpace
from ..primitive import get_iter, get_layout, get_leaves, get_stride, make_ptr, make_view
from ..typing import Int32, Tensor


class SMESwizzle:
    """CUTLASS ``IX11::SMESwizzle`` 4-state enum (copy_ix11_desc.hpp)."""

    NoSwizzle = 0  # 32b row-major / INTER
    Col = 1  # col-major xor swizzle -> colxfb8 (also bf16/fp16 col)
    Row8b = 2  # 8b row-major mod/add swizzle -> rowxfb8 (Swizzle_Mod)
    Row16b = 3  # 16b row-major xor swizzle -> rowxfb16


def MRAsyncCp(sme_swizzle):
    """Create an Iluvatar MR async copy atom (SME Load series).

    Args:
        sme_swizzle: CUTLASS ``IX11::SMESwizzle`` 4-state value, which alone
            determines the SME builtin / IXDL op:
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
