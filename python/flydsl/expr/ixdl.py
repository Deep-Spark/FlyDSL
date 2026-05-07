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

from .._mlir._mlir_libs._mlirDialectsFlyIXDL import (
    MmaOpIX11_MMADType,
    CopyOpIX11_SMEType,
)
from .._mlir.dialects.fly import AddressSpace, PointerType, SwizzleType
from .._mlir.extras import types as T


# Keep these in sync with ``FlyIXDL_SMEMajor`` / ``FlyIXDL_SMECacheOp``
# in ``include/flydsl/Dialect/FlyIXDL/IR/Enums.td``.
SME_MAJOR_MN = 0
SME_MAJOR_K = 1

SME_CACHE_ALL = 0
SME_CACHE_BYPASS_L1 = 1
SME_CACHE_BYPASS_L2 = 2
SME_CACHE_BYPASS_L1L2 = 3

# Keep in sync with ``FlyIXDL_SMESwizzle`` in Enums.td.
SME_SWIZZLE_NONE = 0
SME_SWIZZLE_ROW_XFB16 = 1
SME_SWIZZLE_COL_XFB8 = 2

_SME_SWIZZLE_ALIASES = {
    "none": SME_SWIZZLE_NONE,
    "row_xfb16": SME_SWIZZLE_ROW_XFB16,
    "col_xfb8": SME_SWIZZLE_COL_XFB8,
    SME_SWIZZLE_NONE: SME_SWIZZLE_NONE,
    SME_SWIZZLE_ROW_XFB16: SME_SWIZZLE_ROW_XFB16,
    SME_SWIZZLE_COL_XFB8: SME_SWIZZLE_COL_XFB8,
}

_SME_MAJOR_ALIASES = {
    "mn": SME_MAJOR_MN,
    "k": SME_MAJOR_K,
    SME_MAJOR_MN: SME_MAJOR_MN,
    SME_MAJOR_K: SME_MAJOR_K,
}

_SME_CACHE_ALIASES = {
    "cache_all": SME_CACHE_ALL,
    "bypass_l1": SME_CACHE_BYPASS_L1,
    "bypass_l2": SME_CACHE_BYPASS_L2,
    "bypass_l1_l2": SME_CACHE_BYPASS_L1L2,
    SME_CACHE_ALL: SME_CACHE_ALL,
    SME_CACHE_BYPASS_L1: SME_CACHE_BYPASS_L1,
    SME_CACHE_BYPASS_L2: SME_CACHE_BYPASS_L2,
    SME_CACHE_BYPASS_L1L2: SME_CACHE_BYPASS_L1L2,
}


def _ir(ty):
    """Accept either a raw MLIR type or a flydsl DSL type wrapper."""
    return ty.ir_type if hasattr(ty, "ir_type") else ty


def _logical_shape_16x512b(elem_type, opname):
    ty = _ir(elem_type)
    elem_bits = int(getattr(ty, "width", 0))
    if elem_bits <= 0 or 512 % elem_bits != 0:
        raise ValueError(f"{opname}: unsupported element width {elem_bits}")
    return ty, [16, 512 // elem_bits]


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


def SMECopy(elem_type, shape, stride_byte, major="k", cache_op="cache_all",
            swizzle="none"):
    """Create an ``ivcore11`` SME (Shared-Memory-Engine) copy atom type.

    This atom models a single-thread async bulk copy from ``global`` to
    ``shared`` memory. One atom call transfers ``shape[0] * shape[1] * 64``
    bytes in the background; callers must issue ``cp.async.commit.group``
    and ``cp.async.wait.group`` to synchronize with consumers.

    Args:
        elem_type: Source element type. Must be 8/16/32-bit (i8, f16, bf16,
            f32, i32, ...).
        shape: A ``(shape0, shape1)`` element-space tile. The ixdl ->
            LLVM dispatch converts to hardware segments via
            ``segmentsMinor = shape1 * elementSize / 512``; that must
            equal 1 on ivcore11, so ``shape1 == 64 / sizeof(elem_type)``
            (e.g. ``32`` for bf16/f16, ``16`` for f32, ``64`` for i8).
            ``shape0`` picks the hardware row count and must be one of
            ``{1, 4, 8, 16, 32, 64}``.
        stride_byte: Stride of the source tensor along the *major* axis, in
            bytes. This is embedded in the SME descriptor (word[3]). For a
            row-major ``(M, K)`` tile copied K-major, use ``K *
            sizeof(elem_type)``. Must be statically known; passing dynamic
            strides is a follow-up.
        major: ``"k"`` (row-xfb family, default) or ``"mn"`` (col-xfb). The
            latter maps to ``transpose=1`` in ``ixdl.cp.async``.
        cache_op: One of ``"cache_all"`` (default), ``"bypass_l1"``,
            ``"bypass_l2"``, or ``"bypass_l1_l2"``. Integer values
            ``0..3`` are also accepted.
        swizzle: HW swizzle variant. One of ``"none"`` (default, plain
            ``sme_load_*x1b64``), ``"row_xfb16"`` (16-bit row-swizzle,
            A-operand feeding 16-bit MMA; requires ``elem_type`` to be
            a 16-bit float and ``major="k"``), or ``"col_xfb8"`` (16-bit
            column-swizzle, B-operand; requires 16-bit float and
            ``major="mn"``). Integer values ``0..2`` are also accepted.
            The swizzled variants are the high-performance path — their
            shared-memory layout is *not* plain row-major, so the
            consumer LDS read must use a matching MMA-aware view.
    """
    ty = _ir(elem_type)
    shape0, shape1 = shape
    try:
        mj = _SME_MAJOR_ALIASES[major]
    except KeyError as e:
        raise ValueError(f"unknown SME major: {major!r}") from e
    try:
        kop = _SME_CACHE_ALIASES[cache_op]
    except KeyError as e:
        raise ValueError(f"unknown SME cache_op: {cache_op!r}") from e
    try:
        sw = _SME_SWIZZLE_ALIASES[swizzle]
    except KeyError as e:
        raise ValueError(f"unknown SME swizzle: {swizzle!r}") from e
    return CopyOpIX11_SMEType.get(ty, int(shape0), int(shape1),
                                  int(stride_byte), int(mj), int(kop),
                                  int(sw))


def cp_async_commit_group(loc=None, ip=None):
    """Commit the current IXDL cp.async group."""
    from .primitive import cp_async_commit_group as _cp_async_commit_group

    return _cp_async_commit_group(loc=loc, ip=ip)


def cp_async_wait_group(n=0, loc=None, ip=None):
    """Wait until at most ``n`` IXDL cp.async groups remain pending."""
    from .primitive import cp_async_wait_group as _cp_async_wait_group

    return _cp_async_wait_group(n, loc=loc, ip=ip)


def pipebar_req(barrier_id=0, loc=None, ip=None):
    """Issue an IXDL pipeline-barrier arrive/request."""
    from .primitive import pipebar_req as _pipebar_req

    return _pipebar_req(barrier_id, loc=loc, ip=ip)


def pipebar_wait(barrier_id=0, loc=None, ip=None):
    """Wait on an IXDL pipeline barrier."""
    from .primitive import pipebar_wait as _pipebar_wait

    return _pipebar_wait(barrier_id, loc=loc, ip=ip)


def _encode_sl_waitcnt(*, vm=False, sm=False, lm=False, g2s=False, s2g=False,
                       mba=False, mbt=False, vm_cnt=0, sm_cnt=0, lm_cnt=0,
                       g2s_cnt=0, s2g_cnt=0, mba_cnt=0, mbt_cnt=0):
    cnt = 0
    cnt |= int(bool(vm)) << 0
    cnt |= int(bool(sm)) << 1
    cnt |= int(bool(lm)) << 2
    cnt |= int(bool(g2s)) << 3
    cnt |= int(bool(s2g)) << 4
    cnt |= int(bool(mba)) << 5
    cnt |= int(bool(mbt)) << 6
    cnt |= (int(vm_cnt) & 0x3F) << 7
    cnt |= (int(sm_cnt) & 0x3F) << 13
    cnt |= (int(lm_cnt) & 0x0F) << 19
    cnt |= (int(g2s_cnt) & 0x3F) << 23
    cnt |= (int(s2g_cnt) & 0x1F) << 29
    cnt |= (int(mba_cnt) & 0x0F) << 34
    cnt |= (int(bool(mbt_cnt)) & 0x01) << 38
    return cnt


def sl_waitcnt(cnt=None, *, vm=False, sm=False, lm=False, g2s=False, s2g=False,
               mba=False, mbt=False, vm_cnt=0, sm_cnt=0, lm_cnt=0, g2s_cnt=0,
               s2g_cnt=0, mba_cnt=0, mbt_cnt=0, loc=None, ip=None):
    """Issue ``ixdl.sl.waitcnt``.

    Either pass a pre-encoded ``cnt`` directly, or use the keyword fields to
    build one. Example: ``sl_waitcnt(g2s=True, g2s_cnt=0, lm=True, lm_cnt=0)``.
    """
    from .primitive import sl_waitcnt as _sl_waitcnt

    if cnt is None:
        cnt = _encode_sl_waitcnt(
            vm=vm, sm=sm, lm=lm, g2s=g2s, s2g=s2g, mba=mba, mbt=mbt,
            vm_cnt=vm_cnt, sm_cnt=sm_cnt, lm_cnt=lm_cnt, g2s_cnt=g2s_cnt,
            s2g_cnt=s2g_cnt, mba_cnt=mba_cnt, mbt_cnt=mbt_cnt,
        )
    return _sl_waitcnt(cnt, loc=loc, ip=ip)


def sched_barrier(loc=None, ip=None):
    """Prevent the Iluvatar backend scheduler from moving instructions across this point.

    Lowers directly to the target intrinsic ``llvm.bi.sch.barrier``. This is
    useful in hand-scheduled IXDL hot loops where the DSL order intentionally
    interleaves SLB/LDS loads with MMAD instructions.
    """
    from .._mlir.dialects import llvm as _llvm

    return _llvm.call_intrinsic(
        None,
        "llvm.bi.sch.barrier",
        [],
        [],
        [],
        loc=loc,
        ip=ip,
    )


def SMELayout16x512b(elem_type, transpose=False, target_shape=None):
    """Create the CUTLASS-matched shared-memory layout for a 16x512b SME tile.

    Ported from ``origin/kefan.cao/dev`` (commit ``c39c588``). Keeps the
    atom layout in *bits* (mirroring ``Layout_SME_I_16x512b_*_Atom_Bits``
    in ``cutlass/include/cute/atom/copy_traits_ix11_sme.hpp``), lets
    ``recast_layout`` fold the element width out, and delegates the
    extent-to-CTA expansion to ``tile_to_shape`` so callers do not have to
    hand-nest three levels of tuple-shape. The swizzle constants match the
    upstream FlyDSL convention, which differs from the raw CUTE bit-level
    parameters by one bit in ``M`` — so do not "derive" them; leave them
    verbatim.

    Args:
        elem_type: Element type (bf16, f16, ...).
        transpose: ``False`` (default) picks the K-major row-swizzle
            variant matching ``bi_sme_load_16x1b64_rowxfb16``. ``True``
            picks the MN-major col-swizzle variant (``colxfb8``) used on
            the B operand.
        target_shape: Optional ``(rows, cols)`` override in elements. When
            omitted the layout covers exactly one SME atom
            (``(16, 512 / elem_bits)``). Pass the full CTA tile
            (e.g. ``(BM, BK)``) to have ``tile_to_shape`` replicate the
            atom across the block; the values must be multiples of the
            atom extents.
    """
    from .primitive import (
        make_composed_layout,
        make_int_tuple,
        make_layout,
        recast_layout,
        static,
        tile_to_shape,
    )

    ty, atom_shape = _logical_shape_16x512b(elem_type, "SMELayout16x512b")
    if transpose:
        swizzle = static(SwizzleType.get(2, 3, 4))
        order = make_int_tuple((0, 1))
        outer_bits = make_layout(((4, 4), (32, 4, 4)),
                                 ((32, 2048), (1, 512, 128)))
    else:
        swizzle = static(SwizzleType.get(1, 6, 2))
        order = make_int_tuple((1, 0))
        outer_bits = make_layout(((2, 8), (16, 16, 2)),
                                 ((16, 512), (1, 32, 4096)))

    shape = list(target_shape) if target_shape is not None else atom_shape
    for i, (extent, atom_extent) in enumerate(zip(shape, atom_shape)):
        if extent % atom_extent != 0:
            raise ValueError(
                f"SMELayout16x512b: target_shape[{i}]={extent} is not a "
                f"multiple of atom extent {atom_extent}"
            )

    offset = make_int_tuple(0)
    shape_val = make_int_tuple(tuple(shape))
    outer = tile_to_shape(
        recast_layout(outer_bits, 1, int(ty.width)),
        shape_val,
        order,
    )
    return make_composed_layout(swizzle, offset, outer)


def SMEView16x512b(base_ptr, elem_type, transpose=False, elem_offset=0,
                   target_shape=None):
    """Create a swizzled shared-memory view for a 16x512b SME tile.

    Ported from ``origin/kefan.cao/dev`` (commit ``c39c588``). ``elem_offset``
    is in elements of ``elem_type``; callers can combine multiple tiles by
    giving each its own offset into the same shared-memory base pointer.
    ``target_shape`` mirrors :func:`SMELayout16x512b`.
    """
    from .primitive import add_offset, make_view, recast_iter

    ty, _ = _logical_shape_16x512b(elem_type, "SMEView16x512b")
    smem_ptr_ty = PointerType.get(ty, AddressSpace.Shared)
    smem_ptr = recast_iter(smem_ptr_ty, base_ptr)
    if elem_offset:
        smem_ptr = add_offset(smem_ptr, elem_offset)
    return make_view(smem_ptr, SMELayout16x512b(
        ty, transpose=transpose, target_shape=target_shape,
    ))


__all__ = [
    "MmaOpIX11_MMADType",
    "CopyOpIX11_SMEType",
    "MMAD",
    "SMECopy",
    "cp_async_commit_group",
    "cp_async_wait_group",
    "pipebar_req",
    "pipebar_wait",
    "sl_waitcnt",
    "sched_barrier",
    "SMELayout16x512b",
    "SMEView16x512b",
    "SME_MAJOR_MN",
    "SME_MAJOR_K",
    "SME_CACHE_ALL",
    "SME_CACHE_BYPASS_L1",
    "SME_CACHE_BYPASS_L2",
    "SME_CACHE_BYPASS_L1L2",
    "SME_SWIZZLE_NONE",
    "SME_SWIZZLE_ROW_XFB16",
    "SME_SWIZZLE_COL_XFB8",
]
