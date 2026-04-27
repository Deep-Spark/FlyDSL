from ..._mlir import ir
from ..._mlir.dialects import fly as _fly_dialect
from ..._mlir.dialects.fly import (
    AddressSpace,
    LayoutType,
    MemRefType,
    PointerType,
    SwizzleType,
)
from ..._mlir.dialects.fly_ixdl import (
    MmaAtomIvcore11_MMADType,
    CopyOpIvcore11_SMELoadType,
    CopyOpIvcore11_SLBLoadType,
    CopyOpIvcore11_DescStoreType,
)


def MMAD(m, n, k, elem_ty_ab, elem_ty_acc=None):
    """Create MmaAtomIvcore11_MMAD type for ivcore11 TCU MMAD instruction.

    Args:
        m, n, k: Tile dimensions (must be 16, 16, 16)
        elem_ty_ab: Input element type (f16, bf16, or i8)
        elem_ty_acc: Accumulator element type (default f32)
    """
    ty_ab = elem_ty_ab.ir_type if hasattr(elem_ty_ab, "ir_type") else elem_ty_ab
    if elem_ty_acc is None:
        ty_acc = ir.F32Type.get()
    else:
        ty_acc = elem_ty_acc.ir_type if hasattr(elem_ty_acc, "ir_type") else elem_ty_acc
    return MmaAtomIvcore11_MMADType.get(m, n, k, ty_ab, ty_ab, ty_acc)


def SMECopy16x(bit_size=32):
    """Create CopyAtom for async G2S via SME sme_load_16x1b64."""
    return CopyOpIvcore11_SMELoadType.get(bit_size)


def SLBLoad(bit_size=32):
    """Create CopyAtom for per-thread S2R shared memory load."""
    return CopyOpIvcore11_SLBLoadType.get(bit_size)


def DescStore(bit_size=32):
    """Create CopyAtom for per-thread R2G descriptor store."""
    return CopyOpIvcore11_DescStoreType.get(bit_size)


# ═══════════════════════════════════════════════════════════════════
# Shared-memory tensor with SME swizzle layout
# ═══════════════════════════════════════════════════════════════════


# ── Auto-derive swizzle from MMA atom + dtype ────────────────────────
#
# Maps "(atom_kind, dtype_bit_width) -> element-space (mask, base, shift)".
# The atom_kind string is matched as a substring against the MLIR type of
# whatever the user passes as ``for_mma`` (TiledMma / MmaAtom / raw atom
# Type). Currently we only know about ivcore11 MMAD on 16-bit operands;
# extend this table when new atoms or dtypes need a fixed swizzle.
_MMA_SWIZZLE_TABLE = {
    # ivcore11 MMAD f16/bf16 16x16x16 → SME rowxfb16 swizzle, element-space
    ("ivcore11.mmad", 16): (1, 6, 2),
}


def _derive_swizzle_for_mma(for_mma, dtype):
    """Return an element-space ``(mask, base, shift)`` triple for the
    swizzle that pairs the given ``for_mma`` atom with the SLB layout used
    by ``make_smem_tile``. Raises ``NotImplementedError`` if the
    (atom, dtype) combination is not in the dispatch table.
    """
    obj = for_mma
    if hasattr(obj, "value") and hasattr(obj.value, "type"):
        ty = obj.value.type
    elif hasattr(obj, "type"):
        ty = obj.type
    else:
        ty = obj
    type_str = str(ty).lower()

    bit_width = (getattr(dtype, "bit_size", None)
                 or getattr(dtype, "width", None))

    for atom_substr, atom_bits in _MMA_SWIZZLE_TABLE:
        if atom_substr in type_str and atom_bits == bit_width:
            return _MMA_SWIZZLE_TABLE[(atom_substr, atom_bits)]

    raise NotImplementedError(
        f"make_smem_tile: no auto-swizzle entry for atom={ty} dtype={dtype}. "
        f"Pass an explicit `swizzle=` (element-space) or `swizzle_byte=` "
        f"(byte-space) instead, or extend _MMA_SWIZZLE_TABLE.")


def make_smem_tile(M, K, dtype, swizzle=None, swizzle_byte=None, *,
                   for_mma=None, loc=None, ip=None):
    """Allocate a shared-memory tile and wrap it in a CuTe MemRef with a
    ComposedLayout encoding the SME rowxfb16 swizzle.

    The allocation is done via ``fly.memref.alloca`` with
    ``AddressSpace::Shared``, which the IXDL lowering maps to an
    ``llvm.mlir.global`` in address space 3 (SLB).

    Specifying the swizzle: ``for_mma`` vs ``swizzle`` vs ``swizzle_byte``
    ---------------------------------------------------------------------
    Three mutually-exclusive ways, from most-recommended to lowest-level:

    * ``for_mma=tiled_mma`` (or ``mma_atom``) — auto-derive the swizzle
      from the MMA atom kind and ``dtype``. Most users want this; matches
      the way CUTLASS CuTe pairs SmemLayoutAtom with MMA atoms.
    * ``swizzle_byte=(mask, base, shift)`` — CUTLASS-style **byte-address**
      space parameters, internally rewritten to element-space by
      subtracting ``log2(sizeof(dtype))`` from ``base``. Use when porting
      Swizzle parameters from CUTLASS docs / SLB-bank diagrams.
    * ``swizzle=(mask, base, shift)`` — raw FlyDSL **element-index** space
      parameters (the unit ``layoutCrd2Idx`` produces). Lowest level,
      dtype-specific.

    Why the two spaces differ
    -------------------------
    FlyDSL ``SwizzleAttr<mask, base, shift>`` is applied to the
    **element-index** offset produced by ``layoutCrd2Idx``. CUTLASS source
    code does the same, but most CUTLASS *tutorials / bank-conflict
    diagrams* describe Swizzle bits in **byte-address** space (because
    banks are 4-byte and that picture is more intuitive). The conversion
    between the two, for the same physical swizzle::

        base_element = base_byte - log2(sizeof(elem))

    For f16:  ``<1, 6, 2>`` (element) == ``<1, 7, 2>`` (byte).
    For fp32: ``<1, 5, 2>`` (element) == ``<1, 7, 2>`` (byte).

    Args:
        M: number of rows (e.g. 16).
        K: number of columns (e.g. 32 f16 elements).
        dtype: FlyDSL numeric type (e.g. ``fx.Float16``).
        swizzle: ``(mask, base, shift)`` in **element-index** space, or
                 ``None``.
        swizzle_byte: ``(mask, base, shift)`` in **byte-address** space
                 (CUTLASS-style). Mutually exclusive with ``swizzle``.
        for_mma: a ``TiledMma`` / ``MmaAtom`` / raw atom Type whose atom
                 kind + ``dtype`` will be used to look up the right
                 swizzle automatically. Mutually exclusive with both
                 ``swizzle`` and ``swizzle_byte``.

    Returns:
        An ``ir.Value`` of type ``fly.memref<...>`` backed by shared
        memory, whose layout encodes the SME vertical-packing + XOR
        swizzle.
    """
    explicit = sum(x is not None for x in (swizzle, swizzle_byte, for_mma))
    if explicit > 1:
        raise ValueError(
            "make_smem_tile: pass at most one of `swizzle` (element-space), "
            "`swizzle_byte` (byte-space, CUTLASS convention), or `for_mma` "
            "(auto-derive from atom + dtype).")

    if for_mma is not None:
        swizzle = _derive_swizzle_for_mma(for_mma, dtype)
    elif swizzle_byte is not None:
        bit_width = (getattr(dtype, "bit_size", None)
                     or getattr(dtype, "width", None))
        if bit_width is None:
            raise TypeError(
                "make_smem_tile: swizzle_byte requires a FlyDSL numeric type "
                "(e.g. fx.Float16) with a `.width` attribute. Got a raw MLIR "
                f"type: {dtype!r}. Use the element-space `swizzle=` arg "
                "instead, or wrap with fx.<dtype>.")
        elem_bytes = bit_width // 8
        if elem_bytes <= 0 or (elem_bytes & (elem_bytes - 1)) != 0:
            raise ValueError(
                f"make_smem_tile: cannot infer log2(elem_bytes) for dtype "
                f"with byte-size {elem_bytes}")
        log2_elem = elem_bytes.bit_length() - 1
        b_mask, b_base, b_shift = swizzle_byte
        if b_base < log2_elem:
            raise ValueError(
                f"make_smem_tile: swizzle_byte base={b_base} cannot be "
                f"converted: result would be negative for dtype with "
                f"elem_bytes={elem_bytes}")
        swizzle = (b_mask, b_base - log2_elem, b_shift)
    from ..primitive import (
        make_layout, make_int_tuple, make_view, get_iter, memref_alloca,
    )

    elem_ty = dtype.ir_type if hasattr(dtype, "ir_type") else dtype
    total_elems = M * K

    # --- 1. Allocate SMEM via fly.memref.alloca with AddressSpace::Shared ---
    flat_layout_val = make_layout(total_elems, 1, loc=loc, ip=ip)
    flat_layout_ty = LayoutType(flat_layout_val.type)
    smem_type = MemRefType.get(
        elem_ty, flat_layout_ty,
        address_space=int(AddressSpace.Shared),
    )
    smem = memref_alloca(smem_type, flat_layout_val, loc=loc, ip=ip)

    # --- 2. Build the SME swizzle layout ---
    # SME rowxfb16 vertical-packing: each 16-row sub-tile has sme16x layout
    # with DWord pairs from adjacent rows.
    #
    # For M=16 (single sub-tile):
    #   shape  = ((2, 8), (K//2, 2))
    #   stride = ((1, K), (2, 8*K))
    #
    # For M>16 (M must be a multiple of 16):
    #   shape  = ((2, 8, M//16), (K//2, 2))
    #   stride = ((1, K, 16*K),  (2, 8*K))
    #   Each 16-row sub-tile occupies 16*K contiguous elements with
    #   independent sme16x packing, compatible with per-sub-tile SME loads.
    assert M % 16 == 0, f"make_smem_tile: M={M} must be a multiple of 16"
    assert K % 2 == 0, f"make_smem_tile: K={K} must be even"

    m_lo = 2
    m_hi = 8
    m_sub = M // 16
    k_lo = K // 2
    k_hi = 2
    sub_tile_elems = 16 * K

    if m_sub == 1:
        outer_layout = make_layout(
            ((m_lo, m_hi), (k_lo, k_hi)),
            ((1, K), (2, m_hi * K)),
            loc=loc, ip=ip,
        )
    else:
        outer_layout = make_layout(
            ((m_lo, m_hi, m_sub), (k_lo, k_hi)),
            ((1, K, sub_tile_elems), (2, m_hi * K)),
            loc=loc, ip=ip,
        )

    # --- 3. Extract iterator and create view with composed layout ---
    smem_iter = get_iter(smem, loc=loc, ip=ip)

    if swizzle is not None:
        mask, base, shift = swizzle
        swizzle_ty = SwizzleType.get(mask, base, shift)
        swizzle_val = _fly_dialect.static(swizzle_ty, loc=loc, ip=ip)

        offset_val = make_int_tuple(0, loc=loc, ip=ip)

        composed_layout = _fly_dialect.make_composed_layout(
            swizzle_val, offset_val, outer_layout, loc=loc, ip=ip
        )
        return make_view(smem_iter, composed_layout, loc=loc, ip=ip)
    else:
        return make_view(smem_iter, outer_layout, loc=loc, ip=ip)
