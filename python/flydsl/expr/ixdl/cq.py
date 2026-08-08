# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""High-level API for Iluvatar CQ (ivcore30) TCU MMA and SMEX copy atoms.

Mirrors ``mr.py`` factory shape for the CQ path:

- ``CQMma`` -- TCU MMA (base + long-mtx), analogous to ``MRMma``
- ``CQSmexCp`` / ``CQSmexCpMtx`` / ``CQSmexCpPlain`` -- SMEX G2S async copy,
  analogous to ``MRAsyncCp*`` (CQ uses SMEX, not SME swizzle states)
- ``CQMtxLoadn`` -- SmexMtx S2R matrix-load, pairs with ``CQSmexCp(layout="mtx")``

``CQSmexCp`` carries runtime RowMask / ColMask / optional Pred state.
"""

from ..._mlir import ir
from ..._mlir._mlir_libs._mlirDialectsFlyIXDL import (
    CopyOpCQMtxLoadnType,
    CopyOpCQSmexCpType,
    MmaOpCQMmaType,
)

# cq.smex_cp layout enum (matches C++ / TableGen).
SMEX_LAYOUT_PLAIN = 0
SMEX_LAYOUT_MTX = 1

MTX_LOAD_PATTERN_LOADN16 = 0
MTX_LOAD_PATTERN_LOADN64 = 1
MTX_GATHER_ROW = 0
MTX_GATHER_COL = 1


def _to_ir_type(t) -> "ir.Type":
    """Coerce a FlyDSL numeric type / ir.Type to an ``ir.Type``."""
    if isinstance(t, ir.Type):
        return t
    if hasattr(t, "ir_type"):
        return t.ir_type
    raise TypeError(f"expected a NumericType or ir.Type, got {t!r}")


def CQMma(m, n, k, elem_ty_a, elem_ty_b, elem_ty_acc):
    """Create an Iluvatar CQ (ivcore30) TCU MMA atom (``D = A*B + C``).

    Warp-collective (64 lanes) tensor-core MMA. The returned atom type is passed
    to ``fx.make_mma_atom`` and drives ``fx.gemm`` / the A/B/C register-fragment
    TV layouts (``make_fragment_A/B/C``).

    Args:
        m, n: MMA output tile ``(M, N)``. Supported pairs: 16x16, 32x32, 16x64,
            64x16. 16x16 is the base tile; the others are the FeatureLongMtx
            (long-mtx) tiles.
        k: MMA contraction depth. 16 for f16/bf16, 32 for i8/ui8 and f8.
        elem_ty_a, elem_ty_b: multiplicand element types (``fx.*`` or ``ir.Type``).
        elem_ty_acc: accumulator (and result D) element type.

    Supported dtype families:

    - f16/bf16 -> f32, ``K=16``; A and B must match.
    - i8/ui8 -> i32, ``K=32``; A and B must match signedness.
    - f8E4M3/f8E5M2 -> f32 or f16, ``K=32``; A and B may mix the two f8 formats.

    Register fragments are consumed by ``CQMtxLoadn`` (row gather -> A, column
    gather -> B) when the operands come from a ``CQSmexCp(layout="mtx")`` tile.

    Returns:
        A ``MmaOpCQMmaType`` for ``fx.make_mma_atom``.
    """
    return MmaOpCQMmaType.get(
        int(m),
        int(n),
        int(k),
        _to_ir_type(elem_ty_a),
        _to_ir_type(elem_ty_b),
        _to_ir_type(elem_ty_acc),
    )


def CQSmexCp(*, rows: int = 16, layout: str | int = "mtx"):
    """Create a CQ SMEX global-to-shared (G2S) async copy atom.

    CQ analogue of ``MRAsyncCp``: the CQ SMEX engine, not an SME swizzle state,
    moves a global tile into shared. Wrap with ``fx.make_copy_atom(atom, dtype)``
    and issue via ``fx.copy`` after building a tiled copy. Completion is ordered
    by ``cp_async_commit_group`` / ``cp_async_wait_group`` (see ``sync.py``).

    Args:
        rows: SMEX tile height, selecting the hardware transfer shape --
            4 (256 B), 16 (1024 B), or 64 (4096 B).
        layout: shared destination format.
            - ``"mtx"`` / ``SMEX_LAYOUT_MTX`` -> ``smex_loadn_*_mtx``: writes the
              CQ matrix format that ``CQMtxLoadn`` reads back (pair 16-row with
              ``loadn16``, 64-row with ``loadn64``). This is the SmexMtx contract.
            - ``"plain"`` / ``SMEX_LAYOUT_PLAIN`` -> ``smex_load_*``: linear row
              copy with no matrix-index remap.

    Runtime state (RowMask / ColMask / Pred):
        Defaults are ``row_mask`` / ``col_mask`` all-1s with predication disabled
        (the non-pred IXDL op is emitted). Narrow the transfer by supplying masks
        through ``atom.set_value({"row_mask": ..., "col_mask": ...})``. Passing an
        i1 register tensor as ``fx.copy(..., pred=...)`` selects the ``*.pred.*``
        IXDL path for that copy. ``row_mask`` / ``col_mask`` are arbitrary bit
        prefixes (bit i enables DW i for ``col_mask``).

        The SME gmem global row stride (``stride_byte`` from
        :func:`make_sme_gmem_tensor`) must be 16B-aligned; hardware silently
        truncates the low 4 bits otherwise.

    Returns:
        A ``CopyOpCQSmexCpType`` for ``fx.make_copy_atom``.
    """
    if isinstance(layout, str):
        key = layout.lower()
        if key == "mtx":
            layout_i = SMEX_LAYOUT_MTX
        elif key == "plain":
            layout_i = SMEX_LAYOUT_PLAIN
        else:
            raise ValueError(f"layout must be 'mtx' or 'plain', got {layout!r}")
    else:
        layout_i = int(layout)
    if layout_i not in (SMEX_LAYOUT_MTX, SMEX_LAYOUT_PLAIN):
        raise ValueError(f"layout must be 'mtx' or 'plain', got {layout!r}")
    return CopyOpCQSmexCpType.get(int(rows), layout_i)


# Convenience aliases fixing the layout (rows still overridable), mirroring the
# per-swizzle MRAsyncCp* aliases. CQSmexCpMtx is the SmexMtx G2S that pairs with
# CQMtxLoadn; CQSmexCpPlain is the linear row copy.
CQSmexCpMtx = lambda rows=16: CQSmexCp(rows=rows, layout=SMEX_LAYOUT_MTX)
CQSmexCpPlain = lambda rows=16: CQSmexCp(rows=rows, layout=SMEX_LAYOUT_PLAIN)


def CQMtxLoadn(elem_type, *, pattern: str | int = "loadn16", direction: str | int = "row"):
    """Create the CQ SmexMtx shared-to-register (S2R) matrix-load atom.

    Warp-collective load that gathers a shared matrix tile into per-lane MMA
    register fragments, ready to feed ``CQMma`` through ``make_fragment_A/B``.
    The atom emits the x2 IXDL op (``ixdl.mtx_loadn_*x2``), returning 64 bits per
    lane. Wrap with ``fx.make_copy_atom(atom, dtype)`` and issue via ``fx.copy``.

    Args:
        elem_type: fragment element type; must be 8-bit or 16-bit (its bit width
            is read from ``.width`` / ``.ir_type.width``). Matches the copy-atom
            value type and the ``CQMma`` multiplicand dtype.
        pattern: G2S tile-height pairing only -- ``"loadn16"`` with a 16-row
            ``CQSmexCp(layout="mtx")``, ``"loadn64"`` with a 64-row copy. Not
            x1 vs x2 (register width) and not a different mtx_loadn opcode;
            both emit the same ``*x2`` load. Cover a 64-row shared footprint
            with multiple atom calls that vary EmPart / slot base.
        direction: gather orientation. ``"row"`` produces the CQ MMA A fragment;
            ``"col"`` produces the B fragment.

    SmexMtx vs LegacySme:
        SmexMtx and LegacySme are incompatible shared-buffer contracts; a shared
        buffer must use one contract for both its G2S write and S2R read, never a
        mix. The SmexMtx path bypasses the legacy SME byte swizzle: SMEX writes
        the matrix format and ``mtx_loadn`` consumes its built-in mtx-index
        swizzle in hardware. "Bypass" means the legacy byte-swizzle transform is
        skipped, not that the data is stored unswizzled.

    Returns:
        A ``CopyOpCQMtxLoadnType`` for ``fx.make_copy_atom``.
    """
    if isinstance(pattern, str):
        patterns = {
            "loadn16": MTX_LOAD_PATTERN_LOADN16,
            "loadn64": MTX_LOAD_PATTERN_LOADN64,
        }
        try:
            pattern_i = patterns[pattern.lower()]
        except KeyError:
            raise ValueError(f"pattern must be 'loadn16' or 'loadn64', got {pattern!r}") from None
    else:
        pattern_i = int(pattern)
    if pattern_i not in (MTX_LOAD_PATTERN_LOADN16, MTX_LOAD_PATTERN_LOADN64):
        raise ValueError(f"pattern must be 'loadn16' or 'loadn64', got {pattern!r}")

    if isinstance(direction, str):
        directions = {"row": MTX_GATHER_ROW, "col": MTX_GATHER_COL}
        try:
            direction_i = directions[direction.lower()]
        except KeyError:
            raise ValueError(f"direction must be 'row' or 'col', got {direction!r}") from None
    else:
        direction_i = int(direction)
    if direction_i not in (MTX_GATHER_ROW, MTX_GATHER_COL):
        raise ValueError(f"direction must be 'row' or 'col', got {direction!r}")

    if hasattr(elem_type, "width"):
        elem_bits = int(elem_type.width)
    elif hasattr(elem_type, "ir_type") and hasattr(elem_type.ir_type, "width"):
        elem_bits = int(elem_type.ir_type.width)
    else:
        raise TypeError(f"elem_type must carry bit width information, got {elem_type!r}")
    if elem_bits not in (8, 16):
        raise ValueError(f"CQMtxLoadn requires an 8-bit or 16-bit element type, got {elem_bits}")

    return CopyOpCQMtxLoadnType.get(pattern_i, direction_i, elem_bits)
