# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""High-level API for Iluvatar CQ (ivcore30) TCU MMA and SMEX copy atoms.

Exposes ``CQMma`` for CQ TCU matrix-multiply-accumulate atoms (base + long-mtx)
and ``CQSmexCp`` for SMEX G2S (mtx / plain) with runtime RowMask / ColMask /
optional Pred state. ``CQMtxLoadn`` provides the matching SmexMtx S2R path.
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

    Supported families (``(M,N)`` in {16x16, 32x32, 16x64, 64x16}):

    - f16/bf16 -> f32, ``K=16``, A and B must match
    - i8/ui8 -> i32, ``K=32``, A and B must match signedness
    - f8E4M3/f8E5M2 -> f32 or f16; A/B may mix; ``K=32``
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
    """Create a CQ SMEX G2S copy atom.

    ``rows`` selects the hardware shape: 4 (256B), 16 (1024B), or 64 (4096B).
    ``layout`` is ``\"mtx\"`` / ``SMEX_LAYOUT_MTX`` (``smex_loadn_*_mtx``)
    or ``\"plain\"`` / ``SMEX_LAYOUT_PLAIN`` (``smex_load_*``).

    Runtime state defaults: ``row_mask``/``col_mask`` all-1s; pred disabled
    (non-pred IXDL op). Set ``row_mask``/``col_mask``/``pred`` through
    ``atom.set_value(...)``. Passing an i1 register tensor as ``fx.copy(...,
    pred=...)`` selects the ``*.pred.*`` IXDL path for that copy.

    ``row_mask`` / ``col_mask`` are arbitrary bit prefixes (bit i enables DW i
    for ``col_mask``). The SME gmem global row stride (``stride_byte`` from
    :func:`make_sme_gmem_tensor`) must be 16B-aligned; hardware silently
    truncates the low 4 bits otherwise.
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


def CQMtxLoadn(elem_type, *, pattern: str | int = "loadn16", direction: str | int = "row"):
    """Create the CQ SmexMtx shared-to-register matrix-load atom.

    ``pattern`` records G2S tile-height pairing only: ``loadn16`` with a
    16-row ``CQSmexCp(layout="mtx")``, ``loadn64`` with a 64-row copy. It is
    not x1 vs x2 (register width) and does not select a different mtx_loadn
    opcode -- both patterns emit the same ``ixdl.mtx_loadn_*x2`` (64 bits per
    lane). Cover a 64-row shared footprint with multiple atom calls that vary
    EmPart / slot base, not a wider single instruction.

    ``direction="row"`` produces an MMA A fragment; ``direction="col"``
    produces a B fragment.

    SmexMtx and LegacySme are incompatible shared-buffer contracts. The
    SmexMtx path bypasses the legacy SME byte swizzle because SMEX writes the
    matrix format and ``mtx_loadn`` consumes its built-in mtx-index swizzle.
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
