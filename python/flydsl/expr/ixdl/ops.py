# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Iluvatar ALU / descriptor-store / atomic helpers (non-sync).

Most wrappers emit Iluvatar LLVM intrinsics via ``llvm.call_intrinsic``.
``atomic_cas`` lowers through ``llvm.cmpxchg`` (same ptr round-trip as
``stp.vs``). Hardware CAS is width-based; ``llvm.cmpxchg`` needs integer
operands, so non-integer values of a supported width are bitcast to the
matching signless integer, then bitcast back on the result. Call sites
pick a width the target supports (16/32-bit all gens; 64-bit on Blazer+).
Synchronization helpers live in :mod:`flydsl.expr.ixdl.sync`.
"""

from ..._mlir.dialects import llvm as _llvm
from .. import arith as _arith
from ..typing import T

# Bit-width -> ``llvm.bi.stp.vs[.pred].*`` (b128 = v4i32).
_STP_VS = {
    8: "llvm.bi.stp.vs.i8",
    16: "llvm.bi.stp.vs.i16",
    32: "llvm.bi.stp.vs.i32",
    64: "llvm.bi.stp.vs.i64",
    128: "llvm.bi.stp.vs.v4i32",
}
_STP_VS_PRED = {
    8: "llvm.bi.stp.vs.pred.i8",
    16: "llvm.bi.stp.vs.pred.i16",
    32: "llvm.bi.stp.vs.pred.i32",
    64: "llvm.bi.stp.vs.pred.i64",
    128: "llvm.bi.stp.vs.pred.v4i32",
}


def byte_permute(a, b, sel: int):
    """Emit ``llvm.nvvm.prmt`` byte permute.

    ``sel`` is a compile-time nibble selector: bytes 0-3 from ``a``, 4-7 from ``b``.
    Prefer this over a shift/mask expansion (~16 ALU ops per call).
    """
    from ..numeric import Int32

    result = _llvm.call_intrinsic(
        T.i32,
        "llvm.nvvm.prmt",
        [
            _arith.unwrap(a),
            _arith.unwrap(b),
            _arith.unwrap(_arith.constant(int(sel), type=T.i32)),
        ],
        [],
        [],
    )
    return Int32(result)


def readfirstlane(val):
    """Broadcast lane 0's value across the warp (``llvm.bi.readfirstlane``).

    Mirrors ROCm ``rocdl.readfirstlane`` for Iluvatar: turn a potentially
    divergent VGPR into a warp-uniform value that can live in an SGPR (addresses,
    sequence lengths, block-table entries). Supports ``i32`` and ``i64``.
    """
    from ..._mlir.ir import IntegerType
    from ..numeric import Int32, Int64

    raw = _arith.unwrap(val)
    width = IntegerType(raw.type).width
    if width == 32:
        return Int32(_llvm.call_intrinsic(T.i32, "llvm.bi.readfirstlane", [raw], [], []))
    if width == 64:
        return Int64(_llvm.call_intrinsic(T.i64, "llvm.bi.readfirstlane.i64", [raw], [], []))
    raise TypeError(f"readfirstlane supports i32/i64, got width={width}")


def _dot4_call(name, a, b, c):
    from ..numeric import Int32

    return Int32(
        _llvm.call_intrinsic(
            T.i32,
            name,
            [_arith.unwrap(a), _arith.unwrap(b), _arith.unwrap(c)],
            [],
            [],
        )
    )


def idot4(a, b, c):
    """Signed int8x4 dot-accumulate: r = sum_i(a[i]*b[i]) + c.

    Emits ``llvm.bi.idot4``. ``a`` / ``b`` are packed i32 (four signed int8
    lanes); ``c`` and the result are i32.
    """
    return _dot4_call("llvm.bi.idot4", a, b, c)


def udot4(a, b, c):
    """Unsigned int8x4 dot-accumulate: r = sum_i(a[i]*b[i]) + c.

    Emits ``llvm.bi.udot4``. ``a`` / ``b`` are packed i32 (four unsigned int8
    lanes); ``c`` and the result are i32.
    """
    return _dot4_call("llvm.bi.udot4", a, b, c)


def _llvm_ptr(ptr, *, addrspace: int | None = 1):
    """Materialize ``!fly.ptr`` as ``!llvm.ptr`` / ``!llvm.ptr<AS>`` via ptrtoint/inttoptr.

    Same round-trip rationale as ``_as1_llvm_ptr``: tracing only has ``!fly.ptr``,
    and emitting ``unrealized_conversion_cast`` at trace time deadlocks dialect
    conversion. FlyToIXDL folds the round-trip after converting ``fly.ptr``.
    """
    from ..._mlir import ir
    from ..primitive import ptrtoint

    addr = _arith.unwrap(ptrtoint(ptr))
    ty = "!llvm.ptr" if addrspace is None else f"!llvm.ptr<{int(addrspace)}>"
    return _llvm.inttoptr(ir.Type.parse(ty), addr)


def _as1_llvm_ptr(ptr):
    """Materialize ``!fly.ptr`` as ``!llvm.ptr<1>`` for ``stp.vs`` intrinsics."""
    return _llvm_ptr(ptr, addrspace=1)


def _cmpxchg_int_type(elem_ty):
    """Map an element IR type to the signless integer used by ``llvm.cmpxchg``.

    Returns ``(int_ty, needs_bitcast)``. Floats (and other same-width non-ints)
    are bitcast to ``iN`` for the intrinsic, then bitcast back on the result.
    """
    from ..._mlir.ir import FloatType, IntegerType

    if isinstance(elem_ty, IntegerType):
        return elem_ty, False
    if isinstance(elem_ty, FloatType):
        return IntegerType.get_signless(int(elem_ty.width)), True
    raise TypeError(
        f"atomic_cas unsupported element type {elem_ty}; expected integer or float "
        "(same-width bitcast to integer for llvm.cmpxchg)"
    )


def atomic_cas(
    ptr,
    expected,
    desired,
    *,
    success_ordering,
    failure_ordering,
    syncscope: str = "",
    alignment: int | None = None,
):
    """Emit ``llvm.cmpxchg``; return the value previously at ``ptr``.

    Element dtype is taken from ``expected`` / ``desired`` (must match).
    Comparison is **bit-pattern** CAS: non-integer values are bitcast to the
    same-width signless integer for ``llvm.cmpxchg``, then the old value is
    bitcast back. Call sites pick a width the target can lower (16/32-bit all
    gens; 64-bit on Blazer+/``ivcore40+``).

    ``ptr`` is a ``!fly.ptr`` to that element in global memory.
    ``success_ordering`` / ``failure_ordering`` are ``llvm.AtomicOrdering``
    values (e.g. ``acquire``, ``release``, ``acq_rel``). Default
    ``syncscope=""`` is LLVM system scope. Default ``alignment`` is the
    element size in bytes.

    Typical turnstile use::

        # wait until *lock == ticket
        old = ixdl.atomic_cas(lock, ticket, ticket, success_ordering=acquire, ...)
        # arrive: ticket -> ticket+1
        ixdl.atomic_cas(lock, ticket, ticket + 1, success_ordering=release, ...)
    """
    from ..._mlir.dialects import arith as _std_arith
    from ..numeric import Numeric

    exp = _arith.unwrap(expected)
    des = _arith.unwrap(desired)
    if exp.type != des.type:
        raise TypeError(f"atomic_cas expected/desired type mismatch: {exp.type} vs {des.type}")

    elem_ty = exp.type
    int_ty, needs_bitcast = _cmpxchg_int_type(elem_ty)
    width = int(int_ty.width)
    if alignment is None:
        alignment = max(width // 8, 1)

    if needs_bitcast:
        exp = _std_arith.bitcast(int_ty, exp)
        des = _std_arith.bitcast(int_ty, des)

    res = _llvm.cmpxchg(
        _llvm_ptr(ptr, addrspace=None),
        exp,
        des,
        success_ordering,
        failure_ordering,
        syncscope=syncscope,
        alignment=int(alignment),
    )
    previous = _llvm.extractvalue(int_ty, res, [0])
    if needs_bitcast:
        previous = _std_arith.bitcast(elem_ty, previous)
    return Numeric.from_ir_type(elem_ty)(previous)


def _stp_vs_call(bits: int, val, ptr, voffset, soffset, kop: int = 0, pred=None):
    """Emit ``llvm.bi.stp.vs[.pred].*`` for ``bits``-wide stores.

    Args are ``(val, base, voffset, soffset, kop[, pred])``:

    * ``ptr`` / ``base`` -- AS1 pointer to the tile origin (``!fly.ptr``,
      lowered via ptrtoint/inttoptr and folded in ``convert-fly-to-ixdl``)
    * ``voffset`` -- per-lane byte offset (bake once per lane)
    * ``soffset`` -- per-store secondary byte offset (often a constexpr tile offset)
    * ``kop`` -- store cache policy immediate (default 0)
    * ``pred`` -- optional i1; when set, emits the ``.pred`` intrinsic (pred last)

    Public helpers are named by store bit-width only (``b8`` / ``b16`` /
    ``b32`` / ``b64`` / ``b128``), not by element dtype.
    """
    args = [
        _arith.unwrap(val),
        _as1_llvm_ptr(ptr),
        _arith.unwrap(voffset),
        _arith.unwrap(soffset),
        _arith.unwrap(_arith.constant(int(kop), type=T.i32)),
    ]
    if pred is not None:
        args.append(_arith.unwrap(pred))
        return _llvm.call_intrinsic(None, _STP_VS_PRED[bits], args, [], [])
    return _llvm.call_intrinsic(None, _STP_VS[bits], args, [], [])


def stp_vs_b8(val, ptr, voffset, soffset, kop=0):
    return _stp_vs_call(8, val, ptr, voffset, soffset, kop)


def stp_vs_b16(val, ptr, voffset, soffset, kop=0):
    return _stp_vs_call(16, val, ptr, voffset, soffset, kop)


def stp_vs_b32(val, ptr, voffset, soffset, kop=0):
    return _stp_vs_call(32, val, ptr, voffset, soffset, kop)


def stp_vs_b64(val, ptr, voffset, soffset, kop=0):
    return _stp_vs_call(64, val, ptr, voffset, soffset, kop)


def stp_vs_b128(val, ptr, voffset, soffset, kop=0):
    return _stp_vs_call(128, val, ptr, voffset, soffset, kop)


def stp_vs_pred_b8(pred, val, ptr, voffset, soffset, kop=0):
    return _stp_vs_call(8, val, ptr, voffset, soffset, kop, pred=pred)


def stp_vs_pred_b16(pred, val, ptr, voffset, soffset, kop=0):
    return _stp_vs_call(16, val, ptr, voffset, soffset, kop, pred=pred)


def stp_vs_pred_b32(pred, val, ptr, voffset, soffset, kop=0):
    return _stp_vs_call(32, val, ptr, voffset, soffset, kop, pred=pred)


def stp_vs_pred_b64(pred, val, ptr, voffset, soffset, kop=0):
    return _stp_vs_call(64, val, ptr, voffset, soffset, kop, pred=pred)


def stp_vs_pred_b128(pred, val, ptr, voffset, soffset, kop=0):
    return _stp_vs_call(128, val, ptr, voffset, soffset, kop, pred=pred)
