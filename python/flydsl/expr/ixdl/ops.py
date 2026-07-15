# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Iluvatar ALU / descriptor-store intrinsics (non-sync).

Wrappers that emit Iluvatar LLVM intrinsics via ``llvm.call_intrinsic``.
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


def _as1_llvm_ptr(ptr):
    """Materialize ``!fly.ptr`` as ``!llvm.ptr<1>`` for ``stp.vs`` intrinsics.

    Why the ``fly.ptrtoint`` / ``llvm.inttoptr`` round-trip:
    Tracing only has ``!fly.ptr``, but ``llvm.call_intrinsic`` for ``stp.vs``
    requires a ``!llvm.ptr`` operand. There is no first-class FlyIXDL op for
    these intrinsics, and emitting ``unrealized_conversion_cast`` at trace time
    deadlocks dialect conversion. Crossing via i64 keeps types legal until
    FlyToIXDL converts ``fly.ptr`` to ``llvm.ptr`` and folds the round-trip
    (see ``FoldStpVsPtrRoundtrip``).
    """
    from ..._mlir import ir
    from ..primitive import ptrtoint

    addr = _arith.unwrap(ptrtoint(ptr))
    return _llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), addr)


def _stp_vs_call(bits: int, val, ptr, voffset, soffset, kop: int = 0, pred=None):
    """Emit ``llvm.bi.stp.vs[.pred].*`` for ``bits``-wide stores.

    Args are ``(val, base, voffset, soffset, kop[, pred])``:

    * ``ptr`` / ``base`` — AS1 pointer to the tile origin (``!fly.ptr``,
      lowered via ptrtoint/inttoptr and folded in ``convert-fly-to-ixdl``)
    * ``voffset`` — per-lane byte offset (bake once per lane)
    * ``soffset`` — per-store secondary byte offset (often a constexpr tile offset)
    * ``kop`` — store cache policy immediate (default 0)
    * ``pred`` — optional i1; when set, emits the ``.pred`` intrinsic (pred last)

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
