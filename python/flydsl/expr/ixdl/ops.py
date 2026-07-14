# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Iluvatar ML / ALU intrinsics (non-sync).

Wrappers that emit Iluvatar LLVM intrinsics via ``llvm.call_intrinsic``.
Synchronization helpers live in :mod:`flydsl.expr.ixdl.sync`.
"""

from ..._mlir.dialects import llvm as _llvm
from .. import arith as _arith
from ..typing import T

# Descriptor-store LLVM names (Iluvatar ``__ivcorex_ml_mem_store{,_pred}_*``).
_STP_VS = {
    "i8": "llvm.bi.stp.vs.i8",
    "i16": "llvm.bi.stp.vs.i16",
    "i32": "llvm.bi.stp.vs.i32",
    "i64": "llvm.bi.stp.vs.i64",
    "i32x4": "llvm.bi.stp.vs.v4i32",
}
_STP_VS_PRED = {
    "i8": "llvm.bi.stp.vs.pred.i8",
    "i16": "llvm.bi.stp.vs.pred.i16",
    "i32": "llvm.bi.stp.vs.pred.i32",
    "i64": "llvm.bi.stp.vs.pred.i64",
    "i32x4": "llvm.bi.stp.vs.pred.v4i32",
}


def ml_byte_permute(a, b, sel: int):
    """Emit ``llvm.nvvm.prmt`` (Iluvatar ``ml_byte_permute_b32``).

    ``sel`` is a compile-time nibble selector: bytes 0-3 from ``a``, 4-7 from ``b``.
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


def _stp_vs_call(intrin: str, val, ptr, wco, wso, kop: int = 0, pred=None):
    """Emit ``llvm.bi.stp.vs[.pred].*``: ``(val, base, wco, wso, kop[, pred])``."""
    args = [
        _arith.unwrap(val),
        _as1_llvm_ptr(ptr),
        _arith.unwrap(wco),
        _arith.unwrap(wso),
        _arith.unwrap(_arith.constant(int(kop), type=T.i32)),
    ]
    if pred is not None:
        args.append(_arith.unwrap(pred))
    return _llvm.call_intrinsic(None, intrin, args, [], [])


def stp_vs_i8(val, ptr, wco, wso, kop=0):
    """Emit ``llvm.bi.stp.vs.i8`` (from ``__ivcorex_ml_mem_store_i8``)."""
    return _stp_vs_call(_STP_VS["i8"], val, ptr, wco, wso, kop)


def stp_vs_i16(val, ptr, wco, wso, kop=0):
    """Emit ``llvm.bi.stp.vs.i16`` (from ``__ivcorex_ml_mem_store_i16``)."""
    return _stp_vs_call(_STP_VS["i16"], val, ptr, wco, wso, kop)


def stp_vs_i32(val, ptr, wco, wso, kop=0):
    """Emit ``llvm.bi.stp.vs.i32`` (from ``__ivcorex_ml_mem_store_i32``).

    ``base`` is an AS1 pointer to the tile origin, ``wco`` is the per-lane byte
    offset (baked once), and ``wso`` is the per-store secondary byte offset.
    """
    return _stp_vs_call(_STP_VS["i32"], val, ptr, wco, wso, kop)


def stp_vs_i64(val, ptr, wco, wso, kop=0):
    """Emit ``llvm.bi.stp.vs.i64`` (from ``__ivcorex_ml_mem_store_i64``)."""
    return _stp_vs_call(_STP_VS["i64"], val, ptr, wco, wso, kop)


def stp_vs_i32x4(val, ptr, wco, wso, kop=0):
    """Emit ``llvm.bi.stp.vs.v4i32`` (from ``__ivcorex_ml_mem_store_i32x4``)."""
    return _stp_vs_call(_STP_VS["i32x4"], val, ptr, wco, wso, kop)


def stp_vs_pred_i8(pred, val, ptr, wco, wso, kop=0):
    """Emit ``llvm.bi.stp.vs.pred.i8`` (from ``__ivcorex_ml_mem_store_pred_i8``).

    Clang builtin argument order is ``(pred, val, desc, ...)``; the LLVM
    intrinsic puts ``pred`` last: ``(val, ptr, wco, wso, kop, pred)``.
    """
    return _stp_vs_call(_STP_VS_PRED["i8"], val, ptr, wco, wso, kop, pred=pred)


def stp_vs_pred_i16(pred, val, ptr, wco, wso, kop=0):
    """Emit ``llvm.bi.stp.vs.pred.i16`` (from ``__ivcorex_ml_mem_store_pred_i16``)."""
    return _stp_vs_call(_STP_VS_PRED["i16"], val, ptr, wco, wso, kop, pred=pred)


def stp_vs_pred_i32(pred, val, ptr, wco, wso, kop=0):
    """Emit ``llvm.bi.stp.vs.pred.i32`` (from ``__ivcorex_ml_mem_store_pred_i32``)."""
    return _stp_vs_call(_STP_VS_PRED["i32"], val, ptr, wco, wso, kop, pred=pred)


def stp_vs_pred_i64(pred, val, ptr, wco, wso, kop=0):
    """Emit ``llvm.bi.stp.vs.pred.i64`` (from ``__ivcorex_ml_mem_store_pred_i64``)."""
    return _stp_vs_call(_STP_VS_PRED["i64"], val, ptr, wco, wso, kop, pred=pred)


def stp_vs_pred_i32x4(pred, val, ptr, wco, wso, kop=0):
    """Emit ``llvm.bi.stp.vs.pred.v4i32`` (from ``__ivcorex_ml_mem_store_pred_i32x4``)."""
    return _stp_vs_call(_STP_VS_PRED["i32x4"], val, ptr, wco, wso, kop, pred=pred)
