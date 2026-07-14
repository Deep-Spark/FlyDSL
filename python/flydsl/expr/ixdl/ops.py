# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Iluvatar ML / ALU intrinsics (non-sync).

Wrappers that emit Iluvatar LLVM intrinsics via ``llvm.call_intrinsic``.
Synchronization helpers live in :mod:`flydsl.expr.ixdl.sync`.
"""

from ..._mlir.dialects import llvm as _llvm
from .. import arith as _arith
from ..typing import T


def ml_byte_permute(a, b, sel: int):
    """Iluvatar ``ml_byte_permute_b32``.

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


def stp_vs_i32(val, ptr, wco, wso, kop=0):
    """Iluvatar descriptor store: ``llvm.bi.stp.vs.i32`` → ``ml_lsa_store_dword``.

    Matches Iluvatar ``__ivcorex_ml_mem_store_i32``: ``base`` is an AS1 pointer to
    the tile origin, ``wco`` is the per-lane byte offset (baked once), and
    ``wso`` is the per-store secondary byte offset. Prefer this over plain
    ``ptr_store`` + combined GEP for ivcore11 i32 epilogues (lower Memory
    Throttle / higher GMEM store throughput).

    Why the ``fly.ptrtoint`` → ``llvm.inttoptr`` round-trip:
    Tracing only has ``!fly.ptr``, but ``llvm.call_intrinsic`` for ``stp.vs``
    requires a ``!llvm.ptr`` operand. There is no first-class FlyIXDL op for
    this intrinsic, and emitting ``unrealized_conversion_cast`` at trace time
    deadlocks dialect conversion. Crossing via i64 keeps types legal until
    FlyToIXDL converts ``fly.ptr`` → ``llvm.ptr`` and folds the round-trip
    (see ``FoldStpVsPtrRoundtrip``).
    """
    from ..._mlir import ir
    from ..primitive import ptrtoint

    # !fly.ptr → i64 → !llvm.ptr<1>; folded away in convert-fly-to-ixdl.
    addr = _arith.unwrap(ptrtoint(ptr))
    llvm_ptr = _llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), addr)
    return _llvm.call_intrinsic(
        None,
        "llvm.bi.stp.vs.i32",
        [
            _arith.unwrap(val),
            llvm_ptr,
            _arith.unwrap(wco),
            _arith.unwrap(wso),
            _arith.unwrap(_arith.constant(int(kop), type=T.i32)),
        ],
        [],
        [],
    )
