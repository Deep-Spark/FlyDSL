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
