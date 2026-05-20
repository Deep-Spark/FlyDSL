# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Iluvatar MR async-copy atom constructors.

MR async-copy atom state:

- ``soffset`` (`i32`), default zero — per-call source offset in elements
- ``imm_offset`` (`i32`), default zero — shared-memory byte offset
- ``stride_byte`` (`i32`), default zero — optional runtime global pitch in bytes

When ``stride_byte == 0``, lowering falls back to static layout-derived pitch.
When non-zero, ``stride_byte`` overrides layout pitch for dynamic-stride flows.
"""

import importlib

from ..._mlir import ir  # type: ignore[reportMissingImports]

# FlyIXDL Python bindings are opt-in: they are built only when Iluvatar is
# enabled (FLYDSL_BACKENDS includes iluvatar). Default ROCm builds do not ship
# _mlirDialectsFlyIXDL, so a hard import would break expr.iluvatar on import.
# Fall back to ir.Type.parse() below when the extension is absent.
_EXT_MODULE = "flydsl._mlir._mlir_libs._mlirDialectsFlyIXDL"

try:
    _ixdl_ext = importlib.import_module(_EXT_MODULE)
except ImportError:
    _ixdl_ext = None


def _get_copy_op_type(type_class_name: str, asm_type: str):
    if _ixdl_ext is not None:
        type_cls = getattr(_ixdl_ext, type_class_name, None)
        if type_cls is not None:
            return type_cls.get()
    return ir.Type.parse(asm_type)


def AsyncCopy4x64B8Row():
    return _get_copy_op_type("CopyOpMRAsyncCopy4x64B8RowType", "!fly_ixdl.mr.async_copy.4x64.b8.row")


def AsyncCopy4x64B8Col():
    return _get_copy_op_type("CopyOpMRAsyncCopy4x64B8ColType", "!fly_ixdl.mr.async_copy.4x64.b8.col")


def AsyncCopy16x64B8Row():
    return _get_copy_op_type("CopyOpMRAsyncCopy16x64B8RowType", "!fly_ixdl.mr.async_copy.16x64.b8.row")


def AsyncCopy16x64B8Col():
    return _get_copy_op_type("CopyOpMRAsyncCopy16x64B8ColType", "!fly_ixdl.mr.async_copy.16x64.b8.col")


def AsyncCopy4x32B16Row():
    return _get_copy_op_type("CopyOpMRAsyncCopy4x32B16RowType", "!fly_ixdl.mr.async_copy.4x32.b16.row")


def AsyncCopy4x32B16Col():
    return _get_copy_op_type("CopyOpMRAsyncCopy4x32B16ColType", "!fly_ixdl.mr.async_copy.4x32.b16.col")


def AsyncCopy16x32B16Row():
    return _get_copy_op_type("CopyOpMRAsyncCopy16x32B16RowType", "!fly_ixdl.mr.async_copy.16x32.b16.row")


def AsyncCopy16x32B16Col():
    return _get_copy_op_type("CopyOpMRAsyncCopy16x32B16ColType", "!fly_ixdl.mr.async_copy.16x32.b16.col")


def AsyncCopy1x1B64():
    return _get_copy_op_type("CopyOpMRAsyncCopy1x1B64Type", "!fly_ixdl.mr.async_copy.1x1b64")


def AsyncCopy1x4B64():
    return _get_copy_op_type("CopyOpMRAsyncCopy1x4B64Type", "!fly_ixdl.mr.async_copy.1x4b64")


def AsyncCopy1x8B64():
    return _get_copy_op_type("CopyOpMRAsyncCopy1x8B64Type", "!fly_ixdl.mr.async_copy.1x8b64")


def AsyncCopy1x16B64():
    return _get_copy_op_type("CopyOpMRAsyncCopy1x16B64Type", "!fly_ixdl.mr.async_copy.1x16b64")


def AsyncCopy4x16B32Row():
    return _get_copy_op_type("CopyOpMRAsyncCopy4x16B32RowType", "!fly_ixdl.mr.async_copy.4x16.b32.row")


def AsyncCopy8x16B32Row():
    return _get_copy_op_type("CopyOpMRAsyncCopy8x16B32RowType", "!fly_ixdl.mr.async_copy.8x16.b32.row")


def AsyncCopy16x16B32Row():
    return _get_copy_op_type("CopyOpMRAsyncCopy16x16B32RowType", "!fly_ixdl.mr.async_copy.16x16.b32.row")


def AsyncCopy16x16B32Col():
    return _get_copy_op_type("CopyOpMRAsyncCopy16x16B32ColType", "!fly_ixdl.mr.async_copy.16x16.b32.col")


__all__ = [
    "AsyncCopy4x64B8Row",
    "AsyncCopy4x64B8Col",
    "AsyncCopy16x64B8Row",
    "AsyncCopy16x64B8Col",
    "AsyncCopy4x32B16Row",
    "AsyncCopy4x32B16Col",
    "AsyncCopy16x32B16Row",
    "AsyncCopy16x32B16Col",
    "AsyncCopy1x1B64",
    "AsyncCopy1x4B64",
    "AsyncCopy1x8B64",
    "AsyncCopy1x16B64",
    "AsyncCopy4x16B32Row",
    "AsyncCopy8x16B32Row",
    "AsyncCopy16x16B32Row",
    "AsyncCopy16x16B32Col",
]
