# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import ctypes
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import List

from .._mlir import ir
from .._mlir.execution_engine import ExecutionEngine
from .protocol import fly_pointers


def _resolve_shared_lib(basename: str, *, extra_dirs: List[Path] | None = None) -> str:
    mlir_libs_dir = Path(__file__).resolve().parent.parent / "_mlir" / "_mlir_libs"
    search_dirs = [mlir_libs_dir]
    search_dirs.extend(extra_dirs or [])

    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    for raw_dir in ld_library_path.split(":"):
        if raw_dir:
            search_dirs.append(Path(raw_dir))

    # Common default install location for ixcc/corex.
    search_dirs.append(Path.home() / "new/sw_home/local/corex/lib64")

    for directory in search_dirs:
        if not directory.exists():
            continue
        exact = directory / basename
        if exact.exists():
            return str(exact)
        matches = sorted(directory.glob(f"{basename}*"))
        if matches:
            return str(matches[0])

    raise FileNotFoundError(
        f"Required shared library not found: {basename}\n"
        f"Looked in: {[str(d) for d in search_dirs]}"
    )


@lru_cache(maxsize=1)
def _resolve_runtime_libs() -> List[str]:
    from .backends import get_backend

    backend = get_backend()
    return [_resolve_shared_lib(name) for name in backend.jit_runtime_lib_basenames()]


class _ArgPacker:
    """Thread-local buffer for packing C pointer arguments."""

    def __init__(self):
        self._tls = threading.local()

    def pack(self, ptrs: List[ctypes.c_void_p]):
        size = len(ptrs)
        buf = getattr(self._tls, "packed_args", None)
        capacity = getattr(self._tls, "capacity", 0)
        if buf is None or capacity < size:
            buf = (ctypes.c_void_p * size)()
            self._tls.packed_args = buf
            self._tls.capacity = size
        for i, ptr in enumerate(ptrs):
            buf[i] = ptr
        return buf


class CompiledArtifact:
    def __init__(
        self,
        compiled_module: ir.Module,
        func_name: str,
        source_ir: str = None,
    ):
        self._ir_text = str(compiled_module)
        self._entry = func_name
        self._source_ir = source_ir
        self._module = None
        self._engine = None
        self._func_exe = None
        self._lock = threading.Lock()
        self._packer = _ArgPacker()

    def __getstate__(self):
        return {
            "ir_text": self._ir_text,
            "entry": self._entry,
            "source_ir": self._source_ir,
        }

    def __setstate__(self, state):
        self._ir_text = state["ir_text"]
        self._entry = state["entry"]
        self._source_ir = state["source_ir"]
        self._module = None
        self._engine = None
        self._func_exe = None
        self._lock = threading.Lock()
        self._packer = _ArgPacker()

    def _ensure_engine(self):
        with self._lock:
            if self._engine is not None:
                return

            with ir.Context() as ctx:
                ctx.load_all_available_dialects()
                self._module = ir.Module.parse(self._ir_text)
                self._engine = ExecutionEngine(
                    self._module,
                    opt_level=3,
                    shared_libs=_resolve_runtime_libs(),
                )
                self._engine.initialize()

    def _get_func_exe(self):
        if self._func_exe is None:
            if self._engine is None:
                self._ensure_engine()
            func_ptr = self._engine.raw_lookup(self._entry)
            self._func_exe = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(func_ptr)
        return self._func_exe

    def __call__(self, *args, **kwargs):
        func_exe = self._get_func_exe()

        all_c_ptrs: List[ctypes.c_void_p] = []
        for arg in args:
            all_c_ptrs.extend(fly_pointers(arg))

        packed_args = self._packer.pack(all_c_ptrs)
        return func_exe(packed_args)

    def dump(self, compiled: bool = True):
        if compiled:
            print("=" * 60)
            print("Compiled MLIR IR:")
            print("=" * 60)
            print(self._ir_text)
        else:
            if self._source_ir is None:
                print("Original IR not available")
            else:
                print("=" * 60)
                print("Original MLIR IR:")
                print("=" * 60)
                print(self._source_ir)

    @property
    def ir(self) -> str:
        return self._ir_text

    @property
    def source_ir(self) -> str:
        return self._source_ir
