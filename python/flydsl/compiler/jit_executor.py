# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import ctypes
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from .._mlir import ir
from .._mlir.execution_engine import ExecutionEngine
from .protocol import fly_pointers


def _iter_fallback_dirs() -> List[Path]:
    """Ordered extra search dirs for JIT runtime shared libraries.

    ROCm installs everything into ``_mlir/_mlir_libs`` at build time, so the
    fallback list is never consulted there. The IXDL backend on the other
    hand depends on ``libmlir_cuda_runtime.so``, which ships with ixcc and
    lives outside FlyDSL's own build tree.
    """
    dirs: List[Path] = []

    def _push(p: Optional[str]) -> None:
        if not p:
            return
        pp = Path(p).expanduser()
        if pp.is_dir() and pp not in dirs:
            dirs.append(pp)

    _push(os.environ.get("FLYDSL_RUNTIME_LIB_DIR"))
    for entry in (os.environ.get("FLYDSL_RUNTIME_LIB_DIRS") or "").split(os.pathsep):
        _push(entry)
    for entry in (os.environ.get("LD_LIBRARY_PATH") or "").split(os.pathsep):
        _push(entry)

    # Known install roots for the Iluvatar (ixcc) stack. Harmless on other
    # hosts: the paths simply won't exist and are skipped by _push().
    _push("/home/caokefan/sw_home/sdk/ixcc/build/lib")
    sw_home = os.environ.get("SW_HOME")
    if sw_home:
        _push(os.path.join(sw_home, "local", "corex", "lib64"))
        _push(os.path.join(sw_home, "sdk", "ixcc", "build", "lib"))

    return dirs


def _find_runtime_lib(name: str, primary: Path) -> Path:
    """Return an existing path for *name*, searching primary first."""
    candidate = primary / name
    if candidate.exists():
        return candidate
    for d in _iter_fallback_dirs():
        p = d / name
        if p.exists():
            return p
    searched = [str(primary)] + [str(d) for d in _iter_fallback_dirs()]
    raise FileNotFoundError(
        f"Required JIT runtime library not found: {name}\n"
        f"Searched (in order): {searched}\n"
        f"Set FLYDSL_RUNTIME_LIB_DIR (or FLYDSL_RUNTIME_LIB_DIRS) to the "
        f"directory containing it, or rebuild FlyDSL so it is installed "
        f"under _mlir/_mlir_libs/."
    )


@lru_cache(maxsize=1)
def _resolve_runtime_libs() -> List[str]:
    from .backends import get_backend

    backend = get_backend()
    mlir_libs_dir = Path(__file__).resolve().parent.parent / "_mlir" / "_mlir_libs"
    libs = [_find_runtime_lib(name, mlir_libs_dir) for name in backend.jit_runtime_lib_basenames()]
    return [str(p) for p in libs]


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

            # Create context and immediately exit the with-block, but
            # keep a reference so the context is not garbage-collected.
            # Destroying the context while ExecutionEngine still holds
            # HSA code objects causes GPU memory access faults.
            ctx = ir.Context()
            with ctx:
                ctx.load_all_available_dialects()
                self._module = ir.Module.parse(self._ir_text)
                self._engine = ExecutionEngine(
                    self._module,
                    opt_level=3,
                    shared_libs=_resolve_runtime_libs(),
                )
                self._engine.initialize()
            # Store ctx to prevent GC (but no longer the active context)
            self._ctx = ctx

    def _get_func_exe(self):
        if self._func_exe is None:
            if self._engine is None:
                self._ensure_engine()
            func_ptr = self._engine.raw_lookup(self._entry)
            self._func_exe = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(func_ptr)
        return self._func_exe

    def __call__(self, *args, **kwargs):
        func_exe = self._get_func_exe()

        owned: list = []
        all_c_ptrs: List[ctypes.c_void_p] = []
        for arg in args:
            ptrs = fly_pointers(arg)
            owned.append(ptrs)
            owned.append(arg)
            all_c_ptrs.extend(ptrs)

        packed_args = self._packer.pack(all_c_ptrs)

        result = func_exe(packed_args)
        del owned
        return result

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
