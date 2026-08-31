#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Verify every native extension in a FlyDSL wheel actually imports.

The static checks (`verify_wheel_deps.py`, `verify_wheel_contents.py`,
`verify_wheel_toolchain.py`) all inspect the wheel without loading it, so a
wheel whose extension modules crash on import passes them cleanly. This script
closes that gap by importing each `_mlir_libs/*.so` module for real.

Each import runs in its own subprocess, because the failure mode being guarded
against is a SIGSEGV during module initialization, which would otherwise take
this script down with it.

Must run under an interpreter matching the wheel's ABI tag, which in practice
means inside the build container.

Usage:
  verify_wheel_import.py <wheel> [<wheel> ...]
  verify_wheel_import.py <wheel> --check-aslr-off

Exits non-zero if any extension module fails to import. `--check-aslr-off`
additionally reports (without failing) whether the modules survive with ASLR
disabled: the MLIR bindings currently carry a layout-sensitive fault, and that
probe is what distinguishes a genuinely sound build from a lucky one.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

EXT_PACKAGE = "flydsl._mlir._mlir_libs"


def _extension_modules(root: Path) -> list[str]:
    libs_dir = root / "flydsl" / "_mlir" / "_mlir_libs"
    if not libs_dir.is_dir():
        return []
    names = set()
    for so in libs_dir.glob("_*.so"):
        # _mlir.cpython-312-x86_64-linux-gnu.so -> _mlir
        stem = so.name.split(".")[0]
        names.add(f"{EXT_PACKAGE}.{stem}")
    return sorted(names)


def _import_ok(module: str, pythonpath: Path, *, no_aslr: bool) -> tuple[bool, str]:
    cmd = [sys.executable, "-c", f"import {module}"]
    if no_aslr:
        if shutil.which("setarch") is None:
            return True, "setarch unavailable, skipped"
        cmd = ["setarch", "-R", *cmd]
    # Inherit the environment: a conda interpreter needs HOME and its own vars
    # to start at all, and stripping them turns every import into a false
    # positive that looks exactly like the crash being guarded against.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(pythonpath)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode == 0:
        return True, ""
    if proc.returncode < 0:
        return False, f"killed by signal {-proc.returncode}"
    last = (proc.stderr or "").strip().splitlines()
    return False, last[-1] if last else f"exit {proc.returncode}"


def _check_wheel(wheel: Path, *, check_aslr_off: bool) -> int:
    print(f"== {wheel.name}")
    violations = 0
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(wheel) as zf:
            zf.extractall(root)
        # Extraction drops the executable bit; the loader does not need it, but
        # keep permissions sane for anything that shells out to the libraries.
        for so in root.rglob("*.so*"):
            so.chmod(0o755)

        modules = _extension_modules(root)
        if not modules:
            print(f"::error::{wheel.name}: no extension modules found under {EXT_PACKAGE}")
            return 1

        for module in modules:
            ok, detail = _import_ok(module, root, no_aslr=False)
            if ok:
                print(f"   {module:<50} OK")
            else:
                print(f"::error::{wheel.name}: cannot import {module} ({detail})")
                violations += 1

        if check_aslr_off:
            unlucky = []
            for module in modules:
                ok, _ = _import_ok(module, root, no_aslr=True)
                if not ok:
                    unlucky.append(module)
            if unlucky:
                print(
                    f"::warning::{wheel.name}: {len(unlucky)}/{len(modules)} extension modules "
                    "fail to import with ASLR disabled. The build works by virtue of its memory "
                    "layout rather than being sound; an unrelated code change can flip it."
                )

    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", nargs="+", help="wheel files (globs allowed)")
    parser.add_argument(
        "--check-aslr-off",
        action="store_true",
        help="also probe with ASLR disabled and warn if the build is only accidentally sound",
    )
    args = parser.parse_args()

    paths: list[Path] = []
    for pattern in args.wheels:
        matched = [Path(p) for p in glob.glob(pattern)]
        if not matched:
            print(f"::error::no wheel matched {pattern}")
            return 1
        paths.extend(matched)

    violations = 0
    for wheel in paths:
        violations += _check_wheel(wheel, check_aslr_off=args.check_aslr_off)

    if violations:
        print(f"::error::{violations} extension module(s) failed to import")
        return 1
    print("all extension modules import cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
