#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

MLIR_LLVM_LIB_RE = re.compile(r"^\s*(lib(?:MLIR|LLVM)[^\s]*)\s*=>\s*(.+?)\s+\(")
NOT_FOUND_RE = re.compile(r"^\s*(lib(?:MLIR|LLVM)[^\s]*)\s*=>\s*not found$")


def _find_mlir_libs_dir() -> Path:
    import flydsl  # noqa: PLC0415

    pkg_dir = Path(flydsl.__file__).resolve().parent
    libs_dir = pkg_dir / "_mlir" / "_mlir_libs"
    if not libs_dir.is_dir():
        raise RuntimeError(f"flydsl mlir libs directory not found: {libs_dir}")
    return libs_dir


def _iter_elfs(libs_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in libs_dir.glob("*.so*")
        if p.is_file() and not p.is_symlink()
    )


def _check_ldd(elf_path: Path, libs_dir: Path) -> list[str]:
    proc = subprocess.run(
        ["ldd", str(elf_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return [f"{elf_path.name}: ldd failed ({proc.stderr.strip() or 'unknown error'})"]

    violations: list[str] = []
    for raw in proc.stdout.splitlines():
        line = raw.strip()
        if not line:
            continue

        m_nf = NOT_FOUND_RE.match(line)
        if m_nf:
            violations.append(f"{elf_path.name}: `{m_nf.group(1)}` not found")
            continue

        m = MLIR_LLVM_LIB_RE.match(line)
        if not m:
            continue

        lib_name, resolved = m.group(1), m.group(2)
        resolved_path = Path(resolved).resolve()
        try:
            resolved_path.relative_to(libs_dir)
        except ValueError:
            violations.append(
                f"{elf_path.name}: `{lib_name}` resolves outside wheel payload -> {resolved_path}"
            )
    return violations


def main() -> int:
    try:
        libs_dir = _find_mlir_libs_dir()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    elf_paths = _iter_elfs(libs_dir)
    if not elf_paths:
        print(f"ERROR: no .so files found under {libs_dir}", file=sys.stderr)
        return 2

    violations: list[str] = []
    for elf in elf_paths:
        violations.extend(_check_ldd(elf, libs_dir))

    if violations:
        print("Installed flydsl linkage gate failed:")
        for item in violations:
            print(f"- {item}")
        return 1

    print(
        f"Installed flydsl linkage gate passed: {len(elf_paths)} ELF files checked under {libs_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
