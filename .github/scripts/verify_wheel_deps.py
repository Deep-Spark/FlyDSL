#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ELF_MAGIC = b"\x7fELF"
MLIR_LLVM_LIB_RE = re.compile(r"^lib(?:MLIR|LLVM).+\.so(?:\..*)?$")
NEEDED_RE = re.compile(r"Shared library:\s*\[(.+?)\]")


def _looks_like_elf(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(4) == ELF_MAGIC
    except OSError:
        return False


def _needed_libs(elf_path: Path) -> list[str]:
    proc = subprocess.run(
        ["readelf", "-d", str(elf_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"readelf failed for {elf_path}: {proc.stderr.strip()}")
    return NEEDED_RE.findall(proc.stdout)


def _collect_elf_entries(wheel_path: Path, temp_dir: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    with zipfile.ZipFile(wheel_path) as zf:
        for member in zf.namelist():
            if ".so" not in member:
                continue
            target = temp_dir / member
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as dst:
                dst.write(src.read())
            if _looks_like_elf(target):
                out.append((member, target))
    return out


def _verify_wheel(wheel_path: Path) -> list[str]:
    violations: list[str] = []
    with tempfile.TemporaryDirectory(prefix="verify-wheel-deps-") as tmp:
        temp_dir = Path(tmp)
        elf_entries = _collect_elf_entries(wheel_path, temp_dir)
        bundled_names = {Path(member).name for member, _ in elf_entries}

        for member, elf_path in elf_entries:
            needed = _needed_libs(elf_path)
            for lib in needed:
                if not MLIR_LLVM_LIB_RE.match(lib):
                    continue
                if lib in bundled_names:
                    continue
                violations.append(
                    f"{wheel_path.name}: {member} depends on external MLIR/LLVM lib `{lib}` (not bundled in wheel)"
                )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fail if wheel ELF files depend on MLIR/LLVM shared libraries that are not bundled inside the wheel."
        )
    )
    parser.add_argument("wheels", nargs="+", help="Wheel files or glob patterns (e.g. dist/*.whl)")
    args = parser.parse_args()

    if shutil.which("readelf") is None:
        print("ERROR: `readelf` is required but not found in PATH.", file=sys.stderr)
        return 2

    wheel_paths: list[Path] = []
    for pattern in args.wheels:
        matches = sorted(Path().glob(pattern))
        if matches:
            wheel_paths.extend(p for p in matches if p.is_file())
            continue
        p = Path(pattern)
        if p.is_file():
            wheel_paths.append(p)

    if not wheel_paths:
        print("ERROR: no wheel files found for inputs:", ", ".join(args.wheels), file=sys.stderr)
        return 2

    violations: list[str] = []
    for wheel in wheel_paths:
        violations.extend(_verify_wheel(wheel))

    if violations:
        print("Detected forbidden external MLIR/LLVM dynamic dependencies:")
        for item in violations:
            print(f"- {item}")
        return 1

    print(f"Dependency gate passed for {len(wheel_paths)} wheel(s): no external MLIR/LLVM dynamic deps found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
