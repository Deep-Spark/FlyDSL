#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Verify a FlyDSL Iluvatar wheel was built with the correct toolchain.

Enforces two invariants that, together, prevent the "wheel imports OK on the
build host but SIGSEGVs on the deployment target" class of failure caused by
libstdc++ / glibc ABI mismatches at the nanobind <-> CPython boundary:

  1. Filename platform tag: the wheel filename MUST contain an explicit
     manylinux platform tag with a glibc level (e.g. `manylinux_2_31_x86_64`,
     `manylinux_2_34_x86_64`, `manylinux_2_38_x86_64`, `manylinux2014_x86_64`).
     A bare `linux_x86_64` (auditwheel never ran) or `manylinux_x86_64`
     (no glibc level; not a valid PEP 600 tag) is rejected.

  2. Compiler consistency: every bundled ELF `.so` under `flydsl/_mlir/_mlir_libs`
     MUST report the SAME GCC major version in its ELF `.comment` section, and
     that major version MUST match the platform tag's glibc level per the
     `_MANYLINUX_MIN_GCC_MAJOR` table below. This catches:
       - a wheel built on Ubuntu 24.04 (GCC 13) mislabelled as manylinux_2_31,
       - a wheel built with mixed toolchains (e.g. conda-forge gcc + system gcc),
       - a wheel where auditwheel `patchelf`-rewrote symlinks but not the
         underlying compile-time toolchain.

Usage:
  verify_wheel_toolchain.py <wheel> [<wheel> ...]

Exits non-zero on any violation. Prints one GitHub Actions `::error::` line
per violation so the failing job step lists all root causes at once.
"""

from __future__ import annotations

import argparse
import glob
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ELF_MAGIC = b"\x7fELF"

# Manylinux tag -> (min glibc, exact allowed GCC majors on this repo).
#
# Our Iluvatar A-track uses only these two production images:
#   - `10.150.9.98:80/infra/corex_base/ubuntu:20.04-latest` -> GCC 9
#   - `ghcr.io/deep-spark/flydsl-iluvatar-ci:stable`        -> GCC 13
#
# manylinux2014 / manylinux_2_28 (auditwheel repair on 20.04 with the CentOS 7
# devtoolset flow) are accepted for backward compatibility but not currently
# produced by our workflows.
_MANYLINUX_ALLOWED_GCC_MAJORS: dict[str, tuple[int, ...]] = {
    "manylinux2014_x86_64": (10, 11, 12, 13),
    "manylinux_2_17_x86_64": (10, 11, 12, 13),
    "manylinux_2_28_x86_64": (11, 12, 13),
    "manylinux_2_31_x86_64": (9,),
    "manylinux_2_34_x86_64": (11, 12, 13),
    "manylinux_2_35_x86_64": (11, 12, 13),
    "manylinux_2_38_x86_64": (12, 13),
    "manylinux_2_39_x86_64": (13,),
}

_PLAT_TAG_RE = re.compile(r"-(manylinux[0-9_]+_x86_64|linux_x86_64)\.whl$")
_GCC_COMMENT_RE = re.compile(r"GCC:\s*\(([^)]+)\)\s*(\d+)\.(\d+)")


def _wheel_platform_tag(wheel_name: str) -> str | None:
    m = _PLAT_TAG_RE.search(wheel_name)
    return m.group(1) if m else None


def _gcc_majors_in_comment(elf_path: Path) -> list[tuple[int, str]]:
    """Return every (major, full_version_string) tuple found in .comment.

    A single .so usually has one entry but LTO / conda-forge cross builds can
    stack multiple GCC toolchain tags. We check that ALL of them agree.
    """
    proc = subprocess.run(
        ["readelf", "-p", ".comment", str(elf_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    out = []
    for m in _GCC_COMMENT_RE.finditer(proc.stdout):
        major = int(m.group(2))
        minor = m.group(3)
        vendor = m.group(1).strip()
        out.append((major, f"GCC {major}.{minor} ({vendor})"))
    return out


def _collect_bundled_sos(wheel_path: Path, temp_dir: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    with zipfile.ZipFile(wheel_path) as zf:
        for member in zf.namelist():
            if "/_mlir_libs/" not in member:
                continue
            if ".so" not in member:
                continue
            target = temp_dir / member
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as dst:
                dst.write(src.read())
            try:
                if target.open("rb").read(4) == ELF_MAGIC:
                    out.append((member, target))
            except OSError:
                continue
    return out


def _verify_wheel(wheel_path: Path) -> list[str]:
    errors: list[str] = []

    plat = _wheel_platform_tag(wheel_path.name)
    if plat is None:
        errors.append(f"cannot parse platform tag from wheel filename: {wheel_path.name}")
        return errors
    if plat == "linux_x86_64":
        errors.append(
            "filename declares `linux_x86_64` (no manylinux tag). "
            "auditwheel repair did not run or failed silently. "
            "Ensure `patchelf` and `auditwheel` are installed in the build "
            "container so setup.py's `_auditwheel_repair_in_place` can produce "
            "a manylinux-tagged wheel."
        )
        return errors
    if plat == "manylinux_x86_64":
        errors.append(
            "filename declares bare `manylinux_x86_64` (no glibc level). "
            "This is not a valid PEP 600 tag. The wheel was likely renamed "
            "into a stable-alias name before verification; verify the "
            "original wheel from `dist/` instead."
        )
        return errors

    allowed_majors = _MANYLINUX_ALLOWED_GCC_MAJORS.get(plat)
    if allowed_majors is None:
        errors.append(
            f"unknown manylinux tag `{plat}`. If this is a new supported "
            "target, add it to `_MANYLINUX_ALLOWED_GCC_MAJORS` in "
            "verify_wheel_toolchain.py."
        )
        return errors

    with tempfile.TemporaryDirectory(prefix="verify-wheel-tc-") as td:
        temp_dir = Path(td)
        sos = _collect_bundled_sos(wheel_path, temp_dir)
        if not sos:
            errors.append("no ELF `.so` files found under `_mlir_libs/`; is this the " "right wheel?")
            return errors

        seen_majors: dict[int, list[str]] = {}
        for member, elf in sos:
            comments = _gcc_majors_in_comment(elf)
            if not comments:
                errors.append(f"{member}: no GCC tag found in .comment section (was it stripped?)")
                continue
            for major, label in comments:
                seen_majors.setdefault(major, []).append(f"{member}: {label}")

        if len(seen_majors) > 1:
            lines = []
            for major in sorted(seen_majors):
                lines.append(f"    GCC {major}:")
                for hit in seen_majors[major]:
                    lines.append(f"      - {hit}")
            errors.append(
                "mixed GCC major versions across bundled .so files. This "
                "produces libstdc++ RTTI inconsistencies at the nanobind / "
                "CPython C++ boundary and crashes at type registration.\n" + "\n".join(lines)
            )

        for major in seen_majors:
            if major not in allowed_majors:
                errors.append(
                    f"wheel platform tag `{plat}` allows GCC major(s) "
                    f"{list(allowed_majors)} but bundled .so uses GCC {major}. "
                    f"Rebuild inside the correct docker image "
                    f"(see .github/workflows/build-whl-iluvatar.yaml)."
                )
                break

    return errors


def _iter_wheels(patterns: list[str]) -> list[Path]:
    wheels: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            wheels.extend(Path(p) for p in matches if Path(p).is_file())
            continue
        p = Path(pattern)
        if p.is_file():
            wheels.append(p)
    return wheels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wheels", nargs="+", help="Wheel files or glob patterns (e.g. dist/*.whl)")
    args = parser.parse_args()

    if shutil.which("readelf") is None:
        print("ERROR: `readelf` is required but not found in PATH.", file=sys.stderr)
        return 2

    wheels = _iter_wheels(args.wheels)
    if not wheels:
        print(
            "ERROR: no wheel files matched: " + ", ".join(args.wheels),
            file=sys.stderr,
        )
        return 2

    any_bad = False
    for wheel in wheels:
        print(f"::group::Verifying {wheel.name}")
        errs = _verify_wheel(wheel)
        if errs:
            any_bad = True
            for e in errs:
                print(f"::error file={wheel}::{wheel.name}: {e}")
        else:
            plat = _wheel_platform_tag(wheel.name)
            print(f"OK: {wheel.name} (platform tag = {plat})")
        print("::endgroup::")

    return 1 if any_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
