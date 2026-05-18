# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Unit-test path bootstrap when pytest is run with --confcutdir=tests/unit.

Parent tests/conftest.py is not loaded in that mode, so mirror its FlyDSL
python_packages discovery here.
"""

import os
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]


def _prepend_corex_ld_library_path() -> None:
    corex = os.environ.get("COREX_ROOT", "").strip()
    if not corex:
        for candidate in (
            Path("/home/wcyx/sw_home/local/corex"),
            Path.home() / "sw_home/local/corex",
        ):
            if (candidate / "lib64").is_dir():
                corex = str(candidate)
                break
    if not corex:
        return
    lib64 = str(Path(corex) / "lib64")
    prev = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in prev.split(":") if p]
    if lib64 not in parts:
        os.environ["LD_LIBRARY_PATH"] = f"{lib64}:{prev}" if prev else lib64


_prepend_corex_ld_library_path()

_fly_pkg_dir = _repo_root / "build-fly" / "python_packages"
if _fly_pkg_dir.exists():
    _p = str(_fly_pkg_dir)
    _already = _p in sys.path or any(
        os.path.isdir(ep) and os.path.samefile(ep, _p)
        for ep in sys.path
        if ep
    )
    if not _already:
        sys.path.insert(0, _p)
