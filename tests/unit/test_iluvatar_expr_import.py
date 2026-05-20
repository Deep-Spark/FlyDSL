# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Import-time checks for ``flydsl.expr.iluvatar`` helpers."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l0_backend_agnostic]

_REPO_ROOT = Path(__file__).resolve().parents[2]

_BOOTSTRAP_SOURCE_WITH_BUILD_MLIR = """
import os
import flydsl

build_pkg = os.environ["FLYDSL_TEST_BUILD_FLYDSL_PKG"]
if build_pkg not in flydsl.__path__:
    flydsl.__path__.append(build_pkg)
"""

_ILUVATAR_IMPORT_CHECK = """
import importlib
from flydsl._mlir import ir

expr = importlib.import_module("flydsl.expr")
assert "iluvatar" not in expr.__dict__
iluvatar = expr.iluvatar
assert iluvatar.__name__ == "flydsl.expr.iluvatar"
assert expr.iluvatar is iluvatar

expected = {
    "AsyncCopy4x64B8Row": "!fly_ixdl.mr.async_copy.4x64.b8.row",
    "AsyncCopy4x64B8Col": "!fly_ixdl.mr.async_copy.4x64.b8.col",
    "AsyncCopy16x64B8Row": "!fly_ixdl.mr.async_copy.16x64.b8.row",
    "AsyncCopy16x64B8Col": "!fly_ixdl.mr.async_copy.16x64.b8.col",
    "AsyncCopy4x32B16Row": "!fly_ixdl.mr.async_copy.4x32.b16.row",
    "AsyncCopy4x32B16Col": "!fly_ixdl.mr.async_copy.4x32.b16.col",
    "AsyncCopy16x32B16Row": "!fly_ixdl.mr.async_copy.16x32.b16.row",
    "AsyncCopy16x32B16Col": "!fly_ixdl.mr.async_copy.16x32.b16.col",
    "AsyncCopy1x1B64": "!fly_ixdl.mr.async_copy.1x1b64",
    "AsyncCopy1x4B64": "!fly_ixdl.mr.async_copy.1x4b64",
    "AsyncCopy1x8B64": "!fly_ixdl.mr.async_copy.1x8b64",
    "AsyncCopy1x16B64": "!fly_ixdl.mr.async_copy.1x16b64",
    "AsyncCopy4x16B32Row": "!fly_ixdl.mr.async_copy.4x16.b32.row",
    "AsyncCopy8x16B32Row": "!fly_ixdl.mr.async_copy.8x16.b32.row",
    "AsyncCopy16x16B32Row": "!fly_ixdl.mr.async_copy.16x16.b32.row",
    "AsyncCopy16x16B32Col": "!fly_ixdl.mr.async_copy.16x16.b32.col",
}

with ir.Context() as ctx:
    ctx.load_all_available_dialects()
    with ir.Location.unknown(ctx):
        for ctor_name, type_text in expected.items():
            ty = getattr(iluvatar, ctor_name)()
            assert str(ty) == type_text, (ctor_name, str(ty), type_text)

assert callable(iluvatar.cp_async_commit_group)
assert callable(iluvatar.cp_async_wait_group)
"""


def _build_env():
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("torch is required for flydsl.expr import path in this environment")

    pkg = _REPO_ROOT / "build-fly" / "python_packages" / "flydsl"
    if not pkg.is_dir():
        pytest.skip("build-fly python_packages not found (run scripts/build.sh)")

    env = os.environ.copy()
    bpkg = str(_REPO_ROOT / "build-fly" / "python_packages")
    spkg = str(_REPO_ROOT / "python")
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([spkg, bpkg] + ([prev] if prev else []))
    env["FLYDSL_TEST_BUILD_FLYDSL_PKG"] = str(pkg)
    return env


def _run_subprocess(code: str):
    proc = subprocess.run(
        [sys.executable, "-c", _BOOTSTRAP_SOURCE_WITH_BUILD_MLIR + code],
        cwd=str(_REPO_ROOT),
        env=_build_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(
            "subprocess iluvatar import check failed\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        ) from None


def test_expr_iluvatar_alias_and_async_copy_ctors():
    _run_subprocess(_ILUVATAR_IMPORT_CHECK)
