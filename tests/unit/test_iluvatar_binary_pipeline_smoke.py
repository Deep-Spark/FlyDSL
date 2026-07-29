# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Opt-in Iluvatar compile-only binary pipeline smoke."""

import importlib
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.iluvatar_lower]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _REPO_ROOT / "python"
_COMPILER_DIR = _PYTHON_DIR / "flydsl" / "compiler"


def _required_path_from_env(name: str) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        pytest.skip(f"{name} is not set")

    path = Path(value)
    if not path.is_file():
        pytest.fail(f"{name} does not point to a file: {path}")
    return path


def _require_ixdl_attach_target(fly_opt: Path) -> None:
    result = subprocess.run([str(fly_opt), "--help"], check=True, text=True, capture_output=True)
    if "ixdl-attach-target" not in result.stdout + result.stderr:
        pytest.skip("fly-opt does not register ixdl-attach-target yet")


def _load_backends(monkeypatch):
    """Import flydsl.compiler.backends without importing JIT-only compiler exports."""
    monkeypatch.syspath_prepend(str(_PYTHON_DIR))
    for name in list(sys.modules):
        if name == "flydsl.compiler" or name.startswith("flydsl.compiler.backends"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    compiler_pkg = types.ModuleType("flydsl.compiler")
    compiler_pkg.__path__ = [str(_COMPILER_DIR)]
    monkeypatch.setitem(sys.modules, "flydsl.compiler", compiler_pkg)
    return importlib.import_module("flydsl.compiler.backends")


_MINIMAL_GPU_MODULE = "\n".join(
    [
        "module attributes {gpu.container_module} {",
        "  gpu.module @kernels {",
        "    gpu.func @k() kernel {",
        "      gpu.return",
        "    }",
        "  }",
        "}",
        "",
    ]
)


@pytest.mark.parametrize(
    "chip,target_needle,elf_triple_needle",
    [
        # IXDL printer always emits the resolved chip (no default elision).
        ("ivcore11", '#gpu.object<#ixdl.target<chip = "ivcore11">', "ivcore-1-1-0"),
        ("ivcore30", '#gpu.object<#ixdl.target<chip = "ivcore30">', "ivcore-3-0-0"),
    ],
)
def test_iluvatar_backend_pipeline_lowers_minimal_gpu_module_to_binary(
    monkeypatch, tmp_path, chip, target_needle, elf_triple_needle
):
    fly_opt = _required_path_from_env("FLYDSL_ILUVATAR_FLY_OPT")
    _require_ixdl_attach_target(fly_opt)
    backends = _load_backends(monkeypatch)
    backend = backends.get_backend("iluvatar", arch=chip)

    input_mlir = tmp_path / f"minimal_gpu_module_{chip}.mlir"
    input_mlir.write_text(_MINIMAL_GPU_MODULE)
    pipeline = f"builtin.module({','.join(backend.pipeline_fragments(compile_hints={}))})"

    result = subprocess.run(
        [str(fly_opt), str(input_mlir), f"--pass-pipeline={pipeline}"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "gpu.binary @kernels" in result.stdout
    assert target_needle in result.stdout
    assert elf_triple_needle in result.stdout
    assert "gpu.func @k" not in result.stdout
    if chip == "ivcore11":
        assert 'chip = "ivcore30"' not in result.stdout
        assert "ivcore-3-0-0" not in result.stdout
    else:
        assert 'chip = "ivcore11"' not in result.stdout
        assert "ivcore-1-1-0" not in result.stdout


@pytest.mark.parametrize(
    "chip,module_targets",
    [
        ("ivcore11", 'gpu.module @kernels [#ixdl.target<chip = "ivcore11">]'),
        ("ivcore30", 'gpu.module @kernels [#ixdl.target<chip = "ivcore30">]'),
    ],
)
def test_iluvatar_attach_target_emits_expected_ixdl_target(monkeypatch, tmp_path, chip, module_targets):
    """COMPILE_ONLY-style check: attach-target IR carries the ARCH chip."""
    fly_opt = _required_path_from_env("FLYDSL_ILUVATAR_FLY_OPT")
    _require_ixdl_attach_target(fly_opt)
    backends = _load_backends(monkeypatch)
    backend = backends.get_backend("iluvatar", arch=chip)

    input_mlir = tmp_path / f"attach_{chip}.mlir"
    input_mlir.write_text(_MINIMAL_GPU_MODULE)

    # Stop after attach-target so the gpu.module targets attribute is visible.
    fragments = []
    for frag in backend.pipeline_fragments(compile_hints={}):
        fragments.append(frag)
        if frag.startswith("ixdl-attach-target"):
            break
    pipeline = f"builtin.module({','.join(fragments)})"

    result = subprocess.run(
        [str(fly_opt), str(input_mlir), f"--pass-pipeline={pipeline}"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert module_targets in result.stdout
    if chip == "ivcore30":
        assert 'chip = "ivcore11"' not in result.stdout
    else:
        assert 'chip = "ivcore30"' not in result.stdout


def test_fly_to_ixdl_lowers_scalar_pointer_store(tmp_path):
    fly_opt = _required_path_from_env("FLYDSL_ILUVATAR_FLY_OPT")

    input_mlir = tmp_path / "scalar_store.mlir"
    input_mlir.write_text(
        "\n".join(
            [
                "module {",
                "  func.func @store_i32(%ptr: !fly.ptr<i32, global>) {",
                "    %c7 = arith.constant 7 : i32",
                "    %offset = fly.make_int_tuple() : () -> !fly.int_tuple<1>",
                "    %slot = fly.add_offset(%ptr, %offset)",
                "      : (!fly.ptr<i32, global>, !fly.int_tuple<1>) -> !fly.ptr<i32, global>",
                "    fly.ptr.store(%c7, %slot) : (i32, !fly.ptr<i32, global>) -> ()",
                "    return",
                "  }",
                "}",
                "",
            ]
        )
    )

    result = subprocess.run(
        [str(fly_opt), str(input_mlir), "--pass-pipeline=builtin.module(convert-fly-to-ixdl,canonicalize)"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "llvm.getelementptr" in result.stdout
    assert "llvm.store" in result.stdout
    assert "fly.add_offset" not in result.stdout
    assert "fly.ptr.store" not in result.stdout
