# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Layered IR-level tests for the ivcore11 (FlyIXDL) SME GEMM stack.

These tests do NOT require a GPU. They cover:

- L0  : ``swizzle_mod`` numeric folding through ``fly-opt`` (cute::Swizzle_Mod).
- L1a : ``make_smem_tile`` produces the playbook ComposedLayout structure.
- L1b : ``convert-fly-to-ixdl`` emits the expected ``ixdl.*`` / ``llvm`` ops.

All tests skip cleanly when the Iluvatar build artifacts (fly-opt with FlyIXDL,
or the ``fly_ixdl`` python dialect) are unavailable, so they are safe in
ROCm-only environments.
"""

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l1a_compile_no_target_dialect]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MLIR_DIR = _REPO_ROOT / "tests" / "mlir"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fly_opt_path() -> Path:
    configured = os.environ.get("FLYDSL_FLY_OPT") or os.environ.get("FLY_OPT")
    candidates = [
        Path(configured) if configured else None,
        _REPO_ROOT / "build-fly" / "bin" / "fly-opt",
        _REPO_ROOT / "build-ixcc" / "bin" / "fly-opt",
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate
    pytest.skip("fly-opt binary not available; set FLYDSL_FLY_OPT to run this test")


def _run_fly_opt(mlir_file: Path, *args: str):
    return subprocess.run(
        [str(_fly_opt_path()), str(mlir_file), *args],
        check=False,
        text=True,
        capture_output=True,
    )


def _require_flyixdl_fly_opt():
    """Skip unless fly-opt understands FlyIXDL types (Iluvatar build)."""
    probe = _run_fly_opt(
        _MLIR_DIR / "Conversion" / "FlyToIXDL" / "mr_slb_load.mlir",
        "--fly-rewrite-func-signature",
        "--fly-canonicalize",
        "--fly-layout-lowering",
        "--convert-fly-to-ixdl",
    )
    if probe.returncode != 0 and (
        "unregistered dialect" in probe.stderr
        or "Unknown command line argument" in probe.stderr
        or "fly_ixdl" in probe.stderr
        and "expected" in probe.stderr
    ):
        pytest.skip(f"fly-opt lacks FlyIXDL support: {probe.stderr.splitlines()[:1]}")
    return probe


_IXDL_PIPELINE = (
    "--fly-rewrite-func-signature",
    "--fly-canonicalize",
    "--fly-layout-lowering",
    "--convert-fly-to-ixdl",
)


# ---------------------------------------------------------------------------
# L0: swizzle_mod numeric folding (cute::Swizzle_Mod<2,6,2>)
# ---------------------------------------------------------------------------


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize(
    "func_name,expected_idx",
    [
        ("sm_crd_lt256", 192),  # off=192 < 256 -> identity
        ("sm_crd_256", 320),  # off=256 -> 320
        ("sm_crd_448", 256),  # off=448 -> 256
    ],
)
def test_swizzle_mod_crd2idx_numeric(func_name, expected_idx):
    mlir = _MLIR_DIR / "LayoutAlgebra" / "swizzle_mod_crd2idx.mlir"
    if not mlir.is_file():
        pytest.skip("swizzle_mod_crd2idx.mlir not present")
    result = _run_fly_opt(mlir, "--split-input-file")
    assert result.returncode == 0, result.stderr
    # A successful round-trip pins the folded result type (verifier rejects wrong type).
    assert f"@{func_name}" in result.stdout
    assert f"!fly.int_tuple<{expected_idx}>" in result.stdout


@pytest.mark.l0_backend_agnostic
def test_swizzle_mod_does_not_disturb_xor_swizzle():
    """SwizzleMod must round-trip independently of the existing XOR Swizzle."""
    mlir = _MLIR_DIR / "LayoutAlgebra" / "swizzle_mod.mlir"
    if not mlir.is_file():
        pytest.skip("swizzle_mod.mlir not present")
    result = _run_fly_opt(mlir, "--split-input-file")
    assert result.returncode == 0, result.stderr
    assert "SM<" in result.stdout  # SwizzleMod printed form


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize(
    "func_name,expected_swizzle",
    [
        # Bit-level canonical rowxfb8 swizzle SM<2,9,2> recast 1-bit -> 8-bit (int8)
        # -> element-space SM<2,6,2> (base 9 - log2(8) = 6).
        ("rowxfb8_bit_to_int8_flat", "SM<2,6,2> o 0 o 1024:1"),
        ("rowxfb8_bit_to_int8_atom", "SM<2,6,2>"),
        # Reverse: int8 element SM<2,6,2> downcast 8-bit -> 1-bit -> SM<2,9,2>.
        ("rowxfb8_int8_to_bit_flat", "SM<2,9,2> o 0 o 8192:1"),
    ],
)
def test_swizzle_mod_rowxfb8_recast_bit_element(func_name, expected_swizzle):
    """No smem_ptr_flag specialization: the bit-level rowxfb8 SwizzleMod is
    SM<2,9,2>, which recast_layout (generic upcast) maps to the element-space
    SM<2,6,2> for int8 (and back). A successful round-trip pins the inferred
    result type (the verifier rejects a wrong declared type)."""
    mlir = _MLIR_DIR / "LayoutAlgebra" / "swizzle_mod_recast.mlir"
    if not mlir.is_file():
        pytest.skip("swizzle_mod_recast.mlir not present")
    result = _run_fly_opt(mlir, "--split-input-file")
    assert result.returncode == 0, result.stderr
    assert f"@{func_name}" in result.stdout
    assert expected_swizzle in result.stdout


# ---------------------------------------------------------------------------
# L1a: make_smem_tile layout structure (in-process trace, no device)
# ---------------------------------------------------------------------------


def _trace_make_smem_tile(M, K, dtype=None, **kwargs):
    ir = pytest.importorskip("flydsl._mlir.ir", reason="flydsl python package not built")
    from flydsl._mlir.ir import Context, InsertionPoint, Location, Module  # noqa: F401

    try:
        from flydsl._mlir.dialects import fly as _fly  # noqa: F401
        import flydsl.expr as fx
        from flydsl.expr import iluvatar as ilu
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"FlyIXDL python API unavailable: {exc}")

    if dtype is None:
        dtype = fx.Float16

    with Context() as ctx, Location.unknown():
        ctx.allow_unregistered_dialects = True
        module = Module.create()
        with InsertionPoint(module.body):
            ftype = ir.FunctionType.get([], [])
            func = ir.Operation.create(
                "func.func",
                attributes={
                    "function_type": ir.TypeAttr.get(ftype),
                    "sym_name": ir.StringAttr.get("k"),
                },
                regions=1,
            )
            blk = func.regions[0].blocks.append()
        with InsertionPoint(blk):
            tile = ilu.make_smem_tile(M, K, dtype, **kwargs)
            ir.Operation.create("func.return")
        assert module.operation.verify()
        return str(tile.type), str(module)


@pytest.mark.l1a_compile_no_target_dialect
def test_make_smem_tile_layout_m16():
    ty, _ = _trace_make_smem_tile(16, 32, swizzle=(1, 6, 2))
    # Playbook 5.3 (M==16): shape ((2,8),(K/2,2)) stride ((1,K),(2,8*K)); K=32.
    assert "shared" in ty
    assert "S<1,6,2>" in ty
    assert "((2,8),(16,2)):((1,32),(2,256))" in ty


@pytest.mark.l1a_compile_no_target_dialect
def test_make_smem_tile_layout_m32_subtiled():
    ty, _ = _trace_make_smem_tile(32, 32, swizzle=(1, 6, 2))
    # Playbook 5.3 (M==16*n): n=2 -> ((2,8,2),(16,2)):((1,32,512),(2,256)).
    assert "((2,8,2),(16,2)):((1,32,512),(2,256))" in ty


@pytest.mark.l1a_compile_no_target_dialect
def test_make_smem_tile_for_mma_defaults_to_canonical_swizzle():
    ty, _ = _trace_make_smem_tile(16, 32, for_mma=object())
    assert "S<1,6,2>" in ty


@pytest.mark.l1a_compile_no_target_dialect
def test_make_smem_tile_swizzle_byte_converts_to_element_space():
    # byte-space (1,7,2) for f16 (2 bytes) == element-space (1,6,2).
    ty, _ = _trace_make_smem_tile(16, 32, swizzle=None, swizzle_byte=(1, 7, 2))
    assert "S<1,6,2>" in ty


@pytest.mark.l1a_compile_no_target_dialect
def test_make_smem_tile_swizzle_mod_rowxfb8_int8():
    """int8 rowxfb8 Row8b tile uses the modular SwizzleMod (printed `SM<...>`),
    not the XOR Swizzle (`S<...>`)."""
    from flydsl.expr import Int8  # noqa: F401

    ty, _ = _trace_make_smem_tile(16, 64, dtype=Int8, swizzle_mod=(2, 6, 2))
    assert "SM<2,6,2>" in ty
    assert "S<" not in ty.replace("SM<", "")  # no XOR swizzle emitted


@pytest.mark.l1a_compile_no_target_dialect
def test_make_smem_tile_swizzle_mod_byte_converts_to_element_space_int8():
    # byte-space (2,6,2) for int8 (1 byte) == element-space (2,6,2).
    from flydsl.expr import Int8  # noqa: F401

    ty, _ = _trace_make_smem_tile(16, 64, dtype=Int8, swizzle_mod_byte=(2, 6, 2))
    assert "SM<2,6,2>" in ty


# ---------------------------------------------------------------------------
# L1b: convert-fly-to-ixdl emits the expected ixdl.* ops
# ---------------------------------------------------------------------------


@pytest.mark.l1b_target_dialect
def test_convert_emits_ixdl_mmad():
    _require_flyixdl_fly_opt()
    result = _run_fly_opt(
        _MLIR_DIR / "Conversion" / "FlyToIXDL" / "mr_mmad.mlir", *_IXDL_PIPELINE
    )
    assert result.returncode == 0, result.stderr
    assert "ixdl.mmad" in result.stdout


@pytest.mark.l1b_target_dialect
def test_convert_emits_ixdl_cp_async():
    _require_flyixdl_fly_opt()
    result = _run_fly_opt(
        _MLIR_DIR / "Conversion" / "FlyToIXDL" / "mr_async_copy.mlir", *_IXDL_PIPELINE
    )
    assert result.returncode == 0, result.stderr
    assert "ixdl.cp.async" in result.stdout


@pytest.mark.l1b_target_dialect
def test_convert_slb_load_lowers_to_llvm_load():
    _require_flyixdl_fly_opt()
    result = _run_fly_opt(
        _MLIR_DIR / "Conversion" / "FlyToIXDL" / "mr_slb_load.mlir", *_IXDL_PIPELINE
    )
    assert result.returncode == 0, result.stderr
    assert "llvm.load" in result.stdout


@pytest.mark.l1b_target_dialect
def test_convert_async_copy_rejects_non_shared_dst():
    """Address-space contract: a non-shared G2S destination must fail to legalize."""
    _require_flyixdl_fly_opt()
    result = _run_fly_opt(
        _MLIR_DIR / "Conversion" / "FlyToIXDL" / "mr_async_copy_neg.mlir", *_IXDL_PIPELINE
    )
    assert result.returncode != 0
    assert "failed to legalize operation 'fly.copy_atom_call'" in result.stderr
