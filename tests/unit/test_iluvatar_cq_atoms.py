# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""CQ MMA / async-copy / mtx_loadn Python factory construction tests.

Builds FlyIXDL atom *types* via ``flydsl.expr.ixdl`` factories and wraps them
with ``!fly.mma_atom`` / ``!fly.copy_atom`` (the same path ``make_mma_atom`` /
``make_copy_atom`` use). No GPU required. Also guards that MR factories remain
importable and constructible.
"""

import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.iluvatar_lower]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _require_imports():
    try:
        import flydsl

        generated_pkg = Path(os.environ.get("FLYDSL_PYTHON_PACKAGES", _REPO_ROOT / "build-fly" / "python_packages"))
        generated_flydsl = generated_pkg / "flydsl"
        if generated_flydsl.is_dir() and str(generated_flydsl) not in flydsl.__path__:
            flydsl.__path__.append(str(generated_flydsl))

        from flydsl._mlir import ir
        from flydsl._mlir.dialects import fly_ixdl  # noqa: F401  (register dialect)
        from flydsl._mlir.dialects.fly import CopyAtomType, MmaAtomType
        import flydsl.expr.ixdl as ixdl
    except ModuleNotFoundError as exc:
        pytest.fail(f"FlyDSL / FlyIXDL Python package is not importable: {exc}")
    return ir, CopyAtomType, MmaAtomType, ixdl


def test_cq_async_cp_type_roundtrip():
    ir, CopyAtomType, _, ixdl = _require_imports()
    with ir.Context(), ir.Location.unknown():
        cases = [
            (ixdl.CQAsyncCp(64, 64, 0), "64, 64, transpose = 0"),
            (ixdl.CQAsyncCp64x32Row(), "64, 32, transpose = 0"),
            (ixdl.CQAsyncCp1x64b64(), "1, 1024, transpose = 0"),
            (ixdl.CQAsyncCp64x16Row(), "64, 16, transpose = 0"),
            (ixdl.CQAsyncCp64x16Col(), "64, 16, transpose = 1"),
        ]
        for op, shape_frag in cases:
            text = str(op)
            assert "cq.async_copy<" in text
            assert shape_frag in text
            assert ir.Type.parse(text) == op

        atom = CopyAtomType.get(copy_op=ixdl.CQAsyncCp64x64Row(), val_bits=8)
        assert "fly.copy_atom" in str(atom)
        assert "cq.async_copy<64, 64, transpose = 0>" in str(atom)
        assert ", 8>" in str(atom)


def test_cq_mtx_loadn_type_roundtrip():
    ir, CopyAtomType, _, ixdl = _require_imports()
    with ir.Context(), ir.Location.unknown():
        op = ixdl.CQMtxLoadn(ixdl.CQMtxPattern.Loadn16, ixdl.CQMtxDir.Row, 16, x2=True)
        assert "cq.mtx_loadn<pattern = 0, dir = 0, b = 16, x2 = 1>" in str(op)
        assert ir.Type.parse(str(op)) == op

        op64 = ixdl.CQMtxLoadn(ixdl.CQMtxPattern.Loadn64, ixdl.CQMtxDir.Col, 8, x2=False)
        assert "pattern = 1, dir = 1, b = 8, x2 = 0" in str(op64)
        assert ir.Type.parse(str(op64)) == op64

        atom = CopyAtomType.get(copy_op=op, val_bits=16)
        assert "fly.copy_atom" in str(atom)
        assert "cq.mtx_loadn<" in str(atom)


def test_cq_mma_type_roundtrip():
    ir, _, MmaAtomType, ixdl = _require_imports()
    with ir.Context(), ir.Location.unknown():
        f16 = ir.F16Type.get()
        f32 = ir.F32Type.get()
        i8 = ir.IntegerType.get_signless(8)
        i32 = ir.IntegerType.get_signless(32)

        mma = ixdl.CQMma(16, 16, 16, f16, f16, f32)
        assert "cq.mma<16, 16, 16, (f16, f16) -> f32>" in str(mma)
        assert ir.Type.parse(str(mma)) == mma

        long_mtx = ixdl.CQMma(32, 32, 16, f16, f16, f32)
        assert "cq.mma<32, 32, 16," in str(long_mtx)
        assert ir.Type.parse(str(long_mtx)) == long_mtx

        igemm = ixdl.CQMma(16, 16, 32, i8, i8, i32)
        assert "cq.mma<16, 16, 32," in str(igemm)
        assert ir.Type.parse(str(igemm)) == igemm

        atom = MmaAtomType.get(mma_op=mma)
        assert str(atom) == "!fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 16, (f16, f16) -> f32>>"


def test_mr_factories_unaffected():
    """MR API remains constructible alongside CQ (no export / binding regression)."""
    ir, CopyAtomType, MmaAtomType, ixdl = _require_imports()
    with ir.Context(), ir.Location.unknown():
        f16 = ir.F16Type.get()
        f32 = ir.F32Type.get()

        mr_mma = ixdl.MRMma(16, 16, 16, f16, f16, f32)
        assert "mr.mma<" in str(mr_mma)
        assert "cq.mma<" not in str(mr_mma)
        assert ir.Type.parse(str(mr_mma)) == mr_mma
        assert "mr.mma<" in str(MmaAtomType.get(mma_op=mr_mma))

        mr_cp = ixdl.MRAsyncCp(ixdl.SMESwizzle.NoSwizzle)
        assert "mr.async_copy<swizzle = 0>" in str(mr_cp)
        assert ir.Type.parse(str(mr_cp)) == mr_cp
        atom = CopyAtomType.get(copy_op=ixdl.MRAsyncCpRow16b(), val_bits=16)
        assert "mr.async_copy<swizzle = 3>" in str(atom)
