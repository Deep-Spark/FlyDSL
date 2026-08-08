# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Lightweight CQ atom factory / import tests.

Construct ``CQMma`` / ``CQSmexCp`` / ``CQMtxLoadn`` under the Iluvatar FlyIXDL
bindings and wrap them with ``make_mma_atom`` / ``make_copy_atom``. No GPU or
production GEMM required. Also checks that the MR factories remain importable.
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

        import flydsl.expr as fx
        import flydsl.expr.ixdl as ixdl
        from flydsl._mlir import ir
    except ModuleNotFoundError as exc:
        pytest.fail(f"FlyDSL Python package is not importable: {exc}")
    return ir, fx, ixdl


def _require_cq_api(ixdl):
    missing = [
        name for name in ("CQMma", "CQSmexCp", "CQSmexCpMtx", "CQSmexCpPlain", "CQMtxLoadn") if not hasattr(ixdl, name)
    ]
    if missing:
        pytest.fail(f"CQ atom API missing from flydsl.expr.ixdl: {', '.join(missing)}")


def test_cq_atom_factories_construct_and_wrap():
    """CQ MMA / G2S / S2R factories build atom types usable by make_*_atom."""
    ir, fx, ixdl = _require_imports()
    _require_cq_api(ixdl)

    with ir.Context(), ir.Location.unknown():
        mma_ty = ixdl.CQMma(16, 16, 16, fx.Float16, fx.Float16, fx.Float32)
        mma_s = str(mma_ty)
        assert "fly_ixdl.cq.mma" in mma_s
        assert "16" in mma_s and "f16" in mma_s and "f32" in mma_s
        mma_atom = fx.make_mma_atom(mma_ty)
        assert "cq.mma" in str(mma_atom)

        g2s_ty = ixdl.CQSmexCp(rows=16, layout="mtx")
        g2s_s = str(g2s_ty)
        assert "fly_ixdl.cq.smex_cp" in g2s_s
        assert "layout = mtx" in g2s_s
        g2s_atom = fx.make_copy_atom(g2s_ty, fx.Int8)
        assert "cq.smex_cp" in str(g2s_atom)

        alias_ty = ixdl.CQSmexCpMtx(16)
        assert str(alias_ty) == g2s_s
        plain_ty = ixdl.CQSmexCpPlain(4)
        assert "layout = plain" in str(plain_ty)

        s2r_ty = ixdl.CQMtxLoadn(fx.Int8, pattern="loadn16", direction="row")
        s2r_s = str(s2r_ty)
        assert "fly_ixdl.cq.mtx_loadn" in s2r_s
        assert "loadn16" in s2r_s and "row" in s2r_s
        s2r_atom = fx.make_copy_atom(s2r_ty, fx.Int8)
        assert "cq.mtx_loadn" in str(s2r_atom)


def test_cq_factories_reject_bad_args():
    """Factory argument validation fails fast without touching the GPU."""
    ir, fx, ixdl = _require_imports()
    _require_cq_api(ixdl)

    with ir.Context(), ir.Location.unknown():
        with pytest.raises(ValueError, match="layout"):
            ixdl.CQSmexCp(rows=16, layout="swizzle")
        with pytest.raises(ValueError, match="pattern"):
            ixdl.CQMtxLoadn(fx.Int8, pattern="loadn32", direction="row")
        with pytest.raises(ValueError, match="direction"):
            ixdl.CQMtxLoadn(fx.Int8, pattern="loadn16", direction="diag")
        with pytest.raises(ValueError, match="8-bit or 16-bit"):
            ixdl.CQMtxLoadn(fx.Float32, pattern="loadn16", direction="row")


def test_mr_atom_api_still_importable():
    """CQ factories must not displace the existing MR Python API."""
    ir, fx, ixdl = _require_imports()

    missing = [name for name in ("MRMma", "MRAsyncCp", "MRAsyncCpCol", "MRAsyncCpRow16b") if not hasattr(ixdl, name)]
    if missing:
        pytest.fail(f"MR atom API missing from flydsl.expr.ixdl: {', '.join(missing)}")

    with ir.Context(), ir.Location.unknown():
        mr_mma = ixdl.MRMma(16, 16, 16, fx.Float16, fx.Float16, fx.Float32)
        assert "fly_ixdl.mr.mma" in str(mr_mma)
        mr_cp = ixdl.MRAsyncCpCol()
        assert "fly_ixdl.mr.async_copy" in str(mr_cp)
        assert fx.make_mma_atom(mr_mma) is not None
        assert fx.make_copy_atom(mr_cp, fx.Int8) is not None
