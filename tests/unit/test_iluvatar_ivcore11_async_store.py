# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Lightweight Ivcore11AsyncStore (SME Store series, shared -> global) factory tests.

Constructs ``Ivcore11AsyncStore`` atom types under the Iluvatar FlyIXDL bindings and
wraps them with ``make_copy_atom``. No GPU required.
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


def test_ivcore11_async_store_factories_construct_and_wrap():
    """Ivcore11AsyncStore factory builds atom types usable by make_copy_atom."""
    ir, fx, ixdl = _require_imports()

    with ir.Context(), ir.Location.unknown():
        for store_bytes in (64, 128, 256):
            store_ty = ixdl.Ivcore11AsyncStore(store_bytes)
            store_s = str(store_ty)
            assert "fly_ixdl.ivcore11.async_store" in store_s
            assert f"bytes = {store_bytes}" in store_s
            store_atom = fx.make_copy_atom(store_ty, fx.Float32)
            assert "ivcore11.async_store" in str(store_atom)

        # Convenience aliases produce the same types as the explicit factory.
        assert str(ixdl.Ivcore11AsyncStoreB64()) == str(ixdl.Ivcore11AsyncStore(64))
        assert str(ixdl.Ivcore11AsyncStoreB128()) == str(ixdl.Ivcore11AsyncStore(128))
        assert str(ixdl.Ivcore11AsyncStoreB256()) == str(ixdl.Ivcore11AsyncStore(256))


def test_ivcore11_async_store_rejects_bad_args():
    """Factory argument validation fails fast without touching the GPU."""
    ir, fx, ixdl = _require_imports()

    with ir.Context(), ir.Location.unknown():
        for bad in (0, 32, 100, 512):
            with pytest.raises(ValueError, match="64/128/256"):
                ixdl.Ivcore11AsyncStore(bad)


def test_ivcore11_async_store_exported_from_ixdl():
    """The store API must be part of the public flydsl.expr.ixdl surface."""
    _, _, ixdl = _require_imports()

    missing = [
        name
        for name in ("Ivcore11AsyncStore", "Ivcore11AsyncStoreB64", "Ivcore11AsyncStoreB128", "Ivcore11AsyncStoreB256")
        if not hasattr(ixdl, name)
    ]
    if missing:
        pytest.fail(f"ivcore11 async store API missing from flydsl.expr.ixdl: {', '.join(missing)}")
    for name in ("Ivcore11AsyncStore", "Ivcore11AsyncStoreB64", "Ivcore11AsyncStoreB128", "Ivcore11AsyncStoreB256"):
        assert name in ixdl.__all__
