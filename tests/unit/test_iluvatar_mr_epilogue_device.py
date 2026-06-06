# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Opt-in Iluvatar MR epilogue-only device correctness tests.

Exercises the three ``kernels.iluvatar_mr_epilogue`` store helpers directly.
Accumulator fragments are initialized in register memory (no G2S / S2R / MMA),
then stored through the shared HGEMM epilogue helper:

* ``no_c_read + tiled``: f32 accumulator -> fp16 tiled C store.
* ``no_c_read + shfl``: f32 accumulator -> fp16 shuffle/packed-i32 C store.
* ``read_c_accum``: fp32 C load -> accumulator update -> fp32 tiled C store.

Set ``FLYDSL_ILUVATAR_RUN_MR_EPILOGUE=1`` to run (needs an Iluvatar device).

Stage coverage notes
--------------------

* Original tests use ``grid=(1,1,1)`` and a single 16x32 warp tile, so they cannot
  catch multi-CTA ``bid_x`` / ``bid_y`` C-slice addressing bugs.
* ``test_iluvatar_mr_epilogue_multi_cta_store_device`` launches a 2x2 grid over
  128x128 logical C with per-CTA 64x64 warp tiles.
"""

import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l2_device]

_REPO_ROOT = Path(__file__).resolve().parents[2]

EPILOGUE_STORE_CASES = ("no_c_read_tiled", "no_c_read_shfl", "read_c_accum")

EPILOGUE_M = 16
EPILOGUE_N = 32
EPILOGUE_WARP_ATOMS_M = EPILOGUE_M // 16
EPILOGUE_WARP_ATOMS_N = EPILOGUE_N // 16

MULTI_CTA_EPILOGUE_M = 64
MULTI_CTA_EPILOGUE_N = 64
MULTI_CTA_FULL_M = MULTI_CTA_EPILOGUE_M * 2
MULTI_CTA_FULL_N = MULTI_CTA_EPILOGUE_N * 2
MULTI_CTA_WARP_ATOMS_M = MULTI_CTA_EPILOGUE_M // 16
MULTI_CTA_WARP_ATOMS_N = MULTI_CTA_EPILOGUE_N // 16
MULTI_CTA_STORE_CASES = ("no_c_read_tiled", "no_c_read_shfl")

EPILOGUE_ACC_VALUE = 7.5
EPILOGUE_ACC_DELTA = 3.25


def _require_enabled() -> None:
    if os.environ.get("FLYDSL_ILUVATAR_RUN_MR_EPILOGUE", "").lower() not in {"1", "true", "yes", "on"}:
        pytest.skip("set FLYDSL_ILUVATAR_RUN_MR_EPILOGUE=1 to run Iluvatar MR epilogue device tests")


def _require_imports():
    try:
        import flydsl

        generated_pkg = Path(os.environ.get("FLYDSL_PYTHON_PACKAGES", _REPO_ROOT / "build-fly" / "python_packages"))
        generated_flydsl = generated_pkg / "flydsl"
        if generated_flydsl.is_dir() and str(generated_flydsl) not in flydsl.__path__:
            flydsl.__path__.append(str(generated_flydsl))

        import flydsl.compiler as flyc
        import flydsl.expr as fx
        import flydsl.expr.ixdl as ixdl
        from kernels.iluvatar_mr_common import ATOM_K, ATOM_M, ATOM_N, WARP_SIZE
        from kernels.iluvatar_mr_epilogue import (
            mr_hgemm_epilogue_store_read_c_accum,
            mr_hgemm_epilogue_store_shfl,
            mr_hgemm_epilogue_store_tiled,
        )
    except ModuleNotFoundError as exc:
        pytest.fail(f"FlyDSL Python package is not importable: {exc}")
    return (
        flyc,
        fx,
        ixdl,
        ATOM_K,
        ATOM_M,
        ATOM_N,
        WARP_SIZE,
        mr_hgemm_epilogue_store_shfl,
        mr_hgemm_epilogue_store_tiled,
        mr_hgemm_epilogue_store_read_c_accum,
    )


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        pytest.skip(f"torch is required for Iluvatar MR epilogue device tests: {exc}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA-compatible Iluvatar device is not available")
    return torch


def _configure_iluvatar_env(monkeypatch) -> None:
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)


def _compile_epilogue_kernel(
    flyc,
    fx,
    ixdl,
    atom_k,
    atom_m,
    atom_n,
    warp_size,
    mr_hgemm_epilogue_store_shfl,
    mr_hgemm_epilogue_store_tiled,
    mr_hgemm_epilogue_store_read_c_accum,
    store_case: str,
    *,
    multi_cta: bool = False,
):
    shfl_store = store_case == "no_c_read_shfl"
    tiled_store = store_case == "no_c_read_tiled"
    read_c_accum = store_case == "read_c_accum"
    out_fp16 = not read_c_accum
    cta_m = MULTI_CTA_EPILOGUE_M if multi_cta else EPILOGUE_M
    cta_n = MULTI_CTA_EPILOGUE_N if multi_cta else EPILOGUE_N
    full_m = MULTI_CTA_FULL_M if multi_cta else EPILOGUE_M
    full_n = MULTI_CTA_FULL_N if multi_cta else EPILOGUE_N
    warp_atoms_m = MULTI_CTA_WARP_ATOMS_M if multi_cta else EPILOGUE_WARP_ATOMS_M
    warp_atoms_n = MULTI_CTA_WARP_ATOMS_N if multi_cta else EPILOGUE_WARP_ATOMS_N
    grid = (2, 2, 1) if multi_cta else (1, 1, 1)

    @flyc.kernel(known_block_size=[warp_size, 1, 1])
    def epilogue_kernel(C: fx.Tensor):
        tid = fx.thread_idx.x
        lane_id = tid % fx.Int32(warp_size)
        bid_x, bid_y, _ = fx.block_idx
        gC_full = fx.make_view(fx.get_iter(C), fx.make_layout((full_m, full_n), (full_n, 1)))
        if fx.const_expr(multi_cta):
            gC_warp = fx.slice(
                fx.flat_divide(gC_full, (cta_m, cta_n)),
                (None, None, bid_x, bid_y),
            )
            c_global_n = full_n
        else:
            gC_warp = gC_full
            c_global_n = cta_n

        mma_atom = fx.make_mma_atom(
            ixdl.MRMma(atom_m, atom_n, atom_k, fx.Float16, fx.Float16, fx.Float32)
        )
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (1, 1, 1)))
        thr_mma = tiled_mma.thr_slice(lane_id)
        gC_atoms = fx.flat_divide(gC_warp, (atom_m, atom_n))

        accs = []
        for im in fx.range_constexpr(warp_atoms_m):
            row = []
            for jn in fx.range_constexpr(warp_atoms_n):
                c_tile = fx.slice(gC_atoms, (None, None, im, jn))
                acc = thr_mma.make_fragment_C(c_tile)
                if fx.const_expr(read_c_accum):
                    copy_atom_c_f32 = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
                    tiled_copy_c_f32 = fx.make_tiled_copy_C(copy_atom_c_f32, tiled_mma)
                    thr_copy_c_f32 = tiled_copy_c_f32.get_slice(lane_id)
                    fx.copy(
                        copy_atom_c_f32,
                        thr_copy_c_f32.partition_S(c_tile),
                        thr_copy_c_f32.retile(acc),
                        pred=None,
                    )
                    acc.store(acc.load() + EPILOGUE_ACC_DELTA)
                else:
                    acc.fill(EPILOGUE_ACC_VALUE)
                row.append(acc)
            accs.append(row)

        if fx.const_expr(shfl_store):
            mr_hgemm_epilogue_store_shfl(
                lane_id=lane_id,
                accs=accs,
                gC_warp=gC_warp,
                c_global_n=c_global_n,
                warp_atoms_m=warp_atoms_m,
                warp_atoms_n=warp_atoms_n,
            )
        elif fx.const_expr(tiled_store):
            mr_hgemm_epilogue_store_tiled(
                lane_id=lane_id,
                accs=accs,
                gC_warp=gC_warp,
                tiled_mma=tiled_mma,
                warp_atoms_m=warp_atoms_m,
                warp_atoms_n=warp_atoms_n,
            )
        else:
            mr_hgemm_epilogue_store_read_c_accum(
                lane_id=lane_id,
                accs=accs,
                gC_warp=gC_warp,
                tiled_mma=tiled_mma,
                warp_atoms_m=warp_atoms_m,
                warp_atoms_n=warp_atoms_n,
            )

    @flyc.jit
    def launch(C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        epilogue_kernel(C).launch(grid=grid, block=(warp_size, 1, 1), stream=stream)

    return launch, out_fp16, read_c_accum, full_m, full_n


@pytest.mark.parametrize("store_case", EPILOGUE_STORE_CASES)
def test_iluvatar_mr_epilogue_fragment_store_device(store_case, monkeypatch):
    """Store initialized accumulator fragments via the mode-specific epilogue helper."""

    _require_enabled()
    flyc, fx, ixdl, atom_k, atom_m, atom_n, warp_size, store_shfl, store_tiled, store_read_c = _require_imports()
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)

    launch, out_fp16, read_c_accum, full_m, full_n = _compile_epilogue_kernel(
        flyc,
        fx,
        ixdl,
        atom_k,
        atom_m,
        atom_n,
        warp_size,
        store_shfl,
        store_tiled,
        store_read_c,
        store_case,
    )
    if out_fp16:
        C = torch.empty((full_m, full_n), device="cuda", dtype=torch.float16)
        expected = torch.full((full_m, full_n), EPILOGUE_ACC_VALUE, device="cuda", dtype=torch.float16)
    else:
        row = torch.arange(full_m, device="cuda", dtype=torch.float32).view(full_m, 1)
        col = torch.arange(full_n, device="cuda", dtype=torch.float32).view(1, full_n)
        C = row * 0.25 + col * 0.03125
        expected = C + EPILOGUE_ACC_DELTA if read_c_accum else C

    launch(C)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        C,
        expected,
        rtol=0,
        atol=0,
        msg=f"{store_case} epilogue mismatch",
    )


@pytest.mark.parametrize("store_case", MULTI_CTA_STORE_CASES)
def test_iluvatar_mr_epilogue_multi_cta_store_device(store_case, monkeypatch):
    """Store accumulator fragments through a 2x2 CTA grid (128x128 logical C)."""

    _require_enabled()
    flyc, fx, ixdl, atom_k, atom_m, atom_n, warp_size, store_shfl, store_tiled, store_read_c = _require_imports()
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)

    launch, out_fp16, read_c_accum, full_m, full_n = _compile_epilogue_kernel(
        flyc,
        fx,
        ixdl,
        atom_k,
        atom_m,
        atom_n,
        warp_size,
        store_shfl,
        store_tiled,
        store_read_c,
        store_case,
        multi_cta=True,
    )
    assert out_fp16 and not read_c_accum
    C = torch.empty((full_m, full_n), device="cuda", dtype=torch.float16)
    expected = torch.full((full_m, full_n), EPILOGUE_ACC_VALUE, device="cuda", dtype=torch.float16)

    launch(C)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        C,
        expected,
        rtol=0,
        atol=0,
        msg=f"{store_case} multi-CTA epilogue mismatch",
    )
