# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar Softmax V1 device tests (f16/bf16/f32 row-wise)."""

from __future__ import annotations

import ctypes
import json
import math
import os
import sys
import time
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.iluvatar_lower]

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA-compatible Iluvatar device is not available", allow_module_level=True)

from kernels.norm.iluvatar.softmax_kernel import compile_iluvatar_softmax  # noqa: E402


def _require_perf_enabled() -> None:
    if os.environ.get("FLYDSL_ILUVATAR_RUN_SOFTMAX_PERF", "").lower() not in {"1", "true", "yes", "on"}:
        pytest.skip("set FLYDSL_ILUVATAR_RUN_SOFTMAX_PERF=1 to run Iluvatar softmax perf comparison")


def _configure_iluvatar_env(monkeypatch) -> None:
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)


def _torch_dtype(dtype_str: str):
    if dtype_str == "f16":
        return torch.float16
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "f32":
        return torch.float32
    raise ValueError(dtype_str)


def _bench_gpu_us(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) * 1000.0 / iters)


def _bench_wall_us(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return float((t1 - t0) * 1e6 / iters)


def _maybe_bench_ixdnn_softmax_us(x, *, warmup: int, iters: int):
    """Best-effort benchmark of ixdnn/cudnn softmax forward.

    Returns:
      - float(us): measured average latency
      - None: ixdnn/cudnn softmax API not available in current runtime
    """
    try:
        lib = ctypes.CDLL("libcudnn.so")
    except OSError:
        return None

    lib.cudnnCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.cudnnCreate.restype = ctypes.c_int
    lib.cudnnDestroy.argtypes = [ctypes.c_void_p]
    lib.cudnnDestroy.restype = ctypes.c_int
    lib.cudnnCreateTensorDescriptor.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.cudnnCreateTensorDescriptor.restype = ctypes.c_int
    lib.cudnnDestroyTensorDescriptor.argtypes = [ctypes.c_void_p]
    lib.cudnnDestroyTensorDescriptor.restype = ctypes.c_int
    lib.cudnnSetTensor4dDescriptor.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.cudnnSetTensor4dDescriptor.restype = ctypes.c_int
    lib.cudnnSoftmaxForward.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.cudnnSoftmaxForward.restype = ctypes.c_int

    CUDNN_TENSOR_NCHW = 0
    CUDNN_SOFTMAX_ACCURATE = 1
    CUDNN_SOFTMAX_MODE_CHANNEL = 1
    CUDNN_DATA_FLOAT = 0
    CUDNN_DATA_HALF = 2
    CUDNN_DATA_BFLOAT16 = 9

    m, n = int(x.shape[0]), int(x.shape[1])
    if x.dtype == torch.float16:
        data_type = CUDNN_DATA_HALF
    elif x.dtype == torch.bfloat16:
        data_type = CUDNN_DATA_BFLOAT16
    elif x.dtype == torch.float32:
        data_type = CUDNN_DATA_FLOAT
    else:
        return None

    handle = ctypes.c_void_p()
    x_desc = ctypes.c_void_p()
    y_desc = ctypes.c_void_p()
    y = torch.empty_like(x)
    alpha = ctypes.c_float(1.0)
    beta = ctypes.c_float(0.0)

    def _check(code):
        return code == 0

    if not _check(lib.cudnnCreate(ctypes.byref(handle))):
        return None
    if not _check(lib.cudnnCreateTensorDescriptor(ctypes.byref(x_desc))):
        lib.cudnnDestroy(handle)
        return None
    if not _check(lib.cudnnCreateTensorDescriptor(ctypes.byref(y_desc))):
        lib.cudnnDestroyTensorDescriptor(x_desc)
        lib.cudnnDestroy(handle)
        return None

    try:
        if not _check(lib.cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, data_type, m, n, 1, 1)):
            return None
        if not _check(lib.cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, data_type, m, n, 1, 1)):
            return None

        x_ptr = ctypes.c_void_p(x.data_ptr())
        y_ptr = ctypes.c_void_p(y.data_ptr())
        alpha_p = ctypes.byref(alpha)
        beta_p = ctypes.byref(beta)

        def run_ixdnn():
            rc = lib.cudnnSoftmaxForward(
                handle,
                CUDNN_SOFTMAX_ACCURATE,
                CUDNN_SOFTMAX_MODE_CHANNEL,
                alpha_p,
                x_desc,
                x_ptr,
                beta_p,
                y_desc,
                y_ptr,
            )
            if rc != 0:
                raise RuntimeError(f"cudnnSoftmaxForward failed: status={rc}")

        y.fill_(float("nan"))
        run_ixdnn()
        torch.cuda.synchronize()
        ref = torch.softmax(x, dim=1)
        y_f32 = y.to(torch.float32)
        if not torch.isfinite(y_f32).all().item():
            return None
        max_err = (y_f32 - ref.to(torch.float32)).abs().max().item()
        if (not math.isfinite(max_err)) or max_err > 5e-2:
            return None

        us = _bench_wall_us(run_ixdnn, warmup=warmup, iters=iters)
        if us < 1.0:
            return None
        return us
    finally:
        lib.cudnnDestroyTensorDescriptor(x_desc)
        lib.cudnnDestroyTensorDescriptor(y_desc)
        lib.cudnnDestroy(handle)


@pytest.mark.parametrize(
    "m,n,dtype_str,atol",
    (
        (64, 256, "f16", 1e-2),
        (64, 250, "f16", 1e-2),
        (16, 7424, "f16", 1e-2),
        (32, 1024, "bf16", 2e-2),
        (16, 7296, "bf16", 2e-2),
        (31, 1003, "bf16", 2e-2),
        (64, 256, "f32", 1e-2),
        (64, 250, "f32", 1e-2),
        (16, 7424, "f32", 1e-2),
        (0, 256, "f16", 1e-2),
    ),
)
def test_iluvatar_softmax_rowwise_correctness(m, n, dtype_str, atol, monkeypatch):
    _configure_iluvatar_env(monkeypatch)
    torch.manual_seed(42)

    launch = compile_iluvatar_softmax(N=n, dtype=dtype_str)
    a_f32 = (torch.rand((m, n), device="cuda", dtype=torch.float32) * 4.0) - 2.0
    expected = torch.softmax(a_f32, dim=1).to(torch.float32)

    torch_dtype = _torch_dtype(dtype_str)
    x = a_f32.to(torch_dtype).contiguous()
    out = torch.empty((m, n), device="cuda", dtype=torch_dtype).contiguous()

    ret = launch(x, out, m)
    assert ret is out
    torch.cuda.synchronize()

    if m == 0:
        assert out.numel() == 0
        return

    got = out.to(torch.float32)
    torch.testing.assert_close(
        got,
        expected,
        rtol=2e-2,
        atol=atol,
        msg=f"Iluvatar softmax mismatch: shape=({m},{n}) dtype={dtype_str}",
    )


def test_iluvatar_softmax_m0_noop(monkeypatch):
    _configure_iluvatar_env(monkeypatch)

    launch = compile_iluvatar_softmax(N=256, dtype="f16")
    x = torch.empty((0, 256), device="cuda", dtype=torch.float16).contiguous()
    out = torch.empty((0, 256), device="cuda", dtype=torch.float16).contiguous()

    ret = launch(x, out, 0)
    assert ret is out
    assert out.numel() == 0


def test_iluvatar_softmax_compile_time_guards():
    with pytest.raises(ValueError, match="N must be > 0"):
        compile_iluvatar_softmax(N=0, dtype="f16")
    with pytest.raises(ValueError, match="dtype must be one of"):
        compile_iluvatar_softmax(N=128, dtype="f64")


def test_iluvatar_softmax_runtime_guards(monkeypatch):
    _configure_iluvatar_env(monkeypatch)

    M, N = 4, 16
    launch = compile_iluvatar_softmax(N=N, dtype="f16")
    x = torch.randn((M, N), device="cuda", dtype=torch.float16).contiguous()
    out = torch.empty((M, N), device="cuda", dtype=torch.float16).contiguous()

    with pytest.raises(ValueError, match="expected x shape \\(M,N\\)="):
        launch(x, out, M + 1)

    out_f32 = torch.empty((M, N), device="cuda", dtype=torch.float32).contiguous()
    with pytest.raises(ValueError, match=r"out dtype must be torch\.float16"):
        launch(x, out_f32, M)

    x_nc = torch.randn((N, M), device="cuda", dtype=torch.float16).t()
    assert tuple(x_nc.shape) == (M, N) and not x_nc.is_contiguous()
    with pytest.raises(ValueError, match="x must be contiguous"):
        launch(x_nc, out, M)

    with pytest.raises(ValueError, match="out must not overlap with x"):
        launch(x, x, M)


def test_iluvatar_softmax_perf_vs_torch(monkeypatch):
    _require_perf_enabled()
    _configure_iluvatar_env(monkeypatch)

    warmup = int(os.environ.get("FLYDSL_ILUVATAR_SOFTMAX_PERF_WARMUP", "20"))
    iters = int(os.environ.get("FLYDSL_ILUVATAR_SOFTMAX_PERF_ITERS", "100"))
    metrics = {}

    for m, n, dtype_str in (
        (4096, 8192, "f16"),
        (4096, 8192, "bf16"),
        (4096, 8192, "f32"),
    ):
        launch = compile_iluvatar_softmax(N=n, dtype=dtype_str)

        torch.manual_seed(123)
        a_f32 = (torch.rand((m, n), device="cuda", dtype=torch.float32) * 4.0) - 2.0
        torch_dtype = _torch_dtype(dtype_str)
        x = a_f32.to(torch_dtype).contiguous()
        out = torch.empty((m, n), device="cuda", dtype=torch_dtype).contiguous()

        stream = torch.cuda.current_stream()

        def run_flydsl():
            launch(x, out, m, stream=stream)

        def run_torch():
            torch.softmax(x, dim=1)

        flydsl_us = _bench_gpu_us(run_flydsl, warmup=warmup, iters=iters)
        torch_us = _bench_gpu_us(run_torch, warmup=warmup, iters=iters)
        ixdnn_us = _maybe_bench_ixdnn_softmax_us(x, warmup=warmup, iters=iters)
        speedup = torch_us / flydsl_us if flydsl_us > 0 else 0.0
        speedup_ixdnn = (ixdnn_us / flydsl_us) if (ixdnn_us is not None and flydsl_us > 0) else None

        print(
            f"[iluvatar-softmax-perf] shape={m}x{n} dtype={dtype_str} "
            f"flydsl={flydsl_us:.1f}us torch={torch_us:.1f}us "
            f"ixdnn={f'{ixdnn_us:.1f}us' if ixdnn_us is not None else 'N/A'} "
            f"ixblas=N/A speedup_torch={speedup:.3f}x "
            f"speedup_ixdnn={f'{speedup_ixdnn:.3f}x' if speedup_ixdnn is not None else 'N/A'}"
        )

        point_metrics = {
            "latency_us": float(flydsl_us),
            "torch_latency_us": float(torch_us),
            "speedup_torch": float(speedup),
        }
        if ixdnn_us is not None:
            point_metrics["ixdnn_latency_us"] = float(ixdnn_us)
            if speedup_ixdnn is not None:
                point_metrics["speedup_ixdnn"] = float(speedup_ixdnn)
        metrics[f"{m}x{n}.{dtype_str}"] = point_metrics

    print("PERF_CASE_JSON=" + json.dumps({"metrics": metrics}, sort_keys=True))
