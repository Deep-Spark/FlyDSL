#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Fail-fast CUDA/CoreX preflight for Iluvatar device CI.

Checks that the host ``iluvatar`` kmod is loaded and that the COREX userspace
under ``COREX_ROOT`` can create a primary CUDA context. A mismatched kmod /
userspace pair often later shows up as torch.cuda segfaults that hide the real
``cuDevicePrimaryCtxRetain`` / driver-mismatch failure.
"""

import ctypes
import os
import re
import sys
from pathlib import Path


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as exc:
        return f"<unavailable: {exc}>"


_CUDA_ERROR_NAMES = {
    0: "CUDA_SUCCESS",
    1: "CUDA_ERROR_INVALID_VALUE",
    2: "CUDA_ERROR_OUT_OF_MEMORY",
    3: "CUDA_ERROR_NOT_INITIALIZED",
    100: "CUDA_ERROR_NO_DEVICE",
    101: "CUDA_ERROR_INVALID_DEVICE",
    201: "CUDA_ERROR_INVALID_CONTEXT",
    999: "CUDA_ERROR_UNKNOWN",
    # Iluvatar-specific / extended codes seen in the field.
    803: "IX_ERROR_SYSTEM_DRIVER_MISMATCH",
}


def _fmt_rc(rc: int) -> str:
    return f"{rc} ({_CUDA_ERROR_NAMES.get(rc, 'UNKNOWN')})"


def main() -> int:
    corex_root = Path(os.environ.get("COREX_ROOT", "")).expanduser()
    if not str(corex_root):
        print("::error::COREX_ROOT is required for CUDA/CoreX preflight", file=sys.stderr)
        return 1

    summary_path = Path(os.environ.get("CI_DEVICE_CUDA_PREFLIGHT_SUMMARY", "logs/device-cuda-preflight.md"))
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    kmod_sys = Path("/sys/module/iluvatar")
    kmod_loaded = kmod_sys.is_dir()
    kmod_version = _read_text(kmod_sys / "version") if kmod_loaded else "<not loaded>"
    kmod_srcversion = _read_text(kmod_sys / "srcversion") if kmod_loaded else "<not loaded>"

    userspace_version_h = corex_root / "kmd" / "kernel" / "version.h"
    userspace_version_text = (
        _read_text(userspace_version_h) if userspace_version_h.is_file() else "<missing>"
    )
    userspace_kmd = "<unknown>"
    userspace_drv = "<unknown>"
    userspace_date = "<unknown>"
    for key, pattern in (
        ("kmd", r'#define\s+ITR_KMD_VERSION\s+"([^"]*)"'),
        ("drv", r'#define\s+ITR_DRV_VERSION\s+"([^"]*)"'),
        ("date", r'#define\s+ITR_DRV_DATE\s+"([^"]*)"'),
    ):
        match = re.search(pattern, userspace_version_text)
        if not match:
            continue
        value = match.group(1).strip()
        if key == "kmd":
            userspace_kmd = value
        elif key == "drv":
            userspace_drv = value
        else:
            userspace_date = value

    libcuda = corex_root / "lib64" / "libcuda.so.1"
    lines = [
        "### CUDA / CoreX preflight",
        "",
        "| key | value |",
        "|---|---|",
        f"| COREX_ROOT | `{corex_root}` |",
        f"| libcuda.so.1 | `{libcuda}` |",
        f"| host kmod loaded | `{kmod_loaded}` |",
        f"| host kmod version | `{kmod_version}` |",
        f"| host kmod srcversion | `{kmod_srcversion}` |",
        f"| userspace ITR_KMD_VERSION | `{userspace_kmd}` |",
        f"| userspace ITR_DRV_VERSION | `{userspace_drv}` |",
        f"| userspace ITR_DRV_DATE | `{userspace_date}` |",
        f"| userspace version.h | `{userspace_version_h}` |",
    ]

    status = "ok"
    detail = ""
    try:
        if not kmod_loaded:
            raise RuntimeError(
                "host kernel module 'iluvatar' is not loaded (/sys/module/iluvatar missing)"
            )
        if not libcuda.is_file():
            raise RuntimeError(f"missing userspace libcuda: {libcuda}")

        lib = ctypes.CDLL(str(libcuda))
        rc_init = int(lib.cuInit(0))
        lines.append(f"| cuInit | `{_fmt_rc(rc_init)}` |")
        if rc_init != 0:
            raise RuntimeError(f"cuInit failed: {_fmt_rc(rc_init)}")

        count = ctypes.c_int()
        rc_count = int(lib.cuDeviceGetCount(ctypes.byref(count)))
        lines.append(f"| cuDeviceGetCount | `{_fmt_rc(rc_count)}` count={count.value} |")
        if rc_count != 0 or count.value <= 0:
            raise RuntimeError(f"cuDeviceGetCount failed: {_fmt_rc(rc_count)} count={count.value}")

        dev = ctypes.c_int()
        rc_dev = int(lib.cuDeviceGet(ctypes.byref(dev), 0))
        lines.append(f"| cuDeviceGet(0) | `{_fmt_rc(rc_dev)}` device={dev.value} |")
        if rc_dev != 0:
            raise RuntimeError(f"cuDeviceGet(0) failed: {_fmt_rc(rc_dev)}")

        ctx = ctypes.c_void_p()
        rc_ctx = int(lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev))
        lines.append(
            f"| cuDevicePrimaryCtxRetain | `{_fmt_rc(rc_ctx)}` ctx={hex(ctx.value or 0)} |"
        )
        if rc_ctx != 0:
            raise RuntimeError(
                "cuDevicePrimaryCtxRetain failed: "
                f"{_fmt_rc(rc_ctx)}. Often means host kmod and COREX userspace builds are "
                f"mismatched (host={kmod_version}, userspace ITR_KMD_VERSION={userspace_kmd}, "
                f"ITR_DRV_DATE={userspace_date}). Rebuild/load the kmd under "
                f"{corex_root}/kmd to match this COREX_ROOT."
            )
        rc_release = int(lib.cuDevicePrimaryCtxRelease(dev))
        lines.append(f"| cuDevicePrimaryCtxRelease | `{_fmt_rc(rc_release)}` |")
        if rc_release != 0:
            raise RuntimeError(f"cuDevicePrimaryCtxRelease failed: {_fmt_rc(rc_release)}")
    except Exception as exc:  # noqa: BLE001 - surface any preflight failure clearly
        status = "failed"
        detail = str(exc)

    lines.extend(["", f"- status: **{status}**"])
    if detail:
        lines.extend(["", f"- detail: `{detail}`"])
    summary = "\n".join(lines) + "\n"
    summary_path.write_text(summary, encoding="utf-8")
    print(summary, end="")
    if status != "ok":
        print(f"::error::CUDA/CoreX preflight failed: {detail}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
