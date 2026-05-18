#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# 保存为 FlyDSL 仓库根目录下的 minimal_iluvatar_kernel_store.py，与 build-fly/ 同级。

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_pkg = _ROOT / "build-fly" / "python_packages"
if not _pkg.is_dir():
    sys.exit(f"找不到 {_pkg}，请先在本仓库完成 CMake 构建（见上文）。")
sys.path.insert(0, str(_pkg))
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("FLYDSL_PYTHON_PACKAGES", str(_pkg))

os.environ.setdefault("FLYDSL_COMPILE_BACKEND", "iluvatar")
os.environ.setdefault("FLYDSL_RUNTIME_KIND", "iluvatar")
os.environ.setdefault("ARCH", "ivcore11")
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch


@flyc.kernel
def store_one(out: fx.Tensor):
    out[0] = fx.Int32(7)


@flyc.jit
def launch_store_one(out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    store_one(out).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        sys.exit("需要 torch.cuda 可用（CUDA 兼容栈）。请安装 PyTorch 并确认设备可见。")
    out = torch.zeros(1, device="cuda", dtype=torch.int32)
    launch_store_one(out)
    torch.cuda.synchronize()
    v = int(out.cpu().item())
    if v != 7:
        sys.exit(f"读回失败: 期望 7，得到 {v}")
    print("OK: store kernel wrote int32(7) to out[0]; host read-back matches.")
