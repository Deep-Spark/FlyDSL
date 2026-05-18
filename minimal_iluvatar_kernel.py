#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# 保存为 FlyDSL 仓库根目录下的 minimal_iluvatar_kernel.py，与 build-fly/ 同级。

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---- 仓库根 = 本文件所在目录（请把本文件放在 FlyDSL 根目录）----
_ROOT = Path(__file__).resolve().parent
_pkg = _ROOT / "build-fly" / "python_packages"
if not _pkg.is_dir():
    sys.exit(f"找不到 {_pkg}，请先在本仓库完成 CMake 构建（见上文）。")
sys.path.insert(0, str(_pkg))
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("FLYDSL_PYTHON_PACKAGES", str(_pkg))

# ---- Iluvatar 编译 / 运行时（必须在 import flydsl 之前设置）----
os.environ.setdefault("FLYDSL_COMPILE_BACKEND", "iluvatar")
os.environ.setdefault("FLYDSL_RUNTIME_KIND", "iluvatar")
os.environ.setdefault("ARCH", "ivcore11")
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import flydsl.compiler as flyc
import flydsl.expr as fx


@flyc.kernel
def empty_kernel():
    pass


@flyc.jit
def launch_empty(stream: fx.Stream = fx.Stream(None)):
    empty_kernel().launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)


if __name__ == "__main__":
    launch_empty()
    print("OK: minimal Iluvatar FlyDSL kernel launched (empty body, 1x1x1).")
