# FlyDSL 在 Iluvatar 上的构建指南（无需安装 ROCm）

> **路径约定**：下文中的 Fly 相关路径均相对于 **FlyDSL 仓库根**。通过 skills hub 的 **`link_cursor.sh`** 注入后，本文档在目标仓库的常见落点为 **`.cursor/docs/flydsl/iluvatar-flydsl-build-without-rocm.md`**；路线图为同目录下的 **`iluvatar-flydsl-roadmap.md`**。**`link_cc.sh`** 则对应 **`.claude/docs-hub/flydsl/`** 下同名文件。本文仅覆盖「无 ROCm 的本机构建」，阶段划分仍以路线图与 `.claude/skills/iluvatar-backend-bringup/SKILL.md` 为准。

## 与默认（ROCm / `rocdl`）流程的本质区别

| 项目 | 默认 `README` / `scripts/build.sh` | Iluvatar-only |
|------|-----------------------------------|----------------|
| CMake 后端 | `FLYDSL_BACKENDS` 默认为 **`rocdl`**（见 `cmake/FlyDSLBackends.cmake`） | 显式 **`-DFLYDSL_BACKENDS=iluvatar`** |
| MLIR 来源 | 常用 `scripts/build_llvm.sh` 构建的 **上游** `llvm-project`，安装到 `mlir_install` | 必须指向 **带 `IXGPU` / `IXDL` 的厂商 MLIR**（**ixcc** 构建树中的 `lib/cmake/mlir`）。请使用 ixcc 的 **`22.x` 分支** 构建（与当前 FlyDSL Iluvatar 管线对齐；其他分支未在此文档范围验证）。上游 LLVM **不含** `MLIRIXDLDialect`，无法单独支撑 `FlyIXDL` / `FlyToIXDL` 链接 |
| GPU 运行时 | `lib/Runtime/ROCm` 会 `find_package(hip …)`，依赖 ROCm | 仅启用 `iluvatar` 时 **不会** 进入 `lib/Runtime/ROCm`，**不要求** `/opt/rocm` 或 HIP |
| Iluvatar JIT 运行时 | 默认不编 | `lib/Runtime/Iluvatar` 通过 **`find_package(CUDAToolkit REQUIRED)`** 链接 **`CUDA::cuda_driver`**。此处「CUDA」指 **CoreX 等 CUDA 兼容工具包** 的安装根目录，请用 **`-DCUDAToolkit_ROOT=<CoreX 安装路径>`** 告诉 CMake（与技能文档中的 `CUDAToolkit_ROOT=/path/to/corex` 一致） |

结论：**没有 ROCm 可以完整配置/编译 Iluvatar 栈**；但不能省掉 **厂商 MLIR + CoreX（CUDAToolkit）**。

## 前置条件（检查清单）

1. **C/C++ 工具链**：`cmake`（≥3.20）、C++17、建议 `ninja`。
2. **Python**：3.10+；**强烈建议使用仓库旁虚拟环境**（`python -m venv .venv`），避免系统 Python 的 PEP 668 / 缺包问题。
3. **Python 构建依赖**（CMake 与 Python 绑定需要）：`pip install nanobind numpy pybind11`（与 `scripts/build_llvm.sh` 里安装的一致）。
4. **厂商 MLIR（ixcc）**：克隆 **ixcc** 后检出 **`22.x` 分支**，按厂商文档配置并编译 MLIR；`-DMLIR_DIR` 指向其  
   `…/build/lib/cmake/mlir`（或等价安装前缀下的 `lib/cmake/mlir`）。
5. **CoreX / CUDA 兼容 Toolkit**：用于解析 `CUDAToolkit` 并链接 driver API，以生成 `libfly_iluvatar_jit_runtime.so`。

## 不推荐的做法

- **不要**指望仅运行仓库自带 **`scripts/build_llvm.sh`** 就能满足 Iluvatar：`build_llvm.sh` 按 `thirdparty/llvm-hash.txt` 拉 **上游** LLVM，**不包含** IXDL 相关 target。
- **不要**在无 ROCm 的机器上直接跑默认 **`scripts/build.sh`** 且指望「自动变成 Iluvatar」：该脚本当前按默认缓存配置 **ROCDL 后端**；若 `MLIR_DIR` 仍指向上游 `mlir_install`，也无法链接 `MLIRIXDLDialect`。Iluvatar 机器应 **显式 CMake 配置**（见下）。

## 推荐：显式 CMake 配置 + 编译

在 **FlyDSL 仓库根目录**，将路径换成你本机的 ixcc / CoreX 路径：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install nanobind numpy pybind11

# ixcc 请使用 22.x 分支构建后再填路径
IXCC_MLIR_CMAKE=/path/to/ixcc/build/lib/cmake/mlir   # 含 IXDL 的 MLIR CMake 包目录
COREX_ROOT="${HOME}/sw_home/local/corex"             # 仅使用这一套 CoreX，勿混 /usr/local/corex-* 等

rm -rf build-fly
cmake -S . -B build-fly \
  -G Ninja \
  -DFLYDSL_BACKENDS=iluvatar \
  -DMLIR_DIR="${IXCC_MLIR_CMAKE}" \
  -DCUDAToolkit_ROOT="${COREX_ROOT}" \
  -DPython3_EXECUTABLE="$(command -v python3)"

cmake --build build-fly -j"$(nproc)"
```

说明：

- **`-DFLYDSL_BACKENDS=iluvatar`**：只注册 Iluvatar 描述符（`cmake/backends/iluvatar.cmake`），不编译 ROCm runtime，不依赖 HIP。
- **`scripts/build.sh` 里的 `-DHIP_PLATFORM=amd`**：仅在走默认 `rocdl` 配置且用到 HIP 时才有实质意义；Iluvatar-only 配置下即使传入一般也无碍，但 **Iluvatar 构建不依赖 ROCm**。

可选：**只编工具**（不编完整 Python 包时用于快速验证 MLIR 侧）：

```bash
cmake --build build-fly --target fly-opt -j"$(nproc)"
```

## 安装或使用 Python 包

与 `README` 相同，在仓库根目录：

```bash
pip install -e .
```

或临时使用构建树（指向 `build-fly/python_packages`），并确保 `LD_LIBRARY_PATH` 包含 `flydsl/_mlir/_mlir_libs`（见 `README`「Without installing」一节）。在 Iluvatar 机器上若加载 JIT runtime，通常还需把 **CoreX 的 `lib64`** 加入 `LD_LIBRARY_PATH`，以便解析 `libcuda.so.1` 等依赖（详见 `.claude/skills/iluvatar-backend-bringup/SKILL.md` 中 Phase 5 smoke 测试命令）。

## 编译后如何自测（不要求 ROCm）

- **默认即可跑的 L0 单测**（不绑 GPU / 不绑 vendor）：例如  
  `python3 -m pytest tests/unit/test_device_runtime.py tests/unit/test_iluvatar_jit_runtime_resolution.py -v --confcutdir=tests/unit`
- **L1b（需要刚编出的 `fly-opt`，opt-in）**：设置  
  `FLYDSL_ILUVATAR_FLY_OPT="$PWD/build-fly/bin/fly-opt"`  
  后运行 `tests/unit/test_iluvatar_binary_pipeline_smoke.py`（见技能文档 Phase 5.3c-2）。
- **端到端 JIT / 真机**：需 Iluvatar 运行时、设备与 CoreX 库路径；下面 **「最小 kernel 单文件」** 给出可直接复制运行的示例，无需跑仓库里的 pytest。

## 最小 kernel 单文件（复制即跑）

下面给出 **两个** 自包含示例，均假定：你已完成 **Iluvatar-only** 构建；脚本放在 **FlyDSL 仓库根目录**（与 `build-fly/`、`python/` 同级）。

通用步骤：

1. 在仓库根目录新建空文件（示例 A 用 **`minimal_iluvatar_kernel.py`**，示例 B 用 **`minimal_iluvatar_kernel_store.py`**）。  
2. **原样复制**对应代码框全文，粘贴保存。  
3. 在终端 `cd` 到仓库根，使用文末 **推荐命令**（**`LD_LIBRARY_PATH` 以 CoreX `lib64` 为最前**）+ `python3 <文件名>` 运行（示例 B 额外需要 **PyTorch** 且在该栈上 **`torch.cuda.is_available()`** 为真，见「执行」说明）。

脚本会在导入 FlyDSL 之前自动把 `build-fly/python_packages` 和仓库根加入 `sys.path`，一般**不必**再手动 `export PYTHONPATH`（若你坚持用 `pip install -e .`，也可以）。

### 示例 A：空 kernel（仅 launch，无访存）

```python
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
```

### 示例 B：设备内存标量 store + 主机读回（与仓库单测 `test_iluvatar_jit_stores_single_element` 同款）

比示例 A 多一步：**向全局缓冲写 `int32`、再在 CPU 上校验**。需要 **`torch`**（设备侧分配 `int32` 向量；在 CUDA 兼容栈上通常仍用 `torch.cuda` API）。

```python
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
```

**执行**（改 `COREX_ROOT` 与 `python3` 路径）：**推荐始终让 CoreX 的 `lib64` 在 `LD_LIBRARY_PATH` 最前**，再跟 FlyDSL 的 `_mlir_libs`，以便 `libfly_iluvatar_jit_runtime.so` 解析到带 **`ixdrvInit`** 等符号的 **CoreX** `libcuda.so.1`（若把系统 NVIDIA `libcuda` 放在更前，会出现 `undefined symbol: ixdrvInit, version CUDA`）。

```bash
cd /path/to/FlyDSL
# 只保留一套 CoreX：~/sw_home/local/corex（不要同时把 /usr/local/corex-* 放进 LD_LIBRARY_PATH）
COREX_ROOT="${HOME}/sw_home/local/corex"
export LD_LIBRARY_PATH="${COREX_ROOT}/lib64:$PWD/build-fly/python_packages/flydsl/_mlir/_mlir_libs"
unset LD_PRELOAD   # 若曾试验 ixpti shim，先清掉
python3 minimal_iluvatar_kernel.py
# 或：python3 minimal_iluvatar_kernel_store.py
```

**`libcuda` 与 PyTorch（示例 B）**：请使用 **天数提供的 CoreX 版 PyTorch wheel**，并与 **同一 `COREX_ROOT`** 配套；勿再混入 **pip 官方 NVIDIA CUDA 版 PyTorch** 或 **`/usr/lib/x86_64-linux-gnu` 在前的 `LD_LIBRARY_PATH`**。若 `import torch` 仍报 **`ixdnn*` / `ixpti*`**，说明该 `COREX_ROOT` 下 **DNN/PTI 库版本与 wheel 不一致**，需在 CoreX 安装包中补齐与 **`20260513` wheel** 同批的 `lib64`（自检：`nm -D "${COREX_ROOT}/lib64/libcudnn.so.7" | grep ixdnn` 应有输出）。

示例 A 成功时终端会出现 `OK: minimal Iluvatar FlyDSL kernel launched`；示例 B 成功时会出现 `OK: store kernel wrote int32(7)`。

**说明**：若缺少设备或驱动，会在运行时报错；此时可先只做上文 **「编译后如何自测」** 里的 L0 / L1b，不依赖本脚本。

### 已知问题：`ld.lld: error: unsupported e_machine value: 248`

Iluvatar 设备对象使用 **BI 架构** 的 ELF（`e_machine = 248`）。若 `gpu-module-to-binary` 调用了 **系统/上游 LLVM 自带的 `ld.lld`**（例如 `~/llvm/bin/ld.lld`），链接会失败。

**处理**：让 **CoreX 工具链** 的 `clang++` / `llc` / `lld` 优先于系统 PATH：

```bash
export COREX_ROOT="${HOME}/sw_home/local/corex"
export PATH="${COREX_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${COREX_ROOT}/lib64:.../build-fly/python_packages/flydsl/_mlir/_mlir_libs"
```

自检：`which ld.lld` 应指向 **`${COREX_ROOT}/bin/ld.lld`**（或 `lld`）。也可在运行前执行 `hash -r`。

MLIR 的 `gpu-module-to-binary` 还支持 **`toolkit=<CoreX 根目录>`** 选项；若仅改 PATH 仍失败，可尝试在管线里为 Iluvatar 显式传入 `toolkit=${COREX_ROOT}`（与 ixcc 文档一致）。

## 同时编译 `rocdl` 与 `iluvatar`（可选）

若开发机 **同时** 装有 ROCm 与 CoreX，可尝试：

```bash
-DFLYDSL_BACKENDS="rocdl;iluvatar"
```

此时会同时进入 `lib/Runtime/ROCm` 与 `lib/Runtime/Iluvatar`，**仍需要** ROCm 的 HIP 与 Iluvatar 的 `CUDAToolkit` 两边依赖都满足。**纯 Iluvatar 平台无需此选项。**
