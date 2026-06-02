#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# From ixcc llvm-project layout (sibling llvm/, mlir/, clang/ under IXCC_ROOT),
# do a clean configure+build+install of MLIR/LLVM matching the source tree
# version in ixcc/cmake/Modules/LLVMVersion.cmake (e.g. 22.x).
#
# Usage:
#   export IXCC_ROOT=/path/to/ixcc   # default: $HOME/sw_home/sdk/ixcc
#   bash scripts/build_ixcc_llvm.sh -j32
#   export MLIR_PATH=$IXCC_ROOT/mlir_install
#   bash scripts/build.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IXCC_ROOT="${IXCC_ROOT:-${HOME}/sw_home/sdk/ixcc}"
LLVM_SRC="${IXCC_ROOT}/llvm"
BUILD_DIR="${IXCC_BUILD_DIR:-${IXCC_ROOT}/build}"
INSTALL_PREFIX="${IXCC_MLIR_INSTALL:-${IXCC_ROOT}/mlir_install}"

# ROCm runner pulls in HIP (hipConfig.cmake). Enable only if you have ROCm dev installed:
#   export IXCC_MLIR_ENABLE_ROCM_RUNNER=1
MLIR_ENABLE_ROCM_RUNNER="${IXCC_MLIR_ENABLE_ROCM_RUNNER:-OFF}"

# ixcc LLVM trees have occasionally shipped AMDGPU .td conflicts (tblgen duplicate class).
# Default skips AMDGPU so the build completes; for full ROCm GPU codegen add AMDGPU after fixing ixcc:
#   export IXCC_LLVM_TARGETS="X86;NVPTX;AMDGPU"
IXCC_LLVM_TARGETS="${IXCC_LLVM_TARGETS:-X86;NVPTX}"

PARALLEL_JOBS=$((($(nproc) + 1) / 2))
for arg in "$@"; do
  if [[ "$arg" =~ ^-j([0-9]+)$ ]]; then
    PARALLEL_JOBS="${BASH_REMATCH[1]}"
  fi
done

if [[ ! -f "${LLVM_SRC}/CMakeLists.txt" ]]; then
  echo "Error: missing ${LLVM_SRC}/CMakeLists.txt (set IXCC_ROOT?)" >&2
  exit 1
fi

# Optional: use FlyDSL venv for nanobind
if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/.venv/bin/activate"
fi

if ! python3 -c "import nanobind" 2>/dev/null; then
  echo "Installing nanobind into current Python env..."
  pip install 'nanobind>=2.0' numpy pybind11
fi

NANOBIND_DIR=$(python3 -c "import nanobind, os; print(os.path.dirname(nanobind.__file__) + '/cmake')")

GENERATOR="Unix Makefiles"
if command -v ninja &>/dev/null; then
  GENERATOR="Ninja"
fi

echo "=============================================="
echo "ixcc LLVM/MLIR rebuild from source (22.x tree)"
echo "  IXCC_ROOT:      ${IXCC_ROOT}"
echo "  LLVM_SRC:       ${LLVM_SRC}"
echo "  BUILD_DIR:      ${BUILD_DIR}"
echo "  INSTALL_PREFIX: ${INSTALL_PREFIX}"
echo "  MLIR_ENABLE_ROCM_RUNNER: ${MLIR_ENABLE_ROCM_RUNNER}"
echo "  LLVM_TARGETS_TO_BUILD: ${IXCC_LLVM_TARGETS}"
echo "  PARALLEL_JOBS:  ${PARALLEL_JOBS}"
echo "=============================================="

if [[ -d "${BUILD_DIR}" ]]; then
  echo "Discarding existing build tree: ${BUILD_DIR}"
  rm -rf "${BUILD_DIR}"
fi
mkdir -p "${BUILD_DIR}"

# compiler-rt runtimes are omitted to shorten iteration time; add
# -DLLVM_ENABLE_RUNTIMES=compiler-rt if you need them.

cmake -G "${GENERATOR}" \
  -S "${LLVM_SRC}" \
  -B "${BUILD_DIR}" \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_TARGETS_TO_BUILD="${IXCC_LLVM_TARGETS}" \
  -DLLVM_TOOL_DYNAMIC_COMPILE_BUILD=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=17 \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_INSTALL_UTILS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DMLIR_ENABLE_ROCM_RUNNER="${MLIR_ENABLE_ROCM_RUNNER}" \
  -DMLIR_BINDINGS_PYTHON_NB_DOMAIN=mlir \
  -DPython3_EXECUTABLE="$(command -v python3)" \
  -Dnanobind_DIR="${NANOBIND_DIR}" \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLVM_BUILD_LLVM_DYLIB=OFF \
  -DLLVM_LINK_LLVM_DYLIB=OFF \
  -DCMAKE_INSTALL_RPATH="\$ORIGIN"

echo "Building..."
cmake --build "${BUILD_DIR}" -j"${PARALLEL_JOBS}"

echo "Installing to ${INSTALL_PREFIX} ..."
rm -rf "${INSTALL_PREFIX}"
mkdir -p "${INSTALL_PREFIX}"
cmake --install "${BUILD_DIR}" --prefix "${INSTALL_PREFIX}"

if [[ ! -d "${INSTALL_PREFIX}/lib/cmake/mlir" ]]; then
  echo "Error: install prefix missing lib/cmake/mlir: ${INSTALL_PREFIX}" >&2
  exit 1
fi

echo ""
echo "Done. Configure FlyDSL with:"
echo "  export MLIR_PATH=${INSTALL_PREFIX}"
echo "  bash ${REPO_ROOT}/scripts/build.sh"
