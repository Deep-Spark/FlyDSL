#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Refresh the release IXCC tree used by Iluvatar wheel builds.
# Dev CI (ci-device / perf-daily) continues to use sdk/ixcc @ working.
#
# Default layout:
#   IXCC_RELEASE_ROOT=/home/flydsl/sw_home/sdk/ixcc-release
#   branch: origin/xiang.zhang/disable-dump-ir-asm
#
# Usage:
#   bash .github/scripts/prepare_ix_toolchain_release.sh
#   bash .github/scripts/prepare_ix_toolchain_release.sh --force-rebuild

set -euo pipefail

IXCC_WORKING_ROOT="${IXCC_WORKING_ROOT:-/home/flydsl/sw_home/sdk/ixcc}"
IXCC_RELEASE_ROOT="${IXCC_RELEASE_ROOT:-/home/flydsl/sw_home/sdk/ixcc-release}"
IXCC_RELEASE_REF="${IXCC_RELEASE_REF:-origin/xiang.zhang/disable-dump-ir-asm}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
SW_HOME="${SW_HOME:-$(cd "${IXCC_WORKING_ROOT}/../.." && pwd)}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force-rebuild)
      FORCE_REBUILD=1
      shift
      ;;
    --ixcc-release-root)
      IXCC_RELEASE_ROOT="${2:?}"
      shift 2
      ;;
    --ref)
      IXCC_RELEASE_REF="${2:?}"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

MARKER="${IXCC_RELEASE_ROOT}/build/lib/cmake/mlir/MLIRConfig.cmake"
PYTHON_MARKER="${IXCC_RELEASE_ROOT}/build/tools/mlir/python_packages/mlir_core"
STAMP="${IXCC_RELEASE_ROOT}/.flydsl_ixcc_build_commit"

if [[ ! -d "${IXCC_WORKING_ROOT}/.git" ]]; then
  echo "::error::working IXCC git root missing: ${IXCC_WORKING_ROOT}"
  exit 1
fi

git -C "${IXCC_WORKING_ROOT}" fetch origin xiang.zhang/disable-dump-ir-asm
desired="$(git -C "${IXCC_WORKING_ROOT}" rev-parse "${IXCC_RELEASE_REF}")"

if [[ ! -d "${IXCC_RELEASE_ROOT}/.git" && ! -e "${IXCC_RELEASE_ROOT}/.git" ]]; then
  echo "[ixcc-release] creating worktree at ${IXCC_RELEASE_ROOT}"
  git -C "${IXCC_WORKING_ROOT}" worktree add --detach "${IXCC_RELEASE_ROOT}" "${desired}"
else
  echo "[ixcc-release] checking out ${desired} in ${IXCC_RELEASE_ROOT}"
  git -C "${IXCC_RELEASE_ROOT}" fetch origin xiang.zhang/disable-dump-ir-asm || true
  git -C "${IXCC_RELEASE_ROOT}" checkout --detach "${desired}"
fi

current="$(git -C "${IXCC_RELEASE_ROOT}" rev-parse HEAD)"
need_build=0
if [[ "${FORCE_REBUILD}" == "1" ]]; then
  need_build=1
elif [[ ! -f "${MARKER}" ]]; then
  need_build=1
elif [[ ! -d "${PYTHON_MARKER}" ]]; then
  echo "[ixcc-release] MLIR Python bindings package missing; rebuilding"
  need_build=1
elif [[ -f "${STAMP}" ]]; then
  stamped="$(head -n1 "${STAMP}" | tr -d '[:space:]')"
  short="$(git -C "${IXCC_RELEASE_ROOT}" rev-parse --short HEAD)"
  if [[ "${stamped}" != "${short}" && "${stamped}" != "${current}" ]]; then
    need_build=1
  fi
else
  need_build=1
fi

if [[ "${need_build}" != "1" ]]; then
  echo "[ixcc-release] up to date: $(git -C "${IXCC_RELEASE_ROOT}" rev-parse --short HEAD)"
  echo "[ixcc-release] MLIR: ${MARKER}"
  exit 0
fi

echo "[ixcc-release] configuring + building (this can take a long time)"
export PS1="${PS1:-}"
set +u
# shellcheck disable=SC1091
source "${SW_HOME}/enable"
set -u

cmake -S "${IXCC_RELEASE_ROOT}/llvm" \
  -B "${IXCC_RELEASE_ROOT}/build" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS='clang;lld;mlir' \
  -DLLVM_TARGETS_TO_BUILD='Iluvatar;X86' \
  -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_CXX_STANDARD=17 \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DCLANG_INCLUDE_TESTS=OFF \
  -DMLIR_INCLUDE_TESTS=OFF \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE="$(command -v python3)"

cmake --build "${IXCC_RELEASE_ROOT}/build" -j"$(nproc)"

short="$(git -C "${IXCC_RELEASE_ROOT}" rev-parse --short HEAD)"
{
  echo "${short}"
  date -u +"finished_utc=%Y-%m-%dT%H:%M:%SZ"
} >"${STAMP}"

if [[ ! -f "${MARKER}" ]]; then
  echo "::error::build finished but marker missing: ${MARKER}"
  exit 1
fi
if [[ ! -d "${PYTHON_MARKER}" ]]; then
  echo "::error::build finished but MLIR Python package missing: ${PYTHON_MARKER}"
  echo "::error::Ensure -DMLIR_ENABLE_BINDINGS_PYTHON=ON and Python3 Development.Module are available."
  exit 1
fi

echo "[ixcc-release] ready commit=${short} mlir=${MARKER}"
