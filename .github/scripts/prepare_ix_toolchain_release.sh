#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Refresh the release IXCC tree used by Iluvatar wheel builds.
# Dev CI (ci-device / perf-daily) continues to use sdk/ixcc @ working.
#
# Default layout:
#   host (glibc of the runner): IXCC_RELEASE_ROOT=$SW_HOME/sdk/ixcc-release
#   Ubuntu 20.04 docker:        IXCC_RELEASE_ROOT=$SW_HOME/sdk/ixcc-release-u2004
#   branch: origin/xiang.zhang/ixcc-flydsl-release
#
# Usage:
#   bash .github/scripts/prepare_ix_toolchain_release.sh
#   bash .github/scripts/prepare_ix_toolchain_release.sh --force-rebuild
#   bash .github/scripts/prepare_ix_toolchain_release.sh --docker-image "$U2004_IMAGE"
#
# CI entry: workflow ix-toolchain-daily-refresh, input abi=u2004.

set -euo pipefail

IXCC_WORKING_ROOT="${IXCC_WORKING_ROOT:-/home/flydsl/sw_home/sdk/ixcc}"
IXCC_RELEASE_REF="${IXCC_RELEASE_REF:-origin/xiang.zhang/ixcc-flydsl-release}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
SW_HOME="${SW_HOME:-$(cd "${IXCC_WORKING_ROOT}/../.." && pwd)}"
HOST_RELEASE_ROOT="${SW_HOME}/sdk/ixcc-release"
U2004_RELEASE_ROOT="${SW_HOME}/sdk/ixcc-release-u2004"
IXCC_RELEASE_ROOT="${IXCC_RELEASE_ROOT:-}"
DOCKER_IMAGE="${DOCKER_IMAGE:-}"
PYTHON_BIN="${PYTHON_BIN:-}"
COREX_ROOT="${COREX_ROOT:-${SW_HOME}/local/corex}"

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
    --docker-image)
      DOCKER_IMAGE="${2:?}"
      shift 2
      ;;
    --python)
      PYTHON_BIN="${2:?}"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${IXCC_RELEASE_ROOT}" ]]; then
  if [[ -n "${DOCKER_IMAGE}" ]]; then
    IXCC_RELEASE_ROOT="${U2004_RELEASE_ROOT}"
  else
    IXCC_RELEASE_ROOT="${HOST_RELEASE_ROOT}"
  fi
fi

MARKER="${IXCC_RELEASE_ROOT}/build/lib/cmake/mlir/MLIRConfig.cmake"
PYTHON_MARKER="${IXCC_RELEASE_ROOT}/build/tools/mlir/python_packages/mlir_core"
STAMP="${IXCC_RELEASE_ROOT}/.flydsl_ixcc_build_commit"

if [[ ! -d "${IXCC_WORKING_ROOT}/.git" ]]; then
  echo "::error::working IXCC git root missing: ${IXCC_WORKING_ROOT}"
  exit 1
fi

git -C "${IXCC_WORKING_ROOT}" fetch origin xiang.zhang/ixcc-flydsl-release
desired="$(git -C "${IXCC_WORKING_ROOT}" rev-parse "${IXCC_RELEASE_REF}")"

if [[ ! -d "${IXCC_RELEASE_ROOT}/.git" && ! -e "${IXCC_RELEASE_ROOT}/.git" ]]; then
  echo "[ixcc-release] creating worktree at ${IXCC_RELEASE_ROOT}"
  mkdir -p "$(dirname "${IXCC_RELEASE_ROOT}")"
  git -C "${IXCC_WORKING_ROOT}" worktree add --detach "${IXCC_RELEASE_ROOT}" "${desired}"
else
  echo "[ixcc-release] checking out ${desired} in ${IXCC_RELEASE_ROOT}"
  git -C "${IXCC_RELEASE_ROOT}" fetch origin xiang.zhang/ixcc-flydsl-release || true
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

emit_outputs() {
  local short="$1"
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    {
      echo "ixcc_commit=${short}"
      echo "ixcc_root=${IXCC_RELEASE_ROOT}"
      echo "mlir_cmake=${MARKER%/*}"
    } >> "${GITHUB_OUTPUT}"
  fi
}

if [[ "${need_build}" != "1" ]]; then
  short="$(git -C "${IXCC_RELEASE_ROOT}" rev-parse --short HEAD)"
  echo "[ixcc-release] up to date: ${short}"
  echo "[ixcc-release] MLIR: ${MARKER}"
  emit_outputs "${short}"
  exit 0
fi

echo "[ixcc-release] configuring + building (this can take a long time)"

run_host_build() {
  export PS1="${PS1:-}"
  set +u
  # shellcheck disable=SC1091
  source "${SW_HOME}/enable"
  set -u
  local python_exe="${PYTHON_BIN:-$(command -v python3)}"
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
    -DPython3_EXECUTABLE="${python_exe}"
  cmake --build "${IXCC_RELEASE_ROOT}/build" -j"$(nproc)"
}

run_docker_build() {
  local build_sh="${IXCC_RELEASE_ROOT}/.flydsl_ixcc_docker_build.sh"
  cat >"${build_sh}" <<'EOS'
set -euo pipefail
export HOME="${HOME:-/tmp}"
export LD_LIBRARY_PATH="${COREX_ROOT:+${COREX_ROOT}/lib64:}${LD_LIBRARY_PATH:-}"
resolve_python() {
  if [[ -n "${PYTHON_BIN}" && -x "${PYTHON_BIN}" ]]; then
    echo "${PYTHON_BIN}"
    return
  fi
  if command -v python3.10 >/dev/null 2>&1; then
    command -v python3.10
    return
  fi
  if [[ -x /usr/local/conda/envs/py3.10/bin/python ]]; then
    echo /usr/local/conda/envs/py3.10/bin/python
    return
  fi
  if [[ -x /opt/conda/envs/py3.10/bin/python ]]; then
    echo /opt/conda/envs/py3.10/bin/python
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  echo "cannot find python in container" >&2
  exit 1
}
python_exe="$(resolve_python)"
echo "[ixcc-release] docker python=${python_exe}"
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
  -DPython3_EXECUTABLE="${python_exe}"
cmake --build "${IXCC_RELEASE_ROOT}/build" -j"$(nproc)"
EOS

  local docker_args=(
    run --rm
    --network host
    --user "$(id -u):$(id -g)"
    --ulimit nofile=65536:65536
    -e HOME=/tmp
    -e IXCC_RELEASE_ROOT="${IXCC_RELEASE_ROOT}"
    -e PYTHON_BIN="${PYTHON_BIN}"
    -v "${IXCC_RELEASE_ROOT}:${IXCC_RELEASE_ROOT}"
  )
  if [[ -n "${COREX_ROOT}" && -d "${COREX_ROOT}" ]]; then
    docker_args+=(-v "${COREX_ROOT}:${COREX_ROOT}:ro" -e COREX_ROOT="${COREX_ROOT}")
  fi
  docker "${docker_args[@]}" "${DOCKER_IMAGE}" bash "${build_sh}"
  rm -f "${build_sh}"
}

if [[ -n "${DOCKER_IMAGE}" ]]; then
  echo "[ixcc-release] docker image=${DOCKER_IMAGE}"
  run_docker_build
else
  run_host_build
fi

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
emit_outputs "${short}"
