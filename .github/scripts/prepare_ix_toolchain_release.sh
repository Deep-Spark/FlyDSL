#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Refresh the release IXCC tree used by Iluvatar wheel builds.
# Dev CI (ci-device / perf-daily) continues to use sdk/ixcc @ working.
#
# Default layout:
#   host (glibc of the runner): IXCC_RELEASE_ROOT=$SW_HOME/sdk/ixcc-release
#   Ubuntu 20.04 docker:        IXCC_RELEASE_ROOT=$SW_HOME/sdk/ixcc-release-u2004-cp<py>
#   branch: origin/xiang.zhang/ixcc-flydsl-release
#
# Usage:
#   bash .github/scripts/prepare_ix_toolchain_release.sh
#   bash .github/scripts/prepare_ix_toolchain_release.sh --force-rebuild
#   bash .github/scripts/prepare_ix_toolchain_release.sh --docker-image "$U2004_IMAGE"
#   bash .github/scripts/prepare_ix_toolchain_release.sh --docker-image "$U2004_IMAGE" --python-version 3.12
#
# CI entry: workflow ix-toolchain-daily-refresh, input abi=u2004.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IXCC_WORKING_ROOT="${IXCC_WORKING_ROOT:-/home/flydsl/sw_home/sdk/ixcc}"
IXCC_RELEASE_REF="${IXCC_RELEASE_REF:-origin/xiang.zhang/ixcc-flydsl-release}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
SW_HOME="${SW_HOME:-$(cd "${IXCC_WORKING_ROOT}/../.." && pwd)}"
HOST_RELEASE_ROOT="${SW_HOME}/sdk/ixcc-release"
U2004_RELEASE_ROOT_BASE="${SW_HOME}/sdk/ixcc-release-u2004"
IXCC_RELEASE_ROOT="${IXCC_RELEASE_ROOT:-}"
DOCKER_IMAGE="${DOCKER_IMAGE:-}"
PYTHON_BIN="${PYTHON_BIN:-}"
PYTHON_VERSION="${PYTHON_VERSION:-}"
COREX_ROOT="${COREX_ROOT:-${SW_HOME}/local/corex}"
U2004_CONDA_BASE="${U2004_CONDA_BASE:-${SW_HOME}/sdk/.conda-u2004}"

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
    --python-version)
      PYTHON_VERSION="${2:?}"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -n "${DOCKER_IMAGE}" && -z "${PYTHON_VERSION}" ]]; then
  PYTHON_VERSION="3.12"
fi

if [[ -z "${IXCC_RELEASE_ROOT}" ]]; then
  if [[ -n "${DOCKER_IMAGE}" ]]; then
    IXCC_RELEASE_ROOT="${U2004_RELEASE_ROOT_BASE}-cp${PYTHON_VERSION//./}"
  else
    IXCC_RELEASE_ROOT="${HOST_RELEASE_ROOT}"
  fi
fi

MARKER="${IXCC_RELEASE_ROOT}/build/lib/cmake/mlir/MLIRConfig.cmake"
PYTHON_MARKER="${IXCC_RELEASE_ROOT}/build/tools/mlir/python_packages/mlir_core"
STAMP="${IXCC_RELEASE_ROOT}/.flydsl_ixcc_build_commit"
PYTHON_STAMP="${IXCC_RELEASE_ROOT}/.flydsl_ixcc_python_version"

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
if [[ -n "${DOCKER_IMAGE}" && -n "${PYTHON_VERSION}" && -f "${PYTHON_STAMP}" ]]; then
  stamped_py="$(tr -d '[:space:]' < "${PYTHON_STAMP}")"
  if [[ "${stamped_py}" != "${PYTHON_VERSION}" ]]; then
    echo "[ixcc-release] python version changed ${stamped_py} -> ${PYTHON_VERSION}; rebuilding"
    need_build=1
  fi
fi

emit_outputs() {
  local short="$1"
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    {
      echo "ixcc_commit=${short}"
      echo "ixcc_root=${IXCC_RELEASE_ROOT}"
      echo "mlir_cmake=${MARKER%/*}"
      echo "python_version=${PYTHON_VERSION}"
    } >> "${GITHUB_OUTPUT}"
  fi
}

if [[ "${need_build}" != "1" ]]; then
  short="$(git -C "${IXCC_RELEASE_ROOT}" rev-parse --short HEAD)"
  echo "[ixcc-release] up to date: ${short} python=${PYTHON_VERSION:-host}"
  echo "[ixcc-release] MLIR: ${MARKER}"
  emit_outputs "${short}"
  exit 0
fi

echo "[ixcc-release] configuring + building (this can take a long time)"

if [[ "${need_build}" == "1" ]]; then
  if [[ "${FORCE_REBUILD}" == "1" ]]; then
    echo "[ixcc-release] clearing ${IXCC_RELEASE_ROOT}/build (--force-rebuild)"
    rm -rf "${IXCC_RELEASE_ROOT}/build"
  elif [[ -n "${DOCKER_IMAGE}" && -n "${PYTHON_VERSION}" && -f "${PYTHON_STAMP}" ]]; then
    stamped_py="$(tr -d '[:space:]' < "${PYTHON_STAMP}")"
    if [[ "${stamped_py}" != "${PYTHON_VERSION}" ]]; then
      echo "[ixcc-release] clearing ${IXCC_RELEASE_ROOT}/build (python ${stamped_py} -> ${PYTHON_VERSION})"
      rm -rf "${IXCC_RELEASE_ROOT}/build"
    fi
  elif [[ ! -f "${MARKER}" && -d "${IXCC_RELEASE_ROOT}/build" ]]; then
    echo "[ixcc-release] clearing stale partial build under ${IXCC_RELEASE_ROOT}/build"
    rm -rf "${IXCC_RELEASE_ROOT}/build"
  fi
fi

# Foreign-owned build tree preflight: a prior root-owned build (e.g. from a
# `docker run` without --user) leaves 755 dirs the non-root CI user cannot
# write into. ninja then hits "Permission denied" on .o.d files halfway
# through, and if run_docker_build's fallback tolerates that failure we end
# up shipping a tree missing static libs like libMLIRMlirOptMain.a -- fatal
# only later when FlyDSL tries to link fly-opt. Detect and abort loudly here.
if [[ -d "${IXCC_RELEASE_ROOT}/build" ]]; then
  # test writability on any file under build/ that is not owned by us; -writable
  # requires the effective UID to be able to open O_WRONLY, i.e. reflects both
  # ownership and mode.
  foreign="$(find "${IXCC_RELEASE_ROOT}/build" \
              \( -type f -o -type d \) ! -user "$(id -u)" ! -writable \
              -print -quit 2>/dev/null || true)"
  if [[ -n "${foreign}" ]]; then
    echo "::error::${IXCC_RELEASE_ROOT}/build contains files not writable by uid=$(id -u) (first: ${foreign})."
    echo "::error::This usually means a previous build ran as root inside docker without --user."
    echo "::error::Fix: sudo rm -rf ${IXCC_RELEASE_ROOT}/build   (then re-run this script)."
    exit 1
  fi
fi

# CMake flag drift: if the existing tree was configured without a flag we now
# require, the stamp check above would still declare it up-to-date. Force a
# clean rebuild instead. Skipping this leads to MLIR python bindings built
# against a different nanobind domain than FlyDSL, and nanobind.stubgen
# segfaults at import time during the FlyDSL wheel build.
if [[ -f "${IXCC_RELEASE_ROOT}/build/CMakeCache.txt" ]]; then
  cache="${IXCC_RELEASE_ROOT}/build/CMakeCache.txt"
  required_cache_flags=(
    "MLIR_BINDINGS_PYTHON_NB_DOMAIN:.*=mlir$"
  )
  need_flag_rebuild=0
  for pat in "${required_cache_flags[@]}"; do
    if ! grep -qE "^${pat}" "${cache}"; then
      echo "[ixcc-release] CMakeCache missing required flag (pattern: ${pat}); forcing rebuild"
      need_flag_rebuild=1
    fi
  done
  if (( need_flag_rebuild )); then
    rm -rf "${IXCC_RELEASE_ROOT}/build"
  fi
fi

run_host_build() {
  export PS1="${PS1:-}"
  set +u
  # shellcheck disable=SC1091
  source "${SW_HOME}/enable"
  set -u
  local python_exe="${PYTHON_BIN:-$(command -v python3)}"
  # Pin nanobind location + domain so MLIR python bindings use the exact same
  # nanobind install (and nb_domain) that FlyDSL later links against. Different
  # nanobind paths or an unset domain cause a process-level type registry
  # clash at first import (typical symptom: nanobind.stubgen segfault).
  local nanobind_dir
  nanobind_dir="$("${python_exe}" -c 'import nanobind, os; print(os.path.dirname(nanobind.__file__) + "/cmake")')"
  cmake -S "${IXCC_RELEASE_ROOT}/llvm" \
    -B "${IXCC_RELEASE_ROOT}/build" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS='clang;lld;mlir' \
    -DLLVM_TARGETS_TO_BUILD='Iluvatar;X86' \
    -DLLVM_ENABLE_ASSERTIONS=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLVM_BUILD_LLVM_DYLIB=OFF \
    -DLLVM_LINK_LLVM_DYLIB=OFF \
    -DCMAKE_CXX_STANDARD=17 \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DCLANG_INCLUDE_TESTS=OFF \
    -DMLIR_INCLUDE_TESTS=OFF \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DMLIR_BINDINGS_PYTHON_NB_DOMAIN=mlir \
    -Dnanobind_DIR="${nanobind_dir}" \
    -DPython3_EXECUTABLE="${python_exe}"
  cmake --build "${IXCC_RELEASE_ROOT}/build" -j"$(nproc)"
}

run_docker_build() {
  local build_sh="${IXCC_RELEASE_ROOT}/.flydsl_ixcc_docker_build.sh"
  cat >"${build_sh}" <<EOS
set -euo pipefail
export HOME="\${HOME:-/tmp}"
export LD_LIBRARY_PATH="\${COREX_ROOT:+\\\${COREX_ROOT}/lib64:}\${LD_LIBRARY_PATH:-}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/u2004_conda_python.sh"
python_exe="\$(u2004_resolve_python "${PYTHON_VERSION}")"
u2004_isolate_python_runtime "\${python_exe}"
mapfile -t cmake_compiler_args < <(u2004_cmake_compiler_args "${PYTHON_VERSION}")
mapfile -t cmake_python_args < <(u2004_cmake_python_args "\${python_exe}")
py_root="\$(u2004_python_root "\${python_exe}")"
echo "[ixcc-release] docker python=\${python_exe}"
echo "[ixcc-release] docker python_root=\${py_root}"
# Pin nanobind location + domain so MLIR python bindings use the exact same
# nanobind install (and nb_domain) that FlyDSL later links against. Different
# nanobind paths or an unset domain cause a process-level type registry clash
# at first import (typical symptom: nanobind.stubgen segfault at the very end
# of the FlyDSL wheel build).
nanobind_dir="\$("\${python_exe}" -c 'import nanobind, os; print(os.path.dirname(nanobind.__file__) + "/cmake")')"
echo "[ixcc-release] docker nanobind_dir=\${nanobind_dir}"
cmake -S "\${IXCC_RELEASE_ROOT}/llvm" \\
  -B "\${IXCC_RELEASE_ROOT}/build" \\
  -G Ninja \\
  -DCMAKE_BUILD_TYPE=Release \\
  -DLLVM_ENABLE_PROJECTS='clang;lld;mlir' \\
  -DLLVM_TARGETS_TO_BUILD='Iluvatar;X86' \\
  -DLLVM_ENABLE_ASSERTIONS=OFF \\
  -DBUILD_SHARED_LIBS=OFF \\
  -DLLVM_BUILD_LLVM_DYLIB=OFF \\
  -DLLVM_LINK_LLVM_DYLIB=OFF \\
  -DCMAKE_CXX_STANDARD=17 \\
  -DLLVM_INCLUDE_TESTS=OFF \\
  -DCLANG_INCLUDE_TESTS=OFF \\
  -DMLIR_INCLUDE_TESTS=OFF \\
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \\
  -DMLIR_BINDINGS_PYTHON_NB_DOMAIN=mlir \\
  -Dnanobind_DIR="\${nanobind_dir}" \\
  "\${cmake_python_args[@]}" \\
  "\${cmake_compiler_args[@]}"
if ! cmake --build "\${IXCC_RELEASE_ROOT}/build" -j"\$(nproc)"; then
  echo "[ixcc-release] build returned an error; checking MLIR python libs and FlyDSL link deps" >&2
  if ! u2004_validate_ixcc_python_bindings "${PYTHON_VERSION}" "\${IXCC_RELEASE_ROOT}"; then
    exit 1
  fi
  # Even if python bindings validate, we require the static libs FlyDSL links
  # against (libMLIRMlirOptMain.a, libMLIROptLib.a, ...). Otherwise the wheel
  # build later blows up with "needed by bin/fly-opt, missing and no known
  # rule to make it" and blames FlyDSL for a broken IXCC build. Refuse to
  # continue when any required lib is missing.
  missing=()
  for lib in libMLIRMlirOptMain.a libMLIROptLib.a; do
    if [[ ! -f "\${IXCC_RELEASE_ROOT}/build/lib/\${lib}" ]]; then
      missing+=("\${lib}")
    fi
  done
  if (( \${#missing[@]} > 0 )); then
    echo "::error::ixcc build finished with a compile failure and left required libs missing: \${missing[*]}" >&2
    echo "::error::Rerun with FORCE_REBUILD=1 (which rm -rf's build/) or investigate the compile error." >&2
    exit 1
  fi
  echo "[ixcc-release] continuing after non-fatal ixcc tail failure (likely stubgen); FlyDSL link deps present" >&2
else
  u2004_validate_ixcc_python_bindings "${PYTHON_VERSION}" "\${IXCC_RELEASE_ROOT}"
fi
EOS

  local docker_args=(
    run --rm
    --network host
    --user "$(id -u):$(id -g)"
    --ulimit nofile=65536:65536
    -e HOME=/tmp/flydsl-u2004-home
    -e IXCC_RELEASE_ROOT="${IXCC_RELEASE_ROOT}"
    -e PYTHON_BIN="${PYTHON_BIN}"
    -e U2004_CONDA_BASE="${U2004_CONDA_BASE}"
    -v "${IXCC_RELEASE_ROOT}:${IXCC_RELEASE_ROOT}"
    -v "${SCRIPT_DIR}:${SCRIPT_DIR}:ro"
    -v "${U2004_CONDA_BASE}:${U2004_CONDA_BASE}"
  )
  if [[ -n "${COREX_ROOT}" && -d "${COREX_ROOT}" ]]; then
    docker_args+=(-v "${COREX_ROOT}:${COREX_ROOT}:ro" -e COREX_ROOT="${COREX_ROOT}")
  fi
  docker "${docker_args[@]}" "${DOCKER_IMAGE}" bash "${build_sh}"
  rm -f "${build_sh}"
}

if [[ -n "${DOCKER_IMAGE}" ]]; then
  echo "[ixcc-release] docker image=${DOCKER_IMAGE} python=${PYTHON_VERSION}"
  run_docker_build
else
  run_host_build
fi

short="$(git -C "${IXCC_RELEASE_ROOT}" rev-parse --short HEAD)"
{
  echo "${short}"
  date -u +"finished_utc=%Y-%m-%dT%H:%M:%SZ"
} >"${STAMP}"
if [[ -n "${PYTHON_VERSION}" ]]; then
  echo "${PYTHON_VERSION}" >"${PYTHON_STAMP}"
fi

if [[ ! -f "${MARKER}" ]]; then
  echo "::error::build finished but marker missing: ${MARKER}"
  exit 1
fi
if [[ ! -d "${PYTHON_MARKER}" ]]; then
  echo "::error::build finished but MLIR Python package missing: ${PYTHON_MARKER}"
  echo "::error::Ensure -DMLIR_ENABLE_BINDINGS_PYTHON=ON and Python3 Development.Module are available."
  exit 1
fi

echo "[ixcc-release] ready commit=${short} python=${PYTHON_VERSION:-host} mlir=${MARKER}"
emit_outputs "${short}"
