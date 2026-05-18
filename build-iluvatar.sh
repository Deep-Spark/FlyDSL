#!/usr/bin/env bash
# Iluvatar-only FlyDSL build (no ROCm). Run from repo root or any cwd.
set -euo pipefail

export IXCC_MLIR_CMAKE="${IXCC_MLIR_CMAKE:-/home/wcyx/sw_home/sdk/ixcc/build/lib/cmake/mlir}"
export COREX_ROOT="${COREX_ROOT:-/home/wcyx/sw_home/local/corex}"

FLYDSL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${FLYDSL_ROOT}"

VENV_DIR="${FLYDSL_ROOT}/.venv"
if [[ ! -d "${VENV_DIR}" ]]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "error: .venv not found and 'uv' is not on PATH; install uv or create .venv manually" >&2
    exit 1
  fi
  echo "Creating virtual environment with uv at ${VENV_DIR}"
  uv venv "${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

if command -v uv >/dev/null 2>&1; then
  uv pip install nanobind numpy pybind11
else
  pip install -U pip
  pip install nanobind numpy pybind11
fi

rm -rf build-fly

cmake -S . -B build-fly \
  -G Ninja \
  -DFLYDSL_BACKENDS=iluvatar \
  -DMLIR_DIR="${IXCC_MLIR_CMAKE}" \
  -DCUDAToolkit_ROOT="${COREX_ROOT}" \
  -DPython3_EXECUTABLE="${VENV_DIR}/bin/python3"

cmake --build build-fly -j"$(nproc)"
