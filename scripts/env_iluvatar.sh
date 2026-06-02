# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Single canonical environment for running FlyDSL on Iluvatar ivcore11 (MR).
# Source it (do NOT execute) so the exports land in your current shell:
#
#   source scripts/env_iluvatar.sh
#   python kernels/iluvatar_sme_gemm.py
#   FLYDSL_RUN_DEVICE=1 pytest tests/kernels/test_iluvatar_sme_gemm.py -v
#
# Overridable inputs (export before sourcing to change):
#   SW_HOME         CoreX 4.5.0 userspace prefix     (default: $HOME/sw_home)
#   FLY_BUILD_DIR   FlyDSL build dir w/ python_packages (default: <repo>/build-fly)
#   ARCH / FLYDSL_COMPILE_BACKEND / FLYDSL_RUNTIME_KIND  (defaults below; only
#                   set when currently unset, so explicit overrides win)

# --- locate repo root (works whether sourced from bash or zsh) -------------
if [ -n "${BASH_SOURCE:-}" ]; then
  _ENV_SELF="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
  _ENV_SELF="${(%):-%x}"
else
  _ENV_SELF="$0"
fi
_ENV_SCRIPT_DIR="$(cd "$(dirname "${_ENV_SELF}")" && pwd)"
FLY_REPO_ROOT="$(cd "${_ENV_SCRIPT_DIR}/.." && pwd)"

# --- CoreX userspace (driver + cudart that match the kernel module) --------
: "${SW_HOME:=${HOME}/sw_home}"
if [ -f "${SW_HOME}/enable" ]; then
  # shellcheck disable=SC1091
  source "${SW_HOME}/enable"
else
  echo "[env_iluvatar] WARNING: ${SW_HOME}/enable not found; CoreX libs (libcuda.so.1)"
  echo "[env_iluvatar]          may be missing. Set SW_HOME to your CoreX 4.5.0 prefix."
fi

# --- python interpreter: prefer the repo venv (has the CoreX torch wheel) ---
# The CoreX torch-2.10.0 wheel is installed into <repo>/.venv (cp312); the
# system python3 does not have torch. Put it first so `python`/`python3` resolve
# to the right interpreter. Override with FLY_VENV to point elsewhere.
: "${FLY_VENV:=${FLY_REPO_ROOT}/.venv}"
if [ -x "${FLY_VENV}/bin/python" ] && [ ":${PATH}:" != *":${FLY_VENV}/bin:"* ]; then
  export PATH="${FLY_VENV}/bin:${PATH}"
fi

# --- FlyDSL python + MLIR shared libs --------------------------------------
: "${FLY_BUILD_DIR:=${FLY_REPO_ROOT}/build-fly}"
case "${FLY_BUILD_DIR}" in
  /*) : ;;
  *) FLY_BUILD_DIR="${FLY_REPO_ROOT}/${FLY_BUILD_DIR}" ;;
esac
export FLY_BUILD_DIR

export PYTHONPATH="${FLY_BUILD_DIR}/python_packages:${FLY_REPO_ROOT}:${PYTHONPATH:-}"

_MLIR_LIBS_DIR="${FLY_BUILD_DIR}/python_packages/flydsl/_mlir/_mlir_libs"
if [ -d "${_MLIR_LIBS_DIR}" ] && [ ":${LD_LIBRARY_PATH:-}:" != *":${_MLIR_LIBS_DIR}:"* ]; then
  export LD_LIBRARY_PATH="${_MLIR_LIBS_DIR}:${LD_LIBRARY_PATH:-}"
fi

# --- backend selection (rocm is the flydsl default -> must pin iluvatar) ---
: "${ARCH:=ivcore11}";                  export ARCH
: "${FLYDSL_COMPILE_BACKEND:=iluvatar}"; export FLYDSL_COMPILE_BACKEND
: "${FLYDSL_RUNTIME_KIND:=iluvatar}";    export FLYDSL_RUNTIME_KIND

echo "[env_iluvatar] SW_HOME=${SW_HOME}"
echo "[env_iluvatar] FLY_BUILD_DIR=${FLY_BUILD_DIR}"
echo "[env_iluvatar] python=$(command -v python3 || command -v python || echo '<none>')"
echo "[env_iluvatar] ARCH=${ARCH}  COMPILE_BACKEND=${FLYDSL_COMPILE_BACKEND}  RUNTIME_KIND=${FLYDSL_RUNTIME_KIND}"

unset _ENV_SELF _ENV_SCRIPT_DIR _MLIR_LIBS_DIR
