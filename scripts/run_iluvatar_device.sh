#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# One-shot driver for the ivcore11 (FlyIXDL) SME GEMM on-device bring-up.
# Pins the CoreX userspace + iluvatar backend (via scripts/env_iluvatar.sh),
# then runs either the standalone kernel or the L2 device pytest.
#
# Usage:
#   bash scripts/run_iluvatar_device.sh kernel   # run kernels/iluvatar_sme_gemm.py
#   bash scripts/run_iluvatar_device.sh test     # run L2 device pytest (numeric)
#   bash scripts/run_iluvatar_device.sh          # defaults to: kernel
#
# Honors the same overridable inputs as scripts/env_iluvatar.sh
# (SW_HOME, FLY_BUILD_DIR, ARCH, FLYDSL_COMPILE_BACKEND, FLYDSL_RUNTIME_KIND).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env_iluvatar.sh"

PY="${PYTHON:-$(command -v python || command -v python3)}"
if [ -z "${PY}" ]; then
  echo "[run_iluvatar_device] no python/python3 on PATH" >&2
  exit 1
fi

MODE="${1:-kernel}"
case "${MODE}" in
  kernel)
    echo "[run_iluvatar_device] running kernels/iluvatar_sme_gemm.py via ${PY}"
    exec "${PY}" kernels/iluvatar_sme_gemm.py
    ;;
  test)
    echo "[run_iluvatar_device] running L2 device pytest (FLYDSL_RUN_DEVICE=1) via ${PY}"
    export FLYDSL_RUN_DEVICE=1
    exec "${PY}" -m pytest tests/kernels/test_iluvatar_sme_gemm.py -v --no-header --tb=short
    ;;
  *)
    echo "Usage: bash scripts/run_iluvatar_device.sh [kernel|test]" >&2
    exit 2
    ;;
esac
