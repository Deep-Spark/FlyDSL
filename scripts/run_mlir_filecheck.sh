#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Run tests/mlir FileCheck tests only (same logic as scripts/run_tests.sh section 3).
# Safe to run from an interactive shell: failures exit this script, not your login shell.
#
# Usage:
#   bash scripts/run_mlir_filecheck.sh
#   FLY_BUILD_DIR=/path/to/build-fly bash scripts/run_mlir_filecheck.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${FLY_BUILD_DIR:-${REPO_ROOT}/build-fly}"
if [[ "${BUILD_DIR}" != /* ]]; then
  BUILD_DIR="${REPO_ROOT}/${BUILD_DIR}"
fi

FLY_OPT="${BUILD_DIR}/bin/fly-opt"
FILECHECK=""

if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
  _mlir_dir="$(grep '^MLIR_DIR:' "${BUILD_DIR}/CMakeCache.txt" | sed 's|^MLIR_DIR:[A-Z]*=||')"
  [[ -n "${_mlir_dir}" ]] && FILECHECK="${_mlir_dir}/../../../bin/FileCheck"
fi
if [[ -z "${FILECHECK}" ]] || [[ ! -x "${FILECHECK}" ]]; then
  FILECHECK="$(command -v FileCheck || true)"
fi

echo "BUILD_DIR:  ${BUILD_DIR}"
echo "FLY_OPT:    ${FLY_OPT}"
echo "FileCheck:  ${FILECHECK:-<not found>}"
echo ""

if [[ ! -x "${FLY_OPT}" ]]; then
  echo "Error: fly-opt not found or not executable: ${FLY_OPT}" >&2
  echo "Build FlyDSL first:  bash scripts/build.sh" >&2
  exit 1
fi

if [[ -z "${FILECHECK}" ]] || [[ ! -x "${FILECHECK}" ]]; then
  echo "Error: FileCheck not found. Add MLIR install bin to PATH, e.g.:" >&2
  echo "  export PATH=\"\${MLIR_PATH}/bin:\${PATH}\"" >&2
  exit 1
fi

FAILURES=0
TMP_LOG="$(mktemp)"
trap 'rm -f "${TMP_LOG}"' EXIT

while IFS= read -r -d '' f; do
  run_line="$(grep '^// RUN:' "$f" | head -1 | sed 's|^// RUN: *||')"
  [[ -z "${run_line}" ]] && continue
  cmd="$(echo "${run_line}" | sed "s|%fly-opt|${FLY_OPT}|g; s|%FileCheck|${FILECHECK}|g; s|%s|${f}|g; s|FileCheck|${FILECHECK}|g")"
  if eval "${cmd}" >"${TMP_LOG}" 2>&1; then
    echo "  PASS  ${f#${REPO_ROOT}/tests/mlir/}"
  else
    echo "  FAIL  ${f#${REPO_ROOT}/tests/mlir/}"
    tail -20 "${TMP_LOG}" | sed 's/^/        /'
    FAILURES=$((FAILURES + 1))
  fi
done < <(find "${REPO_ROOT}/tests/mlir" -name "*.mlir" -type f -print0 2>/dev/null | sort -z)

echo ""
if [[ "${FAILURES}" -ne 0 ]]; then
  echo "FileCheck: ${FAILURES} file(s) failed." >&2
  exit 1
fi
echo "FileCheck: all tests passed."
