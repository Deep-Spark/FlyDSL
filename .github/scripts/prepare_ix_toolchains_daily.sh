#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Compatibility wrapper. Delegates to two smaller, single-responsibility
# scripts so callers can migrate incrementally:
#
#   prepare_ixcc_refresh.sh   (IXCC working tree; own 30-min cron workflow)
#   prepare_ixsdk_refresh.sh  (IXSDK working tree; daily cron workflow)
#
# The wrapper keeps the pre-split CLI + GitHub Actions outputs
# (`ixcc_commit`, `ixsdk_commit`, `updated_today`) intact so
# perf-daily-iluvatar.yml, which calls this script inline, continues to
# work without changes. Once perf-daily migrates to the split scripts, this
# wrapper can be deleted.
#
# Intentional differences from the pre-split behavior:
#   - IXCC step now flocks its build with the same lock file used by the
#     30-min ixcc-refresh.yaml cron and by wheel builds, so an inline
#     refresh from perf-daily can safely run concurrently with the cron.
#   - IXCC still uses the pre-split "always rebuild when HEAD differs"
#     semantics; the MLIR-only gate is only enabled via the cron workflow,
#     not from this wrapper. (Perf-daily wants the latest IXCC unconditionally.)

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  prepare_ix_toolchains_daily.sh [options]

Options preserved from the pre-split script:
  --ixcc-root PATH             IXCC working repository root.
  --ixsdk-root PATH            IXSDK working repository root.
  --state-file PATH            Retained for CLI compat; unused by wrapper.
  --ixcc-build-marker PATH     Passed through to prepare_ixcc_refresh.sh.
  --ixsdk-install-marker PATH  Passed through to prepare_ixsdk_refresh.sh.
  --ixcc-commit-stamp PATH     Passed through.
  --ixsdk-commit-stamp PATH    Passed through.
  --force-rebuild              Force both sub-scripts to rebuild.
  -h, --help                   Show help.

Env fallbacks (compat):
  IXCC_WORKING_ROOT, IXSDK_WORKING_ROOT,
  IX_TOOLCHAIN_DAILY_STATE_FILE (retained, unused),
  IXCC_BUILD_MARKER, IXSDK_INSTALL_MARKER,
  IXCC_COMMIT_STAMP, IXSDK_COMMIT_STAMP.

Outputs (GitHub Actions):
  ixcc_commit=<short>          Short SHA at end of IXCC step.
  ixsdk_commit=<short>         Short SHA at end of IXSDK step.
  updated_today=<true|false>   True iff either step actually rebuilt.
EOF
}

IXCC_ROOT="${IXCC_WORKING_ROOT:-/home/flydsl/sw_home/sdk/ixcc}"
IXSDK_ROOT="${IXSDK_WORKING_ROOT:-/home/flydsl/sw_home/sdk/ixsdk}"
STATE_FILE="${IX_TOOLCHAIN_DAILY_STATE_FILE:-/var/tmp/flydsl-ix-toolchain-daily.state}"
IXCC_BUILD_MARKER="${IXCC_BUILD_MARKER:-}"
IXSDK_INSTALL_MARKER="${IXSDK_INSTALL_MARKER:-}"
IXCC_COMMIT_STAMP="${IXCC_COMMIT_STAMP:-}"
IXSDK_COMMIT_STAMP="${IXSDK_COMMIT_STAMP:-}"
FORCE_REBUILD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ixcc-root)             IXCC_ROOT="${2:?}"; shift 2 ;;
    --ixsdk-root)            IXSDK_ROOT="${2:?}"; shift 2 ;;
    --state-file)            STATE_FILE="${2:?}"; shift 2 ;;
    --ixcc-build-marker)     IXCC_BUILD_MARKER="${2:?}"; shift 2 ;;
    --ixsdk-install-marker)  IXSDK_INSTALL_MARKER="${2:?}"; shift 2 ;;
    --ixcc-commit-stamp)     IXCC_COMMIT_STAMP="${2:?}"; shift 2 ;;
    --ixsdk-commit-stamp)    IXSDK_COMMIT_STAMP="${2:?}"; shift 2 ;;
    --force-rebuild)         FORCE_REBUILD=1; shift ;;
    -h|--help)               usage; exit 0 ;;
    *)  echo "::error::unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IXCC_SCRIPT="${SELF_DIR}/prepare_ixcc_refresh.sh"
IXSDK_SCRIPT="${SELF_DIR}/prepare_ixsdk_refresh.sh"
[[ -x "${IXCC_SCRIPT}"  ]] || { echo "::error::${IXCC_SCRIPT} not executable" >&2; exit 1; }
[[ -x "${IXSDK_SCRIPT}" ]] || { echo "::error::${IXSDK_SCRIPT} not executable" >&2; exit 1; }

# Capture each sub-script's GITHUB_OUTPUT into a temp file so we can parse
# it without leaking the sub-script's key names into the real workflow
# outputs. The wrapper only re-publishes the pre-split contract.
tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT
ixcc_out="${tmp_dir}/ixcc.out"
ixsdk_out="${tmp_dir}/ixsdk.out"
: >"${ixcc_out}" >"${ixsdk_out}"

# Build IXCC argv from wrapper CLI/env.
ixcc_args=(--ixcc-root "${IXCC_ROOT}")
[[ -n "${IXCC_BUILD_MARKER}"  ]] && ixcc_args+=(--ixcc-build-marker "${IXCC_BUILD_MARKER}")
[[ -n "${IXCC_COMMIT_STAMP}"  ]] && ixcc_args+=(--ixcc-commit-stamp "${IXCC_COMMIT_STAMP}")
[[ "${FORCE_REBUILD}" == "1"  ]] && ixcc_args+=(--force-rebuild)

echo "[ix-toolchain] delegating IXCC step -> prepare_ixcc_refresh.sh"
GITHUB_OUTPUT="${ixcc_out}" "${IXCC_SCRIPT}" "${ixcc_args[@]}"

ixsdk_args=(--ixsdk-root "${IXSDK_ROOT}")
[[ -n "${IXSDK_INSTALL_MARKER}" ]] && ixsdk_args+=(--ixsdk-install-marker "${IXSDK_INSTALL_MARKER}")
[[ -n "${IXSDK_COMMIT_STAMP}"   ]] && ixsdk_args+=(--ixsdk-commit-stamp "${IXSDK_COMMIT_STAMP}")
[[ "${FORCE_REBUILD}" == "1"    ]] && ixsdk_args+=(--force-rebuild)

echo "[ix-toolchain] delegating IXSDK step -> prepare_ixsdk_refresh.sh"
GITHUB_OUTPUT="${ixsdk_out}" "${IXSDK_SCRIPT}" "${ixsdk_args[@]}"

read_kv() {
  local file="$1" key="$2"
  awk -F= -v k="${key}" '$1==k {sub(/^[^=]*=/,""); print; exit}' "${file}"
}

ixcc_commit="$(read_kv "${ixcc_out}" ixcc_commit)"
ixcc_built="$(read_kv "${ixcc_out}" ixcc_built)"
ixsdk_commit="$(read_kv "${ixsdk_out}" ixsdk_commit)"
ixsdk_installed="$(read_kv "${ixsdk_out}" ixsdk_installed)"

updated_today="false"
[[ "${ixcc_built}" == "true" || "${ixsdk_installed}" == "true" ]] && updated_today="true"

# Retained for CLI compat -- some ops tooling reads this file. Fields match
# the pre-split format.
if [[ -n "${STATE_FILE}" ]]; then
  mkdir -p "$(dirname "${STATE_FILE}")"
  {
    echo "date_utc=$(date -u +%Y-%m-%d)"
    echo "ixcc_commit=${ixcc_commit:-unknown}"
    echo "ixsdk_commit=${ixsdk_commit:-unknown}"
    echo "updated_at_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  } > "${STATE_FILE}"
fi

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  {
    echo "ixcc_commit=${ixcc_commit:-unknown}"
    echo "ixsdk_commit=${ixsdk_commit:-unknown}"
    echo "updated_today=${updated_today}"
  } >> "${GITHUB_OUTPUT}"
fi

echo "[ix-toolchain] ixcc_commit=${ixcc_commit:-unknown}"
echo "[ix-toolchain] ixsdk_commit=${ixsdk_commit:-unknown}"
echo "[ix-toolchain] updated_today=${updated_today}"
