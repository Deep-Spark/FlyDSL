#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Refresh the IXSDK working tree on the runner: fetch, checkout, ff-pull,
# and re-run `make install` if the tracked branch head advanced or the
# install marker went missing.
#
# This is the IXSDK-only successor to prepare_ix_toolchains_daily.sh.
# Cadence is daily (ix-toolchain-daily-refresh.yml); the IXCC half moved
# to its own 30-min workflow (ixcc-refresh.yaml). Keeping IXSDK on daily
# matches the pre-split behavior -- SDK changes are infrequent and
# `make install` is not gated by a "MLIR-only" filter.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  prepare_ixsdk_refresh.sh --ixsdk-root DIR [options]

Required:
  --ixsdk-root DIR          IXSDK working repository root.

Options:
  --ixsdk-branch REF        Remote branch to track. Default: working
  --ixsdk-install-marker P  Optional file that must exist for the install
                            to be considered valid. Default: empty (only
                            the commit stamp is checked).
  --ixsdk-commit-stamp P    Records the commit last installed. Default:
                              <ixsdk-root>/.flydsl_ixsdk_install_commit
  --force-rebuild           Ignore stamp; reinstall unconditionally.
  -h, --help                Show help.

Env fallbacks (only used when the matching flag is omitted):
  IXSDK_WORKING_ROOT, IXSDK_BRANCH, IXSDK_INSTALL_MARKER,
  IXSDK_COMMIT_STAMP.

Outputs (GitHub Actions):
  ixsdk_commit=<short>      Short SHA at end of run.
  ixsdk_installed=<bool>    Whether `make install` ran this call.
EOF
}

IXSDK_ROOT="${IXSDK_WORKING_ROOT:-}"
IXSDK_BRANCH="${IXSDK_BRANCH:-working}"
IXSDK_INSTALL_MARKER="${IXSDK_INSTALL_MARKER:-}"
IXSDK_COMMIT_STAMP="${IXSDK_COMMIT_STAMP:-}"
FORCE_REBUILD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ixsdk-root)             IXSDK_ROOT="${2:?}"; shift 2 ;;
    --ixsdk-branch)           IXSDK_BRANCH="${2:?}"; shift 2 ;;
    --ixsdk-install-marker)   IXSDK_INSTALL_MARKER="${2:?}"; shift 2 ;;
    --ixsdk-commit-stamp)     IXSDK_COMMIT_STAMP="${2:?}"; shift 2 ;;
    --force-rebuild)          FORCE_REBUILD=1; shift ;;
    -h|--help)                usage; exit 0 ;;
    *)  echo "::error::unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

command -v git >/dev/null || { echo "::error::git not found"; exit 1; }

if [[ -z "${IXSDK_ROOT}" ]]; then
  echo "::error::--ixsdk-root (or IXSDK_WORKING_ROOT) is required" >&2
  exit 2
fi
if [[ ! -d "${IXSDK_ROOT}/.git" ]]; then
  echo "::error::IXSDK root is not a git repository: ${IXSDK_ROOT}" >&2
  exit 1
fi

git config --global --add safe.directory "${IXSDK_ROOT}" || true

[[ -n "${IXSDK_COMMIT_STAMP}" ]] || IXSDK_COMMIT_STAMP="${IXSDK_ROOT}/.flydsl_ixsdk_install_commit"

read_stamp() {
  [[ -f "$1" ]] && tr -d '[:space:]' < "$1" || true
}
write_stamp() {
  mkdir -p "$(dirname "$1")"
  printf '%s\n' "$2" > "$1"
}

echo "[ixsdk-refresh] root=${IXSDK_ROOT} branch=${IXSDK_BRANCH}"
git -C "${IXSDK_ROOT}" fetch --quiet origin "${IXSDK_BRANCH}"
git -C "${IXSDK_ROOT}" checkout "${IXSDK_BRANCH}"

local_head="$(git -C "${IXSDK_ROOT}" rev-parse HEAD)"
remote_head="$(git -C "${IXSDK_ROOT}" rev-parse "origin/${IXSDK_BRANCH}")"
stamp_commit="$(read_stamp "${IXSDK_COMMIT_STAMP}")"
marker_ok=1
if [[ -n "${IXSDK_INSTALL_MARKER}" && ! -e "${IXSDK_INSTALL_MARKER}" ]]; then
  marker_ok=0
fi

echo "[ixsdk-refresh] local_head=${local_head:0:12} remote_head=${remote_head:0:12}"
echo "[ixsdk-refresh] stamp=${stamp_commit:0:12} marker_ok=${marker_ok}"

needs_install=0
if [[ "${FORCE_REBUILD}" == "1" ]]; then
  needs_install=1
  echo "[ixsdk-refresh] --force-rebuild set"
elif [[ "${local_head}" != "${remote_head}" ]]; then
  needs_install=1
  echo "[ixsdk-refresh] local != remote"
elif [[ "${stamp_commit}" != "${remote_head}" ]]; then
  needs_install=1
  echo "[ixsdk-refresh] stamp missing/mismatch"
elif [[ "${marker_ok}" == "0" ]]; then
  needs_install=1
  echo "[ixsdk-refresh] install marker missing"
fi

if [[ "${needs_install}" == "1" ]]; then
  git -C "${IXSDK_ROOT}" pull --ff-only origin "${IXSDK_BRANCH}"
  echo "[ixsdk-refresh] make install"
  (cd "${IXSDK_ROOT}" && make install)
  new_short="$(git -C "${IXSDK_ROOT}" rev-parse --short HEAD)"
  new_full="$(git -C "${IXSDK_ROOT}" rev-parse HEAD)"
  write_stamp "${IXSDK_COMMIT_STAMP}" "${new_full}"
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    {
      echo "ixsdk_commit=${new_short}"
      echo "ixsdk_installed=true"
    } >> "${GITHUB_OUTPUT}"
  fi
  echo "[ixsdk-refresh] installed ixsdk_commit=${new_short}"
else
  short="$(git -C "${IXSDK_ROOT}" rev-parse --short HEAD)"
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    {
      echo "ixsdk_commit=${short}"
      echo "ixsdk_installed=false"
    } >> "${GITHUB_OUTPUT}"
  fi
  echo "[ixsdk-refresh] ixsdk_commit=${short} ixsdk_installed=false"
fi
