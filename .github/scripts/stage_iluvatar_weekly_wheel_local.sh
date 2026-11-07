#!/usr/bin/env bash
# Copy weekly Iluvatar wheels to a fixed local directory, overwriting in place.
set -euo pipefail

DIST_DIR="${1:?usage: stage_iluvatar_weekly_wheel_local.sh <dist-dir> <dest-dir>}"
DEST_DIR="${2:?usage: stage_iluvatar_weekly_wheel_local.sh <dist-dir> <dest-dir>}"

mapfile -t wheels < <(find "${DIST_DIR}" -type f -name 'flydsl-*.whl' ! -name 'flydsl-iluvatar-*-manylinux_x86_64.whl' | sort)
if [[ ${#wheels[@]} -eq 0 ]]; then
  echo "No flydsl wheels found under ${DIST_DIR}" >&2
  exit 1
fi

mkdir -p "${DEST_DIR}"
rm -f "${DEST_DIR}"/flydsl-*.whl
rm -f "${DEST_DIR}"/CURRENT.txt "${DEST_DIR}"/index.html

copied=()
for whl in "${wheels[@]}"; do
  base="$(basename "${whl}")"
  cp -f "${whl}" "${DEST_DIR}/${base}"
  copied+=("${base}")
done
if [[ ${#copied[@]} -eq 0 ]]; then
  echo "No flydsl wheels copied from ${DIST_DIR}" >&2
  exit 1
fi
bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/write_iluvatar_wheel_http_index.sh" "${DEST_DIR}"

manifest="$(find "${DIST_DIR}" -type f -name 'build-manifest.json' | head -n 1 || true)"
if [[ -n "${manifest}" ]]; then
  cp -f "${manifest}" "${DEST_DIR}/build-manifest.json"
fi

{
  echo "updated_utc=$(date -u +%FT%TZ)"
  echo "source_ref=${SOURCE_REF:-}"
  echo "github_run_id=${GITHUB_RUN_ID:-}"
  echo "github_run_url=${GITHUB_SERVER_URL:-}/${GITHUB_REPOSITORY:-}/actions/runs/${GITHUB_RUN_ID:-}"
  echo "wheels=${copied[*]}"
} > "${DEST_DIR}/CURRENT.txt"

echo "Staged local weekly wheels in ${DEST_DIR}"
printf '  %s\n' "${copied[@]}"
ls -l "${DEST_DIR}"
