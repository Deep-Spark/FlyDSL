#!/usr/bin/env bash
# Atomically replace the local latest wheel dir after tests pass.
set -euo pipefail

INCOMING_DIR="${1:?usage: promote_iluvatar_weekly_wheel_local.sh <incoming-dir> <latest-dir>}"
LATEST_DIR="${2:?usage: promote_iluvatar_weekly_wheel_local.sh <incoming-dir> <latest-dir>}"

shopt -s nullglob
incoming_wheels=("${INCOMING_DIR}"/flydsl-*.whl)
if [[ ${#incoming_wheels[@]} -eq 0 ]]; then
  echo "No staged wheels in ${INCOMING_DIR}" >&2
  exit 1
fi

parent="$(dirname "${LATEST_DIR}")"
mkdir -p "${parent}"
tmp_dir="${LATEST_DIR}.tmp.${RANDOM}"
old_dir="${LATEST_DIR}.old"
rm -rf "${tmp_dir}" "${old_dir}"
cp -a "${INCOMING_DIR}" "${tmp_dir}"
{
  echo "promoted_utc=$(date -u +%FT%TZ)"
  echo "github_run_id=${GITHUB_RUN_ID:-}"
} >> "${tmp_dir}/CURRENT.txt"

if [[ -d "${LATEST_DIR}" ]]; then
  mv "${LATEST_DIR}" "${old_dir}"
fi
mv "${tmp_dir}" "${LATEST_DIR}"
rm -rf "${old_dir}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${script_dir}/write_iluvatar_wheel_http_index.sh" "${LATEST_DIR}"
parent="$(dirname "${LATEST_DIR}")"
{
  echo '<!doctype html><meta charset="utf-8"><title>flydsl iluvatar wheels</title><ul>'
  for d in ci latest; do
    if [[ -d "${parent}/${d}" ]]; then
      echo "<li><a href=\"${d}/\">${d}/</a></li>"
    fi
  done
  echo '</ul>'
} > "${parent}/index.html"

echo "Promoted ${INCOMING_DIR} -> ${LATEST_DIR}"
ls -l "${LATEST_DIR}"
