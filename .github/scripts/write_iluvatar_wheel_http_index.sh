#!/usr/bin/env bash
# Write a pip --find-links index.html for a wheel directory.
set -euo pipefail

DIR="${1:?usage: write_iluvatar_wheel_http_index.sh <wheel-dir>}"
shopt -s nullglob
{
  echo '<!doctype html><meta charset="utf-8"><title>flydsl iluvatar wheels</title><ul>'
  for whl in "${DIR}"/flydsl-*.whl; do
    base="$(basename "${whl}")"
    echo "<li><a href=\"${base}\">${base}</a></li>"
  done
  echo '</ul>'
} > "${DIR}/index.html"
