#!/usr/bin/env bash
# Locate the Python interpreter for a target version inside an Iluvatar CI
# container image (stable = ghcr.io/deep-spark/flydsl-iluvatar-ci:stable for
# u2404, corex-base-20.04 for u2004 fresh, and their variants).
#
# Walks three fallback tiers, in order:
#   1. /usr/local/conda/envs/py<ver>/bin/python
#   2. /opt/conda/envs/py<ver>/bin/python
#   3. python<ver> on PATH
#
# We ship this fallback because the stable image has moved py<ver> between
# /usr/local/conda and /opt/conda across rebuilds. On 2026-09-06 (run
# 34019397427) the weekly wheel cron's u2404 verify step blew up at tier 1
# while the same job's build step -- which already inlined the fallback --
# passed. Centralising here so build and verify cannot drift again.
#
# Not for the u2004 build path, which resolves Python via sw_home/enable
# (a completely different mechanism); and not for the persistent-tree build
# body, which likewise leans on sw_home. This helper is for containers that
# ship a conda env at a well-known-ish path and nothing else.
#
# Usage:
#   source .github/scripts/resolve_container_python.sh
#   py="$(resolve_container_python 3.12)"
#
# Prints the interpreter path to stdout on success; returns non-zero and
# prints a diagnostic to stderr on failure.

resolve_container_python() {
    local ver="${1:-}"
    if [[ -z "${ver}" ]]; then
        echo "resolve_container_python: version required (e.g. 3.12)" >&2
        return 2
    fi
    if [[ -x "/usr/local/conda/envs/py${ver}/bin/python" ]]; then
        printf '%s\n' "/usr/local/conda/envs/py${ver}/bin/python"
    elif [[ -x "/opt/conda/envs/py${ver}/bin/python" ]]; then
        printf '%s\n' "/opt/conda/envs/py${ver}/bin/python"
    elif command -v "python${ver}" >/dev/null 2>&1; then
        command -v "python${ver}"
    else
        echo "resolve_container_python: cannot find python ${ver} at any known location" >&2
        return 1
    fi
}
