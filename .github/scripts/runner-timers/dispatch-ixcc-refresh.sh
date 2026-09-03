#!/usr/bin/env bash
# Dispatch the ixcc-refresh GitHub Actions workflow via `gh workflow run`.
#
# Purpose: runner-local backstop for GitHub Actions dropping scheduled ticks
# on this repo. Both */30 and hourly cron on .github/workflows/ixcc-refresh.yaml
# have been observed to deliver 0 ticks over multi-hour windows (SWCOMP-3177,
# repro: `gh run list -R Deep-Spark/FlyDSL -w ixcc-refresh.yaml -e schedule`).
# The perf-daily-iluvatar canary is a daily-frequency L3 fallback; this
# systemd timer is the sub-hourly L1 path that fully bypasses GH cron.
#
# Runs as user systemd unit ixcc-refresh-dispatch.service, fired every 30 min
# by ixcc-refresh-dispatch.timer under user `flydsl`. Auth uses the gh CLI
# credentials in ~/.config/gh/hosts.yml (PAT scope: repo + workflow).

set -euo pipefail

REPO="${IXCC_REFRESH_REPO:-Deep-Spark/FlyDSL}"
WF="${IXCC_REFRESH_WORKFLOW:-ixcc-refresh.yaml}"
REF="${IXCC_REFRESH_REF:-iluvatar}"
REASON="${IXCC_REFRESH_REASON:-runner-local systemd timer (GH cron unreliable; SWCOMP-3177)}"

if ! command -v gh >/dev/null 2>&1; then
    echo "gh CLI not on PATH ($PATH)" >&2
    exit 1
fi

gh workflow run -R "${REPO}" "${WF}" --ref "${REF}" \
    -f reason="${REASON}"

echo "dispatched ${WF} on ${REPO}@${REF} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
