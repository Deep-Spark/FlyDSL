#!/usr/bin/env bash
set -euo pipefail

# ── Required environment variables ────────────────────────────────────
#   GITHUB_REPO_URL   – e.g. https://github.com/Deep-Spark/FlyDSL
#   RUNNER_TOKEN      – registration token from GitHub Settings > Actions > Runners
#
# ── Optional ──────────────────────────────────────────────────────────
#   RUNNER_NAME       – default: hostname
#   RUNNER_LABELS     – default: self-hosted,linux,x64
#   RUNNER_GROUP      – default: Default
#   RUNNER_WORKDIR    – default: _work

RUNNER_NAME="${RUNNER_NAME:-$(hostname)}"
RUNNER_LABELS="${RUNNER_LABELS:-self-hosted,linux,x64}"
RUNNER_GROUP="${RUNNER_GROUP:-Default}"
RUNNER_WORKDIR="${RUNNER_WORKDIR:-_work}"

# ── Configure (only if not already configured) ────────────────────────
if [ ! -f .runner ]; then
    echo ">>> Configuring runner '${RUNNER_NAME}' ..."
    ./config.sh \
        --url "${GITHUB_REPO_URL}" \
        --token "${RUNNER_TOKEN}" \
        --name "${RUNNER_NAME}" \
        --labels "${RUNNER_LABELS}" \
        --runnergroup "${RUNNER_GROUP}" \
        --work "${RUNNER_WORKDIR}" \
        --unattended \
        --replace
fi

# ── Graceful shutdown: deregister on SIGTERM / SIGINT ─────────────────
cleanup() {
    echo ">>> Caught signal, removing runner ..."
    ./config.sh remove --token "${RUNNER_TOKEN}" || true
}
trap cleanup SIGTERM SIGINT

# ── Start the runner ──────────────────────────────────────────────────
echo ">>> Starting runner '${RUNNER_NAME}' ..."
exec ./run.sh
