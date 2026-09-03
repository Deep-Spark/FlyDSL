#!/usr/bin/env bash
# Install (or reinstall) the ixcc-refresh dispatch timer under the current
# user's systemd. Idempotent: re-running just refreshes files and reloads.
#
# Prereqs on the target runner:
#   - systemd with user-session support (`systemctl --user` works)
#   - `loginctl enable-linger <user>` so the timer survives logout
#     (the script prints a hint if lingering is off, but does not enable it
#      itself because it requires root)
#   - `gh` CLI on PATH, authenticated for Deep-Spark/FlyDSL with a token
#     that has `repo` + `workflow` scopes (usually via `gh auth login`)
#
# Usage (from the repo root):
#     bash .github/scripts/runner-timers/install.sh
#
# See README.md in this directory for the design and troubleshooting notes.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
target_bin="${HOME}/.local/bin"
target_unit="${HOME}/.config/systemd/user"

mkdir -p "${target_bin}" "${target_unit}"

# --- 1. Copy dispatch script ------------------------------------------------
install -m 0755 "${script_dir}/dispatch-ixcc-refresh.sh" \
    "${target_bin}/dispatch-ixcc-refresh.sh"
echo "installed ${target_bin}/dispatch-ixcc-refresh.sh"

# --- 2. Copy systemd unit files --------------------------------------------
install -m 0644 "${script_dir}/ixcc-refresh-dispatch.service" \
    "${target_unit}/ixcc-refresh-dispatch.service"
install -m 0644 "${script_dir}/ixcc-refresh-dispatch.timer" \
    "${target_unit}/ixcc-refresh-dispatch.timer"
echo "installed ${target_unit}/ixcc-refresh-dispatch.{service,timer}"

# --- 3. Preflight sanity checks --------------------------------------------
if ! command -v gh >/dev/null 2>&1; then
    echo "::error::gh CLI is not on PATH; install it before enabling the timer" >&2
    exit 1
fi
if ! gh auth status >/dev/null 2>&1; then
    echo "::error::gh CLI is not authenticated; run 'gh auth login' first" >&2
    exit 1
fi
if command -v loginctl >/dev/null 2>&1; then
    if ! loginctl show-user "$(whoami)" 2>/dev/null | grep -q '^Linger=yes'; then
        echo "warning: user lingering is not enabled." >&2
        echo "         Run as root: loginctl enable-linger $(whoami)" >&2
        echo "         Otherwise the timer stops when the user logs out." >&2
    fi
fi

# --- 4. Reload + enable + start --------------------------------------------
systemctl --user daemon-reload
systemctl --user enable --now ixcc-refresh-dispatch.timer

# --- 5. Report -------------------------------------------------------------
echo
echo "=== timer status ==="
systemctl --user list-timers --all ixcc-refresh-dispatch.timer --no-pager
echo
echo "next fire time above should be roughly 30 min from now (or sooner on first install)"
echo "verify end-to-end with:"
echo "    systemctl --user start ixcc-refresh-dispatch.service"
echo "    journalctl --user -u ixcc-refresh-dispatch.service -n 20 --no-pager"
