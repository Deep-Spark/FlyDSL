# Runner-local systemd timers

Runner-local ops artifacts that back-stop GitHub Actions when the scheduler
misbehaves. Each subject has its own set of `.service` + `.timer` files plus a
tiny dispatch script; `install.sh` at the top drops them under
`~/.local/bin/` and `~/.config/systemd/user/` and enables the timer.

Currently just one timer lives here:

## `ixcc-refresh-dispatch.timer` (30 min)

### Why

`.github/workflows/ixcc-refresh.yaml` is on-schedule `cron: "0 * * * *"`
(originally `"*/30 * * * *"`), but GitHub Actions has been silently dropping
tickets for it on this repo for multi-hour windows. Same window in which
`ix-toolchain-daily-refresh` also missed daily ticks. Not a workflow-config
issue: state is `active`, YAML is fine, manual `workflow_dispatch` works.

Filed under SWCOMP-3177. Repro:

```
gh run list -R Deep-Spark/FlyDSL -w ixcc-refresh.yaml -e schedule
```

Backstop layers, in order of increasing scope:

| layer | trigger | frequency | notes |
|---|---|---|---|
| L1 | this systemd timer | 30 min | primary; fully bypasses GH cron |
| L2 | `on.schedule.cron` in `ixcc-refresh.yaml` | hourly | dormant while GH drops it; auto-recovers |
| L3 | `.build.lock` mtime canary in `perf-daily-iluvatar.yml` | daily | daily-frequency worst-case backstop |

All three call the same workflow; the MLIR gate inside
`prepare_ixcc_refresh.sh` no-ops redundant invocations, so overlap is free.

### What it installs

- `~/.local/bin/dispatch-ixcc-refresh.sh` -- calls `gh workflow run` for
  `Deep-Spark/FlyDSL` `ixcc-refresh.yaml` `@iluvatar`. Overridable via
  `IXCC_REFRESH_REPO` / `IXCC_REFRESH_WORKFLOW` / `IXCC_REFRESH_REF` /
  `IXCC_REFRESH_REASON` env vars.
- `~/.config/systemd/user/ixcc-refresh-dispatch.service` -- `Type=oneshot`
  wrapper around the script above. Sets `HOME=%h` and a minimal `PATH` so
  `gh` finds its config.
- `~/.config/systemd/user/ixcc-refresh-dispatch.timer` --
  `OnUnitActiveSec=30min`, `Persistent=true`, `AccuracySec=30s`,
  `WantedBy=default.target`.

### Prereqs on the target runner

- `systemd` with user-session support (`systemctl --user` works).
- `loginctl enable-linger <user>` so the timer survives logout. The
  installer warns if lingering is off but cannot enable it itself (requires
  root).
- `gh` CLI on PATH, authenticated (`gh auth login`) with a token that has
  `repo` + `workflow` scopes on `Deep-Spark/FlyDSL`. The dispatch call is
  otherwise the only thing that touches GH.

### Install

From the repo root, as the target user (typically `flydsl` on the
Iluvatar self-hosted runner):

```
bash .github/scripts/runner-timers/install.sh
```

The installer is idempotent -- re-running just refreshes the files, reloads
the daemon, and re-enables the timer. Existing timer state and next-fire
schedule are preserved as much as systemd allows.

### Verify

```
systemctl --user list-timers --all ixcc-refresh-dispatch.timer --no-pager
systemctl --user start ixcc-refresh-dispatch.service   # force one now
journalctl --user -u ixcc-refresh-dispatch.service -n 20 --no-pager
gh run list -R Deep-Spark/FlyDSL -w ixcc-refresh.yaml -L 5
```

A fresh dispatch should appear in the GH run list within a few seconds of
`systemctl start` returning.

### Uninstall

```
systemctl --user disable --now ixcc-refresh-dispatch.timer
rm ~/.config/systemd/user/ixcc-refresh-dispatch.{service,timer}
rm ~/.local/bin/dispatch-ixcc-refresh.sh
systemctl --user daemon-reload
```

### Retire

Once GH Actions stops dropping ticks for `ixcc-refresh.yaml` for at least
a full week, and both `ix-toolchain-daily-refresh` and `perf-daily` also
land their scheduled runs on time, this timer becomes redundant. Retire by
uninstalling (above) on every runner where it was installed; the L2 and L3
layers are enough on their own for the healthy case.
