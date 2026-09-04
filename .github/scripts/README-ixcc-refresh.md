# IXCC refresh on self-hosted Iluvatar runners

The hourly workflow `.github/workflows/ixcc-refresh.yaml` assumes each
Iluvatar runner already has:

- an **internal** IXCC working tree tracking `origin/working`
  (`/home/flydsl/sw_home/sdk/ixcc/`, established long before this workflow),
  and
- an **external** IXCC working tree tracking `origin/xiang.zhang/ixcc-flydsl-release`
  (`/home/flydsl/sw_home/sdk/ixcc-external/`, new -- must be bootstrapped once
  per runner before the cron can succeed for the `external` matrix row).

This file documents the one-time bootstrap. The cron takes over after.

## One-time bootstrap: external tree

Do this on **every** self-hosted runner listed in
`runs-on: [self-hosted, linux, x64, gpu-iluvatar]` for `ixcc-refresh.yaml`.
Run as the `flydsl` user, not root.

```bash
cd /home/flydsl/sw_home/sdk
git clone ssh://git@bitbucket.iluvatar.ai:7999/csys/ixcc.git ixcc-external
cd ixcc-external
git fetch origin xiang.zhang/ixcc-flydsl-release
git checkout xiang.zhang/ixcc-flydsl-release
```

Now do the first build so the MLIR install target exists.
`sw_home/build.sh -r ixcc --host` reads its source tree from
`SW_HOME/sdk/ixcc` by default. We want it to build the *external* tree
instead, so temporarily swap the paths:

```bash
cd /home/flydsl/sw_home/sdk
mv ixcc ixcc.internal.orig
mv ixcc-external ixcc

cd /home/flydsl/sw_home
set +u; source ./enable; set -u
./build.sh -r ixcc --host      # ~30-40min cold

cd /home/flydsl/sw_home/sdk
mv ixcc ixcc-external
mv ixcc.internal.orig ixcc
```

The build must leave the marker file the cron looks for:

```bash
ls -l /home/flydsl/sw_home/sdk/ixcc-external/build/lib/cmake/mlir/MLIRConfig.cmake
```

Seed the commit stamp so the next cron tick doesn't re-classify as
"first-run" and force a rebuild:

```bash
IXCC_EXT=/home/flydsl/sw_home/sdk/ixcc-external
git -C "${IXCC_EXT}" rev-parse HEAD > "${IXCC_EXT}/.flydsl_ixcc_build_commit"
```

## Repo variables

Add these under **Settings > Secrets and variables > Actions > Variables**
(both `flydsl-ci` org and `Deep-Spark/FlyDSL` repo, whichever your runner
consults). Both are optional if the defaults below are correct on every
runner.

| variable                | default (if unset)                            |
|-------------------------|-----------------------------------------------|
| `IXCC_WORKING_ROOT`     | `/home/flydsl/sw_home/sdk/ixcc`               |
| `IXCC_EXTERNAL_ROOT`    | `/home/flydsl/sw_home/sdk/ixcc-external`      |

The persistent-tree wheel path (build-whl-iluvatar.yaml, u2004 persistent
step) picks the MLIR CMake dir from these two variables based on
`ixcc_variant`. `IXCC_MLIR_CMAKE` already exists (consumed by
`ci-device.yml`, `perf-daily-iluvatar.yml`, and `run_*_in_container.sh`)
so PR-B reuses it as the internal-channel dir rather than renaming it
to a hypothetical `IXCC_INTERNAL_MLIR_CMAKE`; only `IXCC_EXTERNAL_MLIR_CMAKE`
is genuinely new. `IXCC_RELEASE_MLIR_CMAKE` (older name for the release /
external tree) is still honoured as a fallback so existing runners keep
working until they migrate.

| variable                    | used by                        | default                                                       |
|-----------------------------|--------------------------------|---------------------------------------------------------------|
| `IXCC_MLIR_CMAKE`           | internal channel + ci-device   | `/home/flydsl/sw_home/sdk/ixcc/build/lib/cmake/mlir`          |
| `IXCC_EXTERNAL_MLIR_CMAKE`  | external channel               | `/home/flydsl/sw_home/sdk/ixcc-external/build/lib/cmake/mlir` |
| `IXCC_RELEASE_MLIR_CMAKE`   | external channel (legacy alias) | (falls back to `IXCC_EXTERNAL_MLIR_CMAKE` default)            |

## Verifying the setup

Dry-run the refresh script against each tree with `--dry-run` (does the
fetch + gate + decision but never touches the tree or invokes build.sh):

```bash
cd $(git rev-parse --show-toplevel)   # FlyDSL checkout on the runner
bash .github/scripts/prepare_ixcc_refresh.sh \
    --ixcc-root  /home/flydsl/sw_home/sdk/ixcc \
    --ixcc-branch working \
    --mlir-gate 'mlir,llvm/include/llvm,llvm/lib/IR,llvm/lib/Support,llvm/lib/Bitcode,llvm/cmake,cmake' \
    --dry-run

bash .github/scripts/prepare_ixcc_refresh.sh \
    --ixcc-root  /home/flydsl/sw_home/sdk/ixcc-external \
    --ixcc-branch xiang.zhang/ixcc-flydsl-release \
    --mlir-gate 'mlir,llvm/include/llvm,llvm/lib/IR,llvm/lib/Support,llvm/lib/Bitcode,llvm/cmake,cmake' \
    --dry-run
```

Expected output on a freshly-bootstrapped, stamp-seeded runner:

```
[ixcc-refresh] stamp == remote; skip
[ixcc-refresh] ixcc_commit=<short> ixcc_built=false ixcc_skip_reason=already-latest
```

Once both dry-runs report `already-latest` (or `no-op-mlir-gate` after
some upstream churn), trigger `ixcc-refresh.yaml` from the Actions tab
with `force_rebuild=false` to smoke-test the real flow.

## Concurrency / lock file

Both variants take an exclusive `flock` on their tree's `.build.lock`
before running `sw_home/build.sh`. The wheel-build workflow (PR-B) will
take a shared `flock` on the *same* file when reading the tree's
`build/`, so a wheel build in flight blocks a refresh rebuild rather
than reading half-linked `.so` files. Default lock file per tree:

- `/home/flydsl/sw_home/sdk/ixcc/.build.lock`
- `/home/flydsl/sw_home/sdk/ixcc-external/.build.lock`

If a runner reboots mid-build, the flock evaporates with the process --
no manual unlock is needed.

## Cron reliability backstops

GitHub Actions has been dropping scheduled ticks for this repo
(SWCOMP-3177), so `ixcc-refresh.yaml` is fronted by two backstops that
both dispatch the same workflow. All three paths share the gate logic in
`prepare_ixcc_refresh.sh`; overlapping dispatches are cheap no-ops --
the first tick after upstream advances rebuilds, the rest gate-skip.

- **L1 runner-local systemd timer.** Every 30 min, fully bypasses GitHub
  cron. See `.github/scripts/runner-timers/README.md` for the install
  script and troubleshooting.
- **L2 GitHub Actions cron.** Hourly on the hour
  (`schedule: cron: '0 * * * *'` in `ixcc-refresh.yaml`). Delivery has
  been unreliable -- treat as best-effort, not primary.
- **L3 perf-daily canary.** `perf-daily-iluvatar.yml` checks each IXCC
  tree's `.build.lock` mtime before its daily run and dispatches
  `ixcc-refresh.yaml` if any tree hasn't been touched for 90 min.

## Escape hatches

- **Force a rebuild anyway.** From the Actions tab, dispatch
  `ixcc-refresh.yaml` with `force_rebuild=true`. This runs both variants
  and bypasses the gate.
- **Skip the MLIR gate for a call.** From the CLI, omit `--mlir-gate`
  (rebuilds on every HEAD advance, matching the pre-split behavior).
- **Rebuild only external.** Not exposed as a dispatch input; run
  `prepare_ixcc_refresh.sh --ixcc-root .../ixcc-external ... --force-rebuild`
  directly on the runner.

## Rollback

If the refresh flow misbehaves and you need to fall back to the pre-PR
behavior:

1. Disable `ixcc-refresh.yaml` in the Actions UI.
2. Stop the runner-local timer:
   `systemctl --user disable --now ixcc-refresh-dispatch.timer`
   (skips the L1 backstop; L3 canary still fires from `perf-daily`).
3. `ix-toolchain-daily-refresh.yml` still runs daily and covers IXSDK;
   IXCC just won't refresh automatically until the workflow + timer
   are re-enabled or a dispatcher is invoked.

There is no data migration or rollback of the trees themselves -- the
refresh gate never rewrites history, only advances `origin/${branch}`.
