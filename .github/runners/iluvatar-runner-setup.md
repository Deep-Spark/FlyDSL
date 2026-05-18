# Iluvatar self-hosted runner setup (deepspark)

One-time preparation on the runner machine before
[`.github/workflows/iluvatar-manual.yaml`](../workflows/iluvatar-manual.yaml)
can succeed. The CI only runs `cmake/ninja` plus the test commands; everything
below must already be present on the runner.

## Runner registration

Register a GitHub Actions self-hosted runner with labels:

```text
self-hosted, linux, X64, gpu-iluvatar
```

Steps (run as the operator user, **not** root):

```bash
mkdir -p ~/actions-runner && cd ~/actions-runner
# Download URL / token from: GitHub repo -> Settings -> Actions -> Runners -> New self-hosted runner
curl -o actions-runner.tar.gz -L <URL_FROM_GITHUB>
tar xzf actions-runner.tar.gz
./config.sh --url https://github.com/Deep-Spark/FlyDSL \
            --token <REGISTRATION_TOKEN> \
            --labels self-hosted,linux,X64,gpu-iluvatar \
            --unattended
# Install as a service so it survives reboots.
sudo ./svc.sh install
sudo ./svc.sh start
```

Confirm the runner shows `Idle` under Settings -> Actions -> Runners.

## Toolchain paths consumed by the workflow

The workflow reads these from the `env:` block; override them by editing the
workflow if your runner uses different paths.

| Variable | Default | Must contain |
|----------|---------|--------------|
| `IXCC_MLIR_CMAKE` | `/home/wcyx/sw_home/sdk/ixcc/build/lib/cmake/mlir` | ixcc 22.x MLIR build, has `MLIRConfig.cmake` and IXDL targets |
| `COREX_ROOT`      | `/home/wcyx/sw_home/local/corex`                   | `bin/ld.lld` (Iluvatar LLD), `lib64/libcuda.so.1` (with `ixdrvInit`) |
| `FLYDSL_SHARED_VENV` | `/home/flydsl/FlyDSL/.venv`                     | Python 3.10+ venv with the deps listed below |

Quick check:

```bash
test -f /home/wcyx/sw_home/sdk/ixcc/build/lib/cmake/mlir/MLIRConfig.cmake
test -x /home/wcyx/sw_home/local/corex/bin/ld.lld
test -e /home/wcyx/sw_home/local/corex/lib64/libcuda.so.1
nm -D /home/wcyx/sw_home/local/corex/lib64/libcuda.so.1 | grep -q ixdrvInit
test -x /home/flydsl/FlyDSL/.venv/bin/python3
```

## Shared venv contents

The runner user must own `${FLYDSL_SHARED_VENV}` and have it pre-populated with:

- `nanobind`, `pybind11`, `numpy` (FlyDSL build-time deps)
- `pytest` (any 9.x is fine)
- **CoreX-compatible** PyTorch wheel from 天数 (not the pip official NVIDIA CUDA build)

One-time bootstrap with [`uv`](https://github.com/astral-sh/uv):

```bash
cd /home/flydsl/FlyDSL
uv venv .venv
source .venv/bin/activate
uv pip install nanobind pybind11 numpy pytest
# Install the CoreX-supplied torch wheel (path provided by the Iluvatar SDK):
uv pip install /path/to/corex/torch-*.whl
```

Cross-check inside the venv:

```bash
.venv/bin/python -c "import nanobind, pybind11, numpy, pytest; print('build/test deps OK')"
LD_LIBRARY_PATH=/home/wcyx/sw_home/local/corex/lib64 \
  .venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Permissions

- Runner service user must read `${IXCC_MLIR_CMAKE}` and `${COREX_ROOT}` (chmod
  `o+rX` recursively if those trees live under another user's home).
- Runner service user must read/write the workflow checkout directory
  (`_work/FlyDSL/FlyDSL`); by default it owns this path.
- GPU access (`/dev/iluvatar*` or equivalent device nodes) must be granted to
  the runner user.

## Workflow trigger

After the runner is online, on the Deep-Spark/FlyDSL repo:

1. Open the **Actions** tab.
2. Choose **Iluvatar Manual CI**.
3. Click **Run workflow**, pick the branch / ref, optionally toggle
   `run_store_kernel` or `clean_build`.

Concurrency is queued (`cancel-in-progress: false`); back-to-back dispatches on
the same ref serialize so the GPU is never shared.

## Troubleshooting

| Symptom | Likely fix |
|---------|------------|
| `MISSING: .../MLIRConfig.cmake` (pre-flight) | Build ixcc 22.x or fix `IXCC_MLIR_CMAKE` |
| `libcublas.so.*[0-9] not found` while importing torch | Re-install CoreX torch wheel; ensure `LD_LIBRARY_PATH` is set via the env-export step (workflow does this automatically) |
| `ld.lld: error: unsupported e_machine value: 248` | `${COREX_ROOT}/bin` not in PATH; check that the env-export step ran |
| `undefined symbol: ixdrvInit` | System NVIDIA `libcuda.so.1` shadowing CoreX libs in `LD_LIBRARY_PATH` |
| GPU not available in `torch.cuda.is_available()` | Device permissions for the runner user, or `IXVISIBLE_DEVICES` set incorrectly |
