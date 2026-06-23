#!/usr/bin/env bash
set -euo pipefail

# Execute Iluvatar L2 device tests inside a CI container while reusing
# host-provided SDK/driver paths (COREX + ixcc MLIR cmake package).

if ! command -v docker >/dev/null 2>&1; then
  echo "::error::docker is required on the self-hosted runner"
  exit 1
fi

: "${CI_DEVICE_IMAGE:?CI_DEVICE_IMAGE is required}"
: "${COREX_ROOT:?COREX_ROOT is required}"
: "${IXCC_MLIR_CMAKE:?IXCC_MLIR_CMAKE is required}"

WORKSPACE="${GITHUB_WORKSPACE:-$(pwd)}"
IXCC_ROOT="$(cd "${IXCC_MLIR_CMAKE}/../../../.." && pwd)"
DEVICE_PYTEST_ARGS_JSON="${DEVICE_PYTEST_ARGS_JSON:-[\"tests/unit\",\"-m\",\"l2_device\"]}"
DEVICE_MUST_PASS_TESTS_JSON="${DEVICE_MUST_PASS_TESTS_JSON:-[]}"
ARCH="${ARCH:-ivcore11}"
CI_DEVICE_CONTAINER_EXTRA_ARGS="${CI_DEVICE_CONTAINER_EXTRA_ARGS:-}"
CI_DEVICE_NETWORK_MODE="${CI_DEVICE_NETWORK_MODE:-bridge}"
CI_DEVICE_IPC_MODE="${CI_DEVICE_IPC_MODE:-private}"
CI_DEVICE_READONLY_ROOTFS="${CI_DEVICE_READONLY_ROOTFS:-1}"
CI_DEVICE_DROP_ALL_CAPS="${CI_DEVICE_DROP_ALL_CAPS:-1}"
CI_DEVICE_NO_NEW_PRIVS="${CI_DEVICE_NO_NEW_PRIVS:-1}"
CI_DEVICE_PIDS_LIMIT="${CI_DEVICE_PIDS_LIMIT:-1024}"
CI_DEVICE_RUN_AS_HOST_USER="${CI_DEVICE_RUN_AS_HOST_USER:-1}"
CI_DEVICE_PRIVILEGED="${CI_DEVICE_PRIVILEGED:-1}"
FLYDSL_ILUVATAR_SMOKE_BLOB_PATH="${FLYDSL_ILUVATAR_SMOKE_BLOB_PATH:-}"
FLYDSL_ILUVATAR_SMOKE_KERNEL="${FLYDSL_ILUVATAR_SMOKE_KERNEL:-}"
FLYDSL_ILUVATAR_LAUNCH_KERNEL="${FLYDSL_ILUVATAR_LAUNCH_KERNEL:-}"
COREX_VERSION_TAG="${COREX_VERSION_TAG:-}"

if [[ ! -d "${WORKSPACE}" ]]; then
  echo "::error::Workspace does not exist: ${WORKSPACE}"
  exit 1
fi
if [[ ! -d "${COREX_ROOT}" ]]; then
  echo "::error::COREX_ROOT does not exist: ${COREX_ROOT}"
  exit 1
fi
if [[ ! -f "${IXCC_MLIR_CMAKE}/MLIRConfig.cmake" ]]; then
  echo "::error::IXCC_MLIR_CMAKE missing MLIRConfig.cmake: ${IXCC_MLIR_CMAKE}"
  exit 1
fi
if [[ ! -d "${IXCC_ROOT}" ]]; then
  echo "::error::Derived IXCC root does not exist: ${IXCC_ROOT}"
  exit 1
fi

mkdir -p "${WORKSPACE}/logs" "${WORKSPACE}/reports"

corex_git_commit=""
corex_version_file=""
corex_lld_version=""
corex_libcuda_md5=""
host_mpi_lib_path=""
host_mpi_lib_dir=""

if [[ -d "${COREX_ROOT}/.git" ]] && command -v git >/dev/null 2>&1; then
  corex_git_commit="$(git -C "${COREX_ROOT}" rev-parse --short HEAD 2>/dev/null || true)"
fi

for vf in VERSION version.txt .version; do
  if [[ -f "${COREX_ROOT}/${vf}" ]]; then
    corex_version_file="$(head -n 1 "${COREX_ROOT}/${vf}" | tr -d '\r' | xargs)"
    break
  fi
done

if [[ -x "${COREX_ROOT}/bin/ld.lld" ]]; then
  corex_lld_version="$("${COREX_ROOT}/bin/ld.lld" --version 2>/dev/null | head -n 1 || true)"
fi

if [[ -f "${COREX_ROOT}/lib64/libcuda.so.1" ]] && command -v md5sum >/dev/null 2>&1; then
  corex_libcuda_md5="$(md5sum "${COREX_ROOT}/lib64/libcuda.so.1" | awk '{print $1}')"
fi

if command -v ldconfig >/dev/null 2>&1; then
  host_mpi_lib_path="$(ldconfig -p 2>/dev/null | awk '/libmpi\.so\.40/ { print $NF; exit }')"
  if [[ -n "${host_mpi_lib_path}" && -f "${host_mpi_lib_path}" ]]; then
    host_mpi_lib_dir="$(dirname "${host_mpi_lib_path}")"
  fi
fi

corex_warning=""
if [[ -z "${COREX_VERSION_TAG}" && -z "${corex_git_commit}" && -z "${corex_version_file}" && -z "${corex_lld_version}" ]]; then
  corex_warning="WARN: unable to infer COREX version metadata; set COREX_VERSION_TAG for traceability"
fi

cat > "${WORKSPACE}/logs/device-container-env.txt" <<EOF
CI_DEVICE_IMAGE=${CI_DEVICE_IMAGE}
COREX_ROOT=${COREX_ROOT}
IXCC_MLIR_CMAKE=${IXCC_MLIR_CMAKE}
IXCC_ROOT=${IXCC_ROOT}
ARCH=${ARCH}
DEVICE_PYTEST_ARGS_JSON=${DEVICE_PYTEST_ARGS_JSON}
DEVICE_MUST_PASS_TESTS_JSON=${DEVICE_MUST_PASS_TESTS_JSON}
FLYDSL_ILUVATAR_SMOKE_BLOB_PATH=${FLYDSL_ILUVATAR_SMOKE_BLOB_PATH}
FLYDSL_ILUVATAR_SMOKE_KERNEL=${FLYDSL_ILUVATAR_SMOKE_KERNEL}
FLYDSL_ILUVATAR_LAUNCH_KERNEL=${FLYDSL_ILUVATAR_LAUNCH_KERNEL}
CI_DEVICE_PRIVILEGED=${CI_DEVICE_PRIVILEGED}
COREX_VERSION_TAG=${COREX_VERSION_TAG}
HOST_MPI_LIB_DIR=${host_mpi_lib_dir}
WORKSPACE=${WORKSPACE}
EOF

{
  echo "COREX_VERSION_TAG: ${COREX_VERSION_TAG:-<unset>}"
  echo "COREX_GIT_COMMIT: ${corex_git_commit:-<unknown>}"
  echo "COREX_VERSION_FILE: ${corex_version_file:-<unknown>}"
  echo "COREX_LLD_VERSION: ${corex_lld_version:-<unknown>}"
  echo "COREX_LIBCUDA_MD5: ${corex_libcuda_md5:-<unknown>}"
  if [[ -n "${corex_warning}" ]]; then
    echo "${corex_warning}"
  fi
} >> "${WORKSPACE}/logs/device-env.txt"

{
  echo "### COREX traceability"
  echo ""
  echo "| key | value |"
  echo "|---|---|"
  echo "| COREX_VERSION_TAG | \`${COREX_VERSION_TAG:-<unset>}\` |"
  echo "| COREX_GIT_COMMIT | \`${corex_git_commit:-<unknown>}\` |"
  echo "| COREX_VERSION_FILE | \`${corex_version_file:-<unknown>}\` |"
  echo "| COREX_LLD_VERSION | \`${corex_lld_version:-<unknown>}\` |"
  echo "| COREX_LIBCUDA_MD5 | \`${corex_libcuda_md5:-<unknown>}\` |"
  if [[ -n "${corex_warning}" ]]; then
    echo ""
    echo "- :warning: ${corex_warning}"
  fi
} > "${WORKSPACE}/logs/corex-version-summary.md"

docker_args=(
  --rm
  --network "${CI_DEVICE_NETWORK_MODE}"
  --ipc "${CI_DEVICE_IPC_MODE}"
  -v "${WORKSPACE}:/workspace"
  -v "${COREX_ROOT}:${COREX_ROOT}:ro"
  -v "${IXCC_ROOT}:${IXCC_ROOT}:ro"
  -v "${IXCC_MLIR_CMAKE}:${IXCC_MLIR_CMAKE}:ro"
  -w /workspace
  -e COREX_ROOT="${COREX_ROOT}"
  -e IXCC_ROOT="${IXCC_ROOT}"
  -e IXCC_MLIR_CMAKE="${IXCC_MLIR_CMAKE}"
  -e ARCH="${ARCH}"
  -e DEVICE_PYTEST_ARGS_JSON="${DEVICE_PYTEST_ARGS_JSON}"
  -e DEVICE_MUST_PASS_TESTS_JSON="${DEVICE_MUST_PASS_TESTS_JSON}"
  -e FLYDSL_ILUVATAR_SMOKE_BLOB_PATH="${FLYDSL_ILUVATAR_SMOKE_BLOB_PATH}"
  -e FLYDSL_ILUVATAR_SMOKE_KERNEL="${FLYDSL_ILUVATAR_SMOKE_KERNEL}"
  -e FLYDSL_ILUVATAR_LAUNCH_KERNEL="${FLYDSL_ILUVATAR_LAUNCH_KERNEL}"
  -e FLYDSL_COMPILE_BACKEND=iluvatar
  -e FLYDSL_RUNTIME_KIND=iluvatar
  -e FLYDSL_RUNTIME_ENABLE_CACHE=0
  -e FLYDSL_ILUVATAR_RUN_JIT_SMOKE=1
  -e CUDAToolkit_ROOT="${COREX_ROOT}"
  -e MLIR_DIR="${IXCC_MLIR_CMAKE}"
  -e HOST_MPI_LIB_DIR="${host_mpi_lib_dir}"
  -e PYTHONUNBUFFERED=1
)

if [[ -n "${host_mpi_lib_dir}" ]]; then
  docker_args+=(-v "${host_mpi_lib_dir}:${host_mpi_lib_dir}:ro")
fi

if [[ "${CI_DEVICE_RUN_AS_HOST_USER}" == "1" ]]; then
  docker_args+=(-u "$(id -u):$(id -g)")
fi

if [[ "${CI_DEVICE_PRIVILEGED}" == "1" ]]; then
  docker_args+=(--privileged)
else
  if [[ "${CI_DEVICE_READONLY_ROOTFS}" == "1" ]]; then
    docker_args+=(--read-only --tmpfs /tmp --tmpfs /var/tmp)
  fi
  if [[ "${CI_DEVICE_DROP_ALL_CAPS}" == "1" ]]; then
    docker_args+=(--cap-drop=ALL)
  fi
  if [[ "${CI_DEVICE_NO_NEW_PRIVS}" == "1" ]]; then
    docker_args+=(--security-opt=no-new-privileges)
  fi
fi
if [[ -n "${CI_DEVICE_PIDS_LIMIT}" ]]; then
  docker_args+=(--pids-limit "${CI_DEVICE_PIDS_LIMIT}")
fi

if [[ -n "${CI_DEVICE_CONTAINER_EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  extra_args=( ${CI_DEVICE_CONTAINER_EXTRA_ARGS} )
  docker_args+=("${extra_args[@]}")
fi

docker run "${docker_args[@]}" \
  "${CI_DEVICE_IMAGE}" \
  bash -lc '
    set -euo pipefail
    export PATH="${COREX_ROOT}/bin:${PATH}"
    if [[ -n "${HOST_MPI_LIB_DIR:-}" ]]; then
      export LD_LIBRARY_PATH="${COREX_ROOT}/lib64:${HOST_MPI_LIB_DIR}:${LD_LIBRARY_PATH:-}"
    else
      export LD_LIBRARY_PATH="${COREX_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
    fi
    export PIP_NO_CACHE_DIR=1
    ASSET_DIR="/workspace/.ci-smoke-assets"
    mkdir -p "${ASSET_DIR}"

    # Inherit image-level Python packages (notably iluvatar-custom torch)
    # to avoid masking them in an isolated CI venv.
    python3 -m venv --system-site-packages /workspace/.ci-venv
    source /workspace/.ci-venv/bin/activate
    python3 -m pip install --upgrade pip
    # Do not use `pip install -e .` in CI device job: editable install triggers
    # setup.py build-time checks (e.g. MLIR_PATH) before our explicit CMake build.
    # Keep this job deterministic by building first, then importing via PYTHONPATH.
    python3 -m pip install --no-cache-dir pytest nanobind pybind11 "numpy<2" patchelf
    if ! command -v patchelf >/dev/null 2>&1; then
      echo "::error::patchelf is required but not found in PATH after pip install"
      exit 1
    fi

    image_site_packages=""
    venv_site_packages=""
    for p in /opt/venv/lib/python*/site-packages; do
      if [[ -d "${p}" ]]; then
        image_site_packages="${p}"
        break
      fi
    done
    for p in /workspace/.ci-venv/lib/python*/site-packages; do
      if [[ -d "${p}" ]]; then
        venv_site_packages="${p}"
        break
      fi
    done
    if [[ -z "${image_site_packages}" ]]; then
      echo "::error::cannot locate image python site-packages under /opt/venv"
      exit 1
    fi
    if [[ -z "${venv_site_packages}" ]]; then
      echo "::error::cannot locate venv python site-packages under /workspace/.ci-venv"
      exit 1
    fi
    # Keep venv-installed numpy preferred, while making image-level torch importable.
    printf "%s\n" "${image_site_packages}" > "${venv_site_packages}/_ci_image_site_packages.pth"

    python3 - <<'PY'
import torch
print(f"torch visible in Run C venv: {torch.__version__}")
PY

    cmake -S . -B build-fly -G Ninja \
      -DFLYDSL_BACKENDS=iluvatar \
      -DMLIR_DIR="${MLIR_DIR}" \
      -DCUDAToolkit_ROOT="${CUDAToolkit_ROOT}" \
      -DPython3_EXECUTABLE="$(command -v python3)"
    cmake --build build-fly -j"$(nproc)"

    # Required by compile-only must-pass tests in test_iluvatar_binary_pipeline_smoke.py.
    fly_opt_bin=""
    for c in /workspace/build-fly/bin/fly-opt /workspace/build-fly/tools/fly-opt/fly-opt; do
      if [[ -x "${c}" ]]; then
        fly_opt_bin="${c}"
        break
      fi
    done
    if [[ -z "${fly_opt_bin}" ]]; then
      echo "::error::missing fly-opt binary: /workspace/build-fly/bin/fly-opt (or legacy tools/fly-opt path)"
      exit 1
    fi
    export FLYDSL_ILUVATAR_FLY_OPT="${fly_opt_bin}"

    runtime_lib="/workspace/build-fly/python_packages/flydsl/_mlir/_mlir_libs/libfly_iluvatar_jit_runtime.so"
    if [[ ! -f "${runtime_lib}" ]]; then
      echo "::error::missing runtime library: ${runtime_lib}"
      exit 1
    fi
    export FLYDSL_ILUVATAR_JIT_RUNTIME_LIB="${runtime_lib}"
    # Use built package + source tree directly for tests.
    export PYTHONPATH="/workspace/build-fly/python_packages:/workspace/python:/workspace:${PYTHONPATH:-}"

    mapfile -t must_pass < <(python3 - <<'"'"'PY'"'"'
import json
import os
for a in json.loads(os.environ["DEVICE_MUST_PASS_TESTS_JSON"]):
    print(a)
PY
)
    if [[ "${#must_pass[@]}" -eq 0 ]]; then
      echo "::error::DEVICE_MUST_PASS_TESTS_JSON resolved to empty list"
      exit 1
    fi

    need_runtime_smoke=0
    for t in "${must_pass[@]}"; do
      if [[ "$t" == "tests/unit/test_iluvatar_runtime_smoke.py" ]]; then
        need_runtime_smoke=1
      fi
    done

    if [[ "${need_runtime_smoke}" == "1" ]]; then
      if [[ -z "${FLYDSL_ILUVATAR_SMOKE_BLOB_PATH:-}" || -z "${FLYDSL_ILUVATAR_SMOKE_KERNEL:-}" || -z "${FLYDSL_ILUVATAR_LAUNCH_KERNEL:-}" ]]; then
        echo "::error::runtime smoke is must-pass but blob path/kernel names are not fully configured"
        exit 1
      fi
      if [[ "${FLYDSL_ILUVATAR_SMOKE_BLOB_PATH}" = /* ]]; then
        blob_path="${FLYDSL_ILUVATAR_SMOKE_BLOB_PATH}"
      else
        blob_path="/workspace/${FLYDSL_ILUVATAR_SMOKE_BLOB_PATH}"
      fi
      if [[ ! -f "${blob_path}" ]]; then
        echo "::error::runtime smoke blob not found in repository path: ${blob_path}"
        exit 1
      fi
      export FLYDSL_ILUVATAR_SMOKE_BLOB="${blob_path}"
      export FLYDSL_ILUVATAR_SMOKE_KERNEL
      export FLYDSL_ILUVATAR_LAUNCH_KERNEL
    fi

    set +e
    python3 -m pytest "${must_pass[@]}" -v -r a --junitxml=reports/device-must-pass.xml 2>&1 | tee logs/device-must-pass.log
    must_pass_status=${PIPESTATUS[0]}
    set -e

    must_pass_skipped="$(python3 - <<'"'"'PY'"'"'
import os
import xml.etree.ElementTree as ET

xml_path = "reports/device-must-pass.xml"
summary_path = "logs/device-must-pass-summary.md"

if not os.path.exists(xml_path):
    with open(summary_path, "w", encoding="utf-8") as out:
        out.write("### must-pass summary\n\n- junit xml missing: reports/device-must-pass.xml\n")
    print(0)
    raise SystemExit(0)

root = ET.parse(xml_path).getroot()
suite_nodes = [root] if root.tag == "testsuite" else list(root.findall(".//testsuite"))
cases = []
skipped = 0
for suite in suite_nodes:
    for case in suite.findall("testcase"):
        file_name = case.attrib.get("file") or case.attrib.get("classname", "")
        status = "passed"
        if case.find("failure") is not None or case.find("error") is not None:
            status = "failed"
        elif case.find("skipped") is not None:
            status = "skipped"
            skipped += 1
        cases.append((file_name, case.attrib.get("name", ""), status))

with open(summary_path, "w", encoding="utf-8") as out:
    out.write("### must-pass summary\n\n")
    out.write("| file | test | status |\n")
    out.write("|---|---|---|\n")
    for file_name, test_name, status in cases:
        out.write(f"| `{file_name}` | `{test_name}` | `{status}` |\n")
    out.write("\n")
    out.write(f"- skipped: {skipped}\n")

print(skipped)
PY
)"

    if [[ "${must_pass_status}" -ne 0 ]]; then
      echo "::error::must-pass tests failed"
      exit "${must_pass_status}"
    fi
    if [[ "${must_pass_skipped}" -gt 0 ]]; then
      echo "::error::must-pass tests contain skipped cases (${must_pass_skipped})"
      exit 1
    fi

    mapfile -t args < <(python3 - <<'"'"'PY'"'"'
import json
import os
for a in json.loads(os.environ["DEVICE_PYTEST_ARGS_JSON"]):
    print(a)
PY
)

    # Full device stage runs on Iluvatar runtime. Exclude ROCm-lower tests to
    # avoid mixing incompatible backend/device paths in one pytest invocation.
    marker_idx=-1
    for i in "${!args[@]}"; do
      if [[ "${args[$i]}" == "-m" || "${args[$i]}" == "--markexpr" ]]; then
        marker_idx=$i
        break
      fi
    done
    if [[ "${marker_idx}" -ge 0 && $((marker_idx + 1)) -lt "${#args[@]}" ]]; then
      marker_expr="${args[$((marker_idx + 1))]}"
      if [[ "${marker_expr}" == *"l2_device"* && "${marker_expr}" != *"rocm_lower"* ]]; then
        args[$((marker_idx + 1))]="(${marker_expr}) and not rocm_lower"
      fi
    fi

    set -o pipefail
    python3 -m pytest "${args[@]}" -v --junitxml=reports/device-full.xml 2>&1 | tee logs/device-full.log
  '
