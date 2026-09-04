#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Persistent-tree wheel build inside a corex-base-20.04 container.
#
# Difference vs _build_wheel_fresh_body.sh: this consumes a pre-existing IXCC
# build tree on the runner (mounted read-only) instead of cloning and rebuilding
# LLVM/MLIR per invocation. Saves ~10-40 min per build; runs alongside the
# ixcc-refresh.yaml cron via a shared flock so we never read half-written .so
# files while refresh is rebuilding.
#
# Design contract:
#   - Runs inside a fresh `docker run --rm` container (no state reused across
#     invocations). Root user; chowns outputs back to HOST_UID:HOST_GID before
#     exit so the runner picks them up as its own files.
#   - The IXCC MLIR build tree at ${IXCC_MLIR_CMAKE} is trusted for both
#     -DMLIR_DIR (CMake configure) AND for the .so files that setup.py copies
#     into flydsl/_mlir/_mlir_libs/ in the wheel. That means the tree MUST have
#     been built with the same Python version we are targeting -- otherwise
#     the wheel imports under Python X the _mlir.so compiled against Python Y,
#     which is exactly the PyHeapTypeObject drift that caused SIGSEGV in
#     docs/ixcc-py312-segfault/. We probe the tree's Python ABI tag up front
#     and fail loud if it disagrees with PYTHON_VERSION.
#   - Refresh (prepare_ixcc_refresh.sh) takes `flock -x` on
#     <ixcc-root>/.build.lock; we take `flock -s` on the same file. Multiple
#     concurrent wheel builds share the read side; refresh waits for all of
#     them to finish before it starts writing, and vice versa.
#   - `source sw_home/enable ${PYTHON_VERSION}` still runs. This sets CPATH /
#     CMAKE_ARGS to python${PYTHON_VERSION} headers so FlyDSL's OWN nanobind
#     extension is compiled against the right headers. IXCC MLIR is already
#     built; enable does not rebuild it.
#   - Python packages are strictly pinned to the same set the u2004 build has
#     been validated against; do not loosen without re-testing wheel import.
#
# Required env (all injected by the launcher yaml):
#   PYTHON_VERSION      -- e.g. 3.12; must map to a py${VER} conda env in image
#                          AND to the Python the IXCC MLIR tree was built with
#   IXCC_MLIR_CMAKE     -- path INSIDE the container to the MLIR CMake config
#                          directory, e.g. /ixcc-mlir/lib/cmake/mlir (mounted
#                          from runner's <ixcc-root>/build/lib/cmake/mlir)
#   IXCC_VARIANT        -- internal | external, for logging + manifest only
#   FLYDSL_VERSION_LOCAL_SUFFIX -- setup.py `dev1+<suffix>` local marker
#   CMAKE_BUILD_TYPE    -- Release / Debug (default Release)
#   HOST_UID / HOST_GID -- ownership for /output/*.whl
#
# Required mounts:
#   /flydsl-src (ro)                        -- runner's actions/checkout tree of FlyDSL
#   /output (rw)                            -- where the final .whl gets copied
#   /root/.ccache (rw)                      -- ccache dir persisted on runner
#   <ixcc-root> (ro)                        -- runner's <ixcc-root>; IXCC_MLIR_CMAKE lives
#                                              inside. Mounted at the same in-container path so
#                                              CMake config strings that reference absolute
#                                              paths (e.g. LLVMConfig.cmake) still resolve.
#   <ixcc-root>/.build.lock (rw, file bind) -- single-file rw overlay on top of the ro root
#                                              mount so `flock -s` can acquire the fd.

set -euo pipefail

: "${PYTHON_VERSION:?PYTHON_VERSION required (e.g. 3.12)}"
: "${IXCC_MLIR_CMAKE:?IXCC_MLIR_CMAKE required (in-container path to build/lib/cmake/mlir)}"
: "${IXCC_VARIANT:?IXCC_VARIANT required (internal|external)}"
: "${HOST_UID:?HOST_UID required}"
: "${HOST_GID:?HOST_GID required}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
FLYDSL_VERSION_LOCAL_SUFFIX="${FLYDSL_VERSION_LOCAL_SUFFIX:-local}"

# Derive the ixcc root inside the container. IXCC_MLIR_CMAKE = <root>/build/lib/cmake/mlir,
# so up 4 dirs is <root>. This is where refresh puts .build.lock.
IXCC_ROOT_IN_CONTAINER="$(cd "${IXCC_MLIR_CMAKE}/../../../.." && pwd)"

echo "::group::Persistent-build inputs"
echo "PYTHON_VERSION              = ${PYTHON_VERSION}"
echo "IXCC_VARIANT                = ${IXCC_VARIANT}"
echo "IXCC_MLIR_CMAKE             = ${IXCC_MLIR_CMAKE}"
echo "IXCC_ROOT_IN_CONTAINER      = ${IXCC_ROOT_IN_CONTAINER}"
echo "CMAKE_BUILD_TYPE            = ${CMAKE_BUILD_TYPE}"
echo "FLYDSL_VERSION_LOCAL_SUFFIX = ${FLYDSL_VERSION_LOCAL_SUFFIX}"
echo "HOST_UID:HOST_GID           = ${HOST_UID}:${HOST_GID}"
echo "container user              = $(id)"
grep -E '^(NAME|VERSION_ID)=' /etc/os-release | sed 's/^/  /'
echo "::endgroup::"

# ---- 1. Sanity-check the mounted IXCC tree --------------------------------
echo "::group::Verify IXCC tree"
if [[ ! -f "${IXCC_MLIR_CMAKE}/MLIRConfig.cmake" ]]; then
    echo "::error::MLIRConfig.cmake missing at ${IXCC_MLIR_CMAKE}"
    echo "::error::did the ixcc-refresh cron not run yet? Bootstrap the variant tree first."
    exit 1
fi
if [[ ! -d "${IXCC_ROOT_IN_CONTAINER}/build/tools/mlir/python_packages/mlir_core" ]]; then
    echo "::error::IXCC tree at ${IXCC_ROOT_IN_CONTAINER} was built without MLIR Python bindings."
    echo "::error::rebuild with -DMLIR_ENABLE_BINDINGS_PYTHON=ON (sw_home/build.sh does this)."
    exit 1
fi

# Assert the pre-built _mlir.so ABI tag matches the target Python. If they
# disagree, we will produce a wheel that segfaults on import under PYTHON_VERSION
# (this is the exact 888/920 PyHeapTypeObject drift that PR-A avoided by fresh
# rebuild). Better to fail here than to publish a broken wheel and rely on
# verify_wheel_import to catch it.
py_tag_expected="cpython-${PYTHON_VERSION//./}"
mlir_libs_dir="${IXCC_ROOT_IN_CONTAINER}/build/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs"
if [[ ! -d "${mlir_libs_dir}" ]]; then
    echo "::error::IXCC tree missing ${mlir_libs_dir}"
    exit 1
fi
mapfile -t mlir_so_files < <(find "${mlir_libs_dir}" -maxdepth 1 -name '_mlir.*.so' -printf '%f\n')
if (( ${#mlir_so_files[@]} == 0 )); then
    echo "::error::no _mlir.*.so under ${mlir_libs_dir}"
    ls -la "${mlir_libs_dir}" >&2
    exit 1
fi
matched=0
for so in "${mlir_so_files[@]}"; do
    echo "  found: ${so}"
    if [[ "${so}" == *"${py_tag_expected}"* ]]; then
        matched=1
    fi
done
if [[ "${matched}" != "1" ]]; then
    echo "::error::IXCC MLIR bindings were built for a different Python."
    echo "::error::target=${py_tag_expected}, found=[${mlir_so_files[*]}]"
    echo "::error::rebuild the ${IXCC_VARIANT} IXCC tree with PYTHON_VERSION=${PYTHON_VERSION}."
    exit 1
fi
echo "IXCC bindings ABI tag ${py_tag_expected} matches PYTHON_VERSION ${PYTHON_VERSION}"
echo "::endgroup::"

# ---- 2. Shared read-lock cooperating with ixcc-refresh --------------------
# refresh uses `flock -x` on <ixcc-root>/.build.lock while rebuilding IXCC;
# we take -s so multiple concurrent wheel builds share the read side, but any
# refresh tick will wait for us before writing. The launcher yaml bind-mounts
# just this one file rw over the otherwise-ro <ixcc-root> mount.
LOCK_FILE="${IXCC_ROOT_IN_CONTAINER}/.build.lock"
if [[ ! -e "${LOCK_FILE}" ]]; then
    echo "::error::${LOCK_FILE} not present -- launcher must bind-mount it rw"
    exit 1
fi
echo "::group::flock -s ${LOCK_FILE}"
exec 200>"${LOCK_FILE}"
if ! flock -s -w 3600 200; then
    echo "::error::could not acquire shared lock on ${LOCK_FILE} within 3600s"
    exit 1
fi
echo "acquired shared lock; refresh will wait for this wheel build to finish"
echo "::endgroup::"

# ---- 3. ccache -------------------------------------------------------------
echo "::group::Install ccache"
if ! command -v ccache >/dev/null 2>&1; then
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ccache
fi
export CCACHE_DIR=/root/.ccache
mkdir -p "${CCACHE_DIR}"
ccache --max-size=20G
ccache --show-stats | head -20
echo "::endgroup::"

# ---- 4. Activate the runner-external sw_home in read-only mode ------------
# sw_home is NOT mounted here (we do not need repo-manager / build.sh); we
# only need `source enable ${PYTHON_VERSION}` to set CPATH and PATH to the
# right conda env. In the persistent flow we ship a tiny sw_home clone into
# the container so the enable script + ixconda hooks are available.
cd /home/build
echo "::group::Fresh-clone sw_home (for the enable script)"
if [[ ! -d sw_home/.git ]]; then
    git clone --depth 1 \
        ssh://git@bitbucket.iluvatar.ai:7999/infra/sw_home.git sw_home
fi
git -C sw_home log -1 --format='sw_home @ %h %ci  %s'
echo "::endgroup::"

echo "::group::source enable ${PYTHON_VERSION}"
cd sw_home
# sw_home/enable references $PS1 and a handful of other interactive-shell
# variables. Relax `set -u` for the source itself; restore immediately after.
# shellcheck disable=SC1091
set +u
source ./enable "${PYTHON_VERSION}"
set -u
: "${SW_HOME:?source enable did not set SW_HOME}"
python3 -c "import sys; assert sys.version_info[:2] == tuple(int(x) for x in '${PYTHON_VERSION}'.split('.')), sys.version"
echo "after source enable ${PYTHON_VERSION}:"
env | grep -E '^(PYTHON_VERSION|CONDA_DEFAULT_ENV|CONDA_PREFIX|CPATH|_COREX_PY_INC|CMAKE_ARGS)=' | sed 's/^/  /'
which python3
python3 -V
echo "::endgroup::"

# ---- 5. SSH setup NOT needed (no fresh clones) ----------------------------

# ---- 6. Python build deps (STRICTLY pinned) -------------------------------
# Identical to fresh body; see there for the rationale.
echo "::group::pip install pinned build deps"
python3 -m pip install --no-cache-dir -U \
    pip \
    'setuptools>=77.0.3,<81.0.0' \
    wheel \
    'numpy==1.26.4' \
    'nanobind>=2.9,<3' \
    pybind11 \
    ninja \
    auditwheel \
    patchelf
python3 -m pip list --format=columns | grep -E '^(pip|setuptools|wheel|numpy|nanobind|pybind11|ninja|auditwheel|patchelf)\b' | sed 's/^/  /'
echo "::endgroup::"

# ---- 7. Copy /flydsl-src (ro mount) -> writable tree ----------------------
FLYDSL_SOURCE=/home/build/FlyDSL
echo "::group::Copy FlyDSL source"
rm -rf "${FLYDSL_SOURCE}"
cp -a /flydsl-src "${FLYDSL_SOURCE}"
chown -R root:root "${FLYDSL_SOURCE}"
# Purge host-side stale build state (see fresh body for why).
rm -rf "${FLYDSL_SOURCE}"/build-* "${FLYDSL_SOURCE}"/dist "${FLYDSL_SOURCE}"/*.egg-info
find "${FLYDSL_SOURCE}" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
git -C "${FLYDSL_SOURCE}" submodule update --init --recursive || true
git -C "${FLYDSL_SOURCE}" log -1 --format='flydsl @ %h %ci  %s'
echo "::endgroup::"

# ---- 8. Configure + build FlyDSL against persistent IXCC ------------------
_default_jobs="$(( $(nproc) - 4 ))"
(( _default_jobs > 64 )) && _default_jobs=64
(( _default_jobs < 1 )) && _default_jobs=1
JOBS="${BUILD_JOBS:-${_default_jobs}}"
ulimit -n 65536 2>/dev/null || true

FLYDSL_BUILD="${FLYDSL_SOURCE}/build-fly-release"
echo "::group::cmake configure FlyDSL"
cmake -G Ninja \
    -S "${FLYDSL_SOURCE}" \
    -B "${FLYDSL_BUILD}" \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DFLYDSL_BACKENDS=iluvatar \
    -DMLIR_DIR="${IXCC_MLIR_CMAKE}" \
    -DCUDAToolkit_ROOT="${SW_HOME}/local/corex" \
    -DPython3_EXECUTABLE="$(command -v python3)" \
    -DCMAKE_SHARED_LINKER_FLAGS='-static-libstdc++ -static-libgcc' \
    -DCMAKE_MODULE_LINKER_FLAGS='-static-libstdc++ -static-libgcc'
echo "::endgroup::"

echo "::group::ninja build FlyDSL (-j${JOBS})"
cmake --build "${FLYDSL_BUILD}" -j"${JOBS}"
echo "::endgroup::"

echo "::group::ccache stats after FlyDSL build"
ccache --show-stats | head -30
echo "::endgroup::"

# ---- 9. bdist_wheel -------------------------------------------------------
# MLIR_PATH points at the ixcc root; setup.py walks
# ${MLIR_PATH}/build/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs/
# to find _mlir.so etc. for packaging. This mirrors the u2404 legacy path.
echo "::group::python setup.py bdist_wheel"
cd "${FLYDSL_SOURCE}"
rm -rf dist
FLY_REBUILD=0 \
FLY_BUILD_DIR="${FLYDSL_BUILD}" \
MLIR_PATH="${IXCC_ROOT_IN_CONTAINER}" \
FLYDSL_RELEASE_TYPE=devreleases \
FLYDSL_VERSION_LOCAL_SUFFIX="${FLYDSL_VERSION_LOCAL_SUFFIX}" \
    python3 setup.py bdist_wheel
ls -la dist/
echo "::endgroup::"

# ---- 10. Copy wheel + commit stamps to /output ----------------------------
echo "::group::Copy wheel + commit stamps to /output"
py_tag="cp${PYTHON_VERSION//./}"
shopt -s nullglob
wheels=(dist/*"${py_tag}"*.whl)
shopt -u nullglob
if (( ${#wheels[@]} == 0 )); then
    echo "::error::no wheel matching ${py_tag} produced under dist/"
    ls -la dist/ >&2
    exit 1
fi
cp -v "${wheels[@]}" /output/
# ixcc HEAD comes from the mounted persistent tree; flydsl HEAD from the
# runner-checked-out tree.
git -C "${IXCC_ROOT_IN_CONTAINER}" rev-parse HEAD          > /output/ixcc_commit.txt
git -C "${IXCC_ROOT_IN_CONTAINER}" rev-parse --short HEAD  > /output/ixcc_commit_short.txt
git -C "${FLYDSL_SOURCE}"          rev-parse HEAD          > /output/flydsl_commit.txt
git -C "${FLYDSL_SOURCE}"          rev-parse --short HEAD  > /output/flydsl_commit_short.txt
chown "${HOST_UID}:${HOST_GID}" /output/*.whl /output/*.txt
echo "::endgroup::"

# ---- 11. verify_wheel_import.py ------------------------------------------
# Same safety net as fresh body. If the ABI-tag check above passed but the
# wheel still segfaults, that's a new class of regression (nanobind / setup.py
# packaging bug) worth failing loudly on.
echo "::group::verify_wheel_import"
python3 "${FLYDSL_SOURCE}/.github/scripts/verify_wheel_import.py" \
    /output/*.whl --check-aslr-off
echo "::endgroup::"

echo ">> persistent wheel build OK: /output/$(cd /output && ls *.whl)"
