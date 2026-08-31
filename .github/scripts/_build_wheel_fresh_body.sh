#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Fresh-build FlyDSL wheel inside a corex-base-20.04 container.
#
# Design contract:
#   - Runs inside a fresh `docker run --rm` container (no state reused across
#     invocations). Root user; chowns outputs back to HOST_UID:HOST_GID before
#     exit so the runner picks them up as its own files.
#   - `source sw_home/enable ${PYTHON_VERSION}` is the switch that activates
#     the correct conda env AND resets `CPATH` / `CMAKE_ARGS` / `_COREX_PY_INC`
#     to point at python${PYTHON_VERSION}. See docs/ixcc-py312-segfault/ for
#     the 32-byte drift that a non-fresh flow can hit.
#   - IXCC LLVM/MLIR is fresh-cloned per invocation, then compiled with
#     ccache. The compiler-object cache is the only state that persists
#     across invocations (via CCACHE_DIR mount). A cold cache costs ~35min;
#     a warm cache costs ~8-12min.
#   - Python packages are strictly pinned to the same set the u2004 build has
#     been validated against; do not loosen without re-testing wheel import.
#
# Required env (all injected by the launcher yaml):
#   PYTHON_VERSION      -- e.g. 3.12; must map to a py${VER} conda env in image
#   IXCC_REF            -- git ref inside the ixcc repo (e.g.
#                          origin/xiang.zhang/ixcc-flydsl-release)
#   FLYDSL_VERSION_LOCAL_SUFFIX -- setup.py `dev1+<suffix>` local marker
#   CMAKE_BUILD_TYPE    -- Release / Debug (default Release)
#   HOST_UID / HOST_GID -- ownership for /output/*.whl
#
# Required mounts:
#   /host-ssh (ro)  -- ${HOME}/.ssh from runner (contains id_ed25519)
#   /flydsl-src (ro) -- runner's actions/checkout tree of FlyDSL
#   /output   (rw)  -- where the final .whl gets copied
#   /root/.ccache (rw) -- ccache dir persisted on runner

set -euo pipefail

: "${PYTHON_VERSION:?PYTHON_VERSION required (e.g. 3.12)}"
: "${IXCC_REF:?IXCC_REF required (e.g. origin/xiang.zhang/ixcc-flydsl-release)}"
: "${HOST_UID:?HOST_UID required}"
: "${HOST_GID:?HOST_GID required}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
FLYDSL_VERSION_LOCAL_SUFFIX="${FLYDSL_VERSION_LOCAL_SUFFIX:-local}"

echo "::group::Fresh-build inputs"
echo "PYTHON_VERSION            = ${PYTHON_VERSION}"
echo "IXCC_REF                  = ${IXCC_REF}"
echo "CMAKE_BUILD_TYPE          = ${CMAKE_BUILD_TYPE}"
echo "FLYDSL_VERSION_LOCAL_SUFFIX = ${FLYDSL_VERSION_LOCAL_SUFFIX}"
echo "HOST_UID:HOST_GID         = ${HOST_UID}:${HOST_GID}"
echo "container user            = $(id)"
echo "container /etc/os-release :"
grep -E '^(NAME|VERSION_ID)=' /etc/os-release | sed 's/^/  /'
echo "::endgroup::"

# ---- 1. ccache -------------------------------------------------------------
# The corex-base-20.04 image ships without ccache; install it once per
# container. The mounted /root/.ccache is where the cache lives across runs.
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

# ---- 2. SSH key + git config for bitbucket --------------------------------
echo "::group::SSH setup for bitbucket.iluvatar.ai"
install -d -o root -g root -m 700 /root/.ssh
install -o root -g root -m 600 /host-ssh/id_ed25519 /root/.ssh/id_ed25519
if [[ -f /host-ssh/known_hosts ]]; then
    install -o root -g root -m 600 /host-ssh/known_hosts /root/.ssh/known_hosts
fi
export GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=accept-new"
echo "::endgroup::"

# ---- 3. Fresh sw_home + activate py3.12 -----------------------------------
# sw_home ships an `enable` script that (as of the branch matching this
# runtime, 3.10-3.14 case) calls ixconda -> conda activate ${PYTHON_VERSION},
# which fires the corex activate.d hook that resets CPATH / CMAKE_ARGS to
# py${PYTHON_VERSION}. Do NOT try to short-circuit that here -- the whole
# point of the fresh flow is to let those hooks own the toolchain.
cd /home/build
echo "::group::Fresh-clone sw_home"
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

# ---- 4. CoreX runtime (ixdriver / ixcc / ixsdk) fresh download ------------
echo "::group::repo-manager download ixdriver ixcc ixsdk"
repo-manager download ixdriver ixcc ixsdk
echo "::endgroup::"

# ---- 5. IXCC source: fresh-clone at requested ref -------------------------
IXCC_REPO='ssh://git@bitbucket.iluvatar.ai:7999/csys/ixcc.git'
IXCC_SOURCE="${SW_HOME}/sdk/ixcc"
echo "::group::Fresh-clone ixcc @ ${IXCC_REF}"
# repo-manager may have laid down an ixcc worktree; we want ours.
rm -rf "${IXCC_SOURCE}"
git clone "${IXCC_REPO}" "${IXCC_SOURCE}"
git -C "${IXCC_SOURCE}" checkout --detach "${IXCC_REF}"
git -C "${IXCC_SOURCE}" log -1 --format='ixcc @ %h %ci  %s'
echo "::endgroup::"

# ---- 6. Python build deps (STRICTLY pinned) -------------------------------
# Match colleague's known-good pin set. Loosening bounds here has historically
# regressed (numpy 2 ABI, nanobind 3 PyHeapTypeObject changes, setuptools 81+
# entry-point deprecation, etc). Bump deliberately with an accompanying
# `verify_wheel_import.py --check-aslr-off` run.
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

# ---- 7. Configure + build IXCC LLVM/MLIR (ccache) -------------------------
# Job count: leave 4 cores unused so ccache misses do not swamp the runner,
# and cap at 64 -- past that, LLVM link steps thrash on the file system and
# ninja stdout can outpace the runner's pipe buffer, silently truncating logs.
_default_jobs="$(( $(nproc) - 4 ))"
(( _default_jobs > 64 )) && _default_jobs=64
(( _default_jobs < 1 )) && _default_jobs=1
JOBS="${BUILD_JOBS:-${_default_jobs}}"
ulimit -n 65536 2>/dev/null || true

IXCC_BUILD="${IXCC_SOURCE}/build-flydsl"
IXCC_INSTALL="${IXCC_SOURCE}/mlir_install"
NANOBIND_DIR="$(python3 -c "import nanobind, os; print(os.path.dirname(nanobind.__file__) + '/cmake')")"

echo "::group::cmake configure IXCC"
cmake -G Ninja \
    -S "${IXCC_SOURCE}/llvm" \
    -B "${IXCC_BUILD}" \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DLLVM_ENABLE_PROJECTS='mlir;clang;lld' \
    -DLLVM_TARGETS_TO_BUILD='X86;NVPTX;Iluvatar' \
    -DLLVM_ENABLE_RUNTIMES=compiler-rt \
    -DCMAKE_CXX_STANDARD=17 \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_INSTALL_UTILS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DMLIR_BINDINGS_PYTHON_NB_DOMAIN=mlir \
    -DPython3_EXECUTABLE="$(command -v python3)" \
    -Dnanobind_DIR="${NANOBIND_DIR}" \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLVM_BUILD_LLVM_DYLIB=OFF \
    -DLLVM_LINK_LLVM_DYLIB=OFF \
    -DMLIR_INCLUDE_TESTS=OFF \
    -DCMAKE_INSTALL_RPATH='$ORIGIN'
echo "::endgroup::"

echo "::group::ninja build IXCC (-j${JOBS})"
cmake --build "${IXCC_BUILD}" -j"${JOBS}"
echo "::endgroup::"

echo "::group::ccache stats after IXCC build"
ccache --show-stats | head -30
echo "::endgroup::"

echo "::group::cmake install IXCC -> ${IXCC_INSTALL}"
rm -rf "${IXCC_INSTALL}"
cmake --install "${IXCC_BUILD}" --prefix "${IXCC_INSTALL}"
test -d "${IXCC_INSTALL}/lib/cmake/mlir"
echo "::endgroup::"

# ---- 8. FlyDSL: copy read-only mount into a writable tree -----------------
# actions/checkout on the runner produced /flydsl-src (mounted ro). Copy so
# the CMake build tree, setup.py egg-info, dist/ can be written.
FLYDSL_SOURCE=/home/build/FlyDSL
echo "::group::Copy FlyDSL source"
rm -rf "${FLYDSL_SOURCE}"
cp -a /flydsl-src "${FLYDSL_SOURCE}"
# The runner mount preserves host uid (1009); container runs as root. Rechown
# so `git` inside doesn't refuse with "dubious ownership".
chown -R root:root "${FLYDSL_SOURCE}"
# Purge anything the host might have left behind that would confuse the
# container-side build (stale CMake caches pin the source path, dist/ leaks
# non-current wheels, __pycache__ from py3.10 poisons setuptools discovery).
rm -rf "${FLYDSL_SOURCE}"/build-* "${FLYDSL_SOURCE}"/dist "${FLYDSL_SOURCE}"/*.egg-info
find "${FLYDSL_SOURCE}" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
# Some submodules may not have been initialised on the runner; be defensive.
git -C "${FLYDSL_SOURCE}" submodule update --init --recursive || true
git -C "${FLYDSL_SOURCE}" log -1 --format='flydsl @ %h %ci  %s'
echo "::endgroup::"

# ---- 9. Configure + build FlyDSL against fresh IXCC -----------------------
FLYDSL_BUILD="${FLYDSL_SOURCE}/build-fly-release"
echo "::group::cmake configure FlyDSL"
cmake -G Ninja \
    -S "${FLYDSL_SOURCE}" \
    -B "${FLYDSL_BUILD}" \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DFLYDSL_BACKENDS=iluvatar \
    -DMLIR_DIR="${IXCC_BUILD}/lib/cmake/mlir" \
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

# ---- 10. bdist_wheel + auditwheel repair (via setup.py) -------------------
echo "::group::python setup.py bdist_wheel"
cd "${FLYDSL_SOURCE}"
rm -rf dist
FLY_REBUILD=0 \
FLY_BUILD_DIR="${FLYDSL_BUILD}" \
MLIR_PATH="${IXCC_INSTALL}" \
FLYDSL_RELEASE_TYPE=devreleases \
FLYDSL_VERSION_LOCAL_SUFFIX="${FLYDSL_VERSION_LOCAL_SUFFIX}" \
    python3 setup.py bdist_wheel
ls -la dist/
echo "::endgroup::"

# ---- 11. Copy wheel to /output BEFORE verify so it survives verify failures
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
# The outer workflow's manifest step needs the ixcc + flydsl commits. flydsl
# is on the runner, but ixcc is fresh-cloned in here -- write the stamps to
# /output so the runner can pick them up.
git -C "${IXCC_SOURCE}"  rev-parse HEAD         > /output/ixcc_commit.txt
git -C "${IXCC_SOURCE}"  rev-parse --short HEAD > /output/ixcc_commit_short.txt
git -C "${FLYDSL_SOURCE}" rev-parse HEAD         > /output/flydsl_commit.txt
git -C "${FLYDSL_SOURCE}" rev-parse --short HEAD > /output/flydsl_commit_short.txt
chown "${HOST_UID}:${HOST_GID}" /output/*.whl /output/*.txt
echo "::endgroup::"

# ---- 12. verify_wheel_import.py (soft ASLR-off probe, hard ASLR-on) ------
# The ASLR-off column has been the canonical detector for the CPATH/py header
# mismatch. In the fresh flow it should always be 5/5; a regression here means
# ixconda's activate hook stopped resetting CPATH, or nanobind grew a new
# layout-sensitive path.
echo "::group::verify_wheel_import"
python3 "${FLYDSL_SOURCE}/.github/scripts/verify_wheel_import.py" \
    /output/*.whl --check-aslr-off
echo "::endgroup::"

echo ">> fresh wheel build OK: /output/$(cd /output && ls *.whl)"
