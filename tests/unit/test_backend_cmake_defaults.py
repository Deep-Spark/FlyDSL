# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""CMake backend default and dependency guardrails."""

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l0_backend_agnostic]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cmake_default_backend_stays_rocdl():
    text = (_REPO_ROOT / "cmake" / "FlyDSLBackends.cmake").read_text()

    assert 'set(FLYDSL_BACKENDS "rocdl"' in text
    assert "set_property(CACHE FLYDSL_BACKENDS PROPERTY STRINGS rocdl iluvatar)" in text
    assert "set(_FLYDSL_BACKENDS_ALLOWED rocdl iluvatar)" in text


def test_iluvatar_backend_descriptor_exists():
    descriptor = _REPO_ROOT / "cmake" / "backends" / "iluvatar.cmake"

    assert descriptor.is_file()


def test_rocm_runtime_is_only_added_for_rocdl_backend():
    text = (_REPO_ROOT / "lib" / "Runtime" / "CMakeLists.txt").read_text()

    assert 'if("rocdl" IN_LIST FLYDSL_BACKENDS)' in text
    assert "add_subdirectory(ROCm)" in text


def test_backend_descriptors_are_loaded_from_selected_backend_list():
    text = (_REPO_ROOT / "cmake" / "FlyDSLBackends.cmake").read_text()

    assert "foreach(_backend ${FLYDSL_BACKENDS})" in text
    assert 'include("${CMAKE_CURRENT_LIST_DIR}/backends/${_backend}.cmake")' in text
    assert 'set(FLYDSL_BACKENDS_TUPLE "(${_backends_joined})")' in text


def test_future_backend_descriptor_is_opt_in(tmp_path):
    """A future backend should be legal only when explicitly selected."""
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("cmake not available")

    cmake_dir = tmp_path / "cmake"
    backend_dir = cmake_dir / "backends"
    backend_dir.mkdir(parents=True)

    text = (_REPO_ROOT / "cmake" / "FlyDSLBackends.cmake").read_text()
    text = text.replace(
        "set_property(CACHE FLYDSL_BACKENDS PROPERTY STRINGS rocdl iluvatar)",
        "set_property(CACHE FLYDSL_BACKENDS PROPERTY STRINGS rocdl iluvatar dummy)",
    )
    text = text.replace(
        "set(_FLYDSL_BACKENDS_ALLOWED rocdl iluvatar)",
        "set(_FLYDSL_BACKENDS_ALLOWED rocdl iluvatar dummy)",
    )
    (cmake_dir / "FlyDSLBackends.cmake").write_text(text)

    (backend_dir / "rocdl.cmake").write_text('set(GUARDRAIL_SELECTED_ROCDL ON CACHE BOOL "" FORCE)\n')
    (backend_dir / "dummy.cmake").write_text('set(GUARDRAIL_SELECTED_DUMMY ON CACHE BOOL "" FORCE)\n')
    (tmp_path / "CMakeLists.txt").write_text(
        "\n".join(
            [
                "cmake_minimum_required(VERSION 3.20)",
                "project(FlyDSLBackendSelectionGuardrail NONE)",
                'include("${CMAKE_CURRENT_LIST_DIR}/cmake/FlyDSLBackends.cmake")',
                'if(FLYDSL_BACKENDS STREQUAL "dummy" AND GUARDRAIL_SELECTED_ROCDL)',
                '  message(FATAL_ERROR "rocdl descriptor was included for dummy-only build")',
                "endif()",
                'if(FLYDSL_BACKENDS STREQUAL "rocdl" AND GUARDRAIL_SELECTED_DUMMY)',
                '  message(FATAL_ERROR "dummy descriptor was included for default build")',
                "endif()",
                "",
            ]
        )
    )

    default_build = tmp_path / "build-default"
    subprocess.run(
        [cmake, "-S", str(tmp_path), "-B", str(default_build)],
        check=True,
        text=True,
        capture_output=True,
    )
    default_cache = (default_build / "CMakeCache.txt").read_text()
    assert "FLYDSL_BACKENDS:STRING=rocdl" in default_cache
    assert "GUARDRAIL_SELECTED_ROCDL:BOOL=ON" in default_cache
    assert "GUARDRAIL_SELECTED_DUMMY" not in default_cache

    dummy_build = tmp_path / "build-dummy"
    subprocess.run(
        [cmake, "-S", str(tmp_path), "-B", str(dummy_build), "-DFLYDSL_BACKENDS=dummy"],
        check=True,
        text=True,
        capture_output=True,
    )
    dummy_cache = (dummy_build / "CMakeCache.txt").read_text()
    assert "FLYDSL_BACKENDS:STRING=dummy" in dummy_cache
    assert "GUARDRAIL_SELECTED_DUMMY:BOOL=ON" in dummy_cache
    assert "GUARDRAIL_SELECTED_ROCDL" not in dummy_cache


def test_iluvatar_backend_descriptor_is_opt_in(tmp_path):
    """The real Iluvatar descriptor should only load when selected."""
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("cmake not available")

    cmake_dir = tmp_path / "cmake"
    backend_dir = cmake_dir / "backends"
    runtime_dir = tmp_path / "lib" / "Runtime"
    rocm_runtime_dir = runtime_dir / "ROCm"
    backend_dir.mkdir(parents=True)
    rocm_runtime_dir.mkdir(parents=True)
    (cmake_dir / "FlyDSLBackends.cmake").write_text(
        (_REPO_ROOT / "cmake" / "FlyDSLBackends.cmake").read_text()
    )
    (runtime_dir / "CMakeLists.txt").write_text(
        (_REPO_ROOT / "lib" / "Runtime" / "CMakeLists.txt").read_text()
    )
    (rocm_runtime_dir / "CMakeLists.txt").write_text(
        'set(GUARDRAIL_ENTERED_ROCM_RUNTIME ON CACHE BOOL "" FORCE)\n'
    )

    (backend_dir / "rocdl.cmake").write_text(
        'set(GUARDRAIL_SELECTED_ROCDL ON CACHE BOOL "" FORCE)\n'
    )
    (backend_dir / "iluvatar.cmake").write_text(
        (_REPO_ROOT / "cmake" / "backends" / "iluvatar.cmake").read_text()
        + '\nset(GUARDRAIL_SELECTED_ILUVATAR ON CACHE BOOL "" FORCE)\n'
    )
    (tmp_path / "CMakeLists.txt").write_text(
        "\n".join(
            [
                "cmake_minimum_required(VERSION 3.20)",
                "project(FlyDSLIluvatarSelectionGuardrail NONE)",
                'include("${CMAKE_CURRENT_LIST_DIR}/cmake/FlyDSLBackends.cmake")',
                "get_property(_dialect_includes GLOBAL PROPERTY FLYDSL_BACKEND_INCLUDE_DIALECT_SUBDIRS)",
                "get_property(_dialect_libs GLOBAL PROPERTY FLYDSL_BACKEND_LIB_DIALECT_SUBDIRS)",
                "get_property(_capi_subdirs GLOBAL PROPERTY FLYDSL_BACKEND_CAPI_SUBDIRS)",
                "get_property(_embed_capi_libs GLOBAL PROPERTY FLYDSL_BACKEND_EMBED_CAPI_LIBS)",
                "get_property(_flyopt_libs GLOBAL PROPERTY FLYDSL_BACKEND_FLYOPT_LINK_LIBS)",
                'list(FIND _dialect_includes "FlyIXDL" _flyixdl_include_idx)',
                'list(FIND _dialect_libs "FlyIXDL" _flyixdl_lib_idx)',
                'list(FIND _capi_subdirs "FlyIXDL" _flyixdl_capi_idx)',
                'list(FIND _embed_capi_libs "MLIRCPIFlyIXDL" _flyixdl_embed_idx)',
                'list(FIND _flyopt_libs "MLIRCPIFlyIXDL" _flyixdl_flyopt_idx)',
                'add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/lib/Runtime")',
                'if(FLYDSL_BACKENDS STREQUAL "iluvatar" AND GUARDRAIL_SELECTED_ROCDL)',
                '  message(FATAL_ERROR "rocdl descriptor was included for iluvatar-only build")',
                "endif()",
                'if(FLYDSL_BACKENDS STREQUAL "rocdl" AND GUARDRAIL_SELECTED_ILUVATAR)',
                '  message(FATAL_ERROR "iluvatar descriptor was included for default build")',
                "endif()",
                'if(FLYDSL_BACKENDS STREQUAL "iluvatar" AND GUARDRAIL_ENTERED_ROCM_RUNTIME)',
                '  message(FATAL_ERROR "ROCm runtime was included for iluvatar-only build")',
                "endif()",
                'if(FLYDSL_BACKENDS STREQUAL "rocdl" AND NOT _flyixdl_include_idx EQUAL -1)',
                '  message(FATAL_ERROR "FlyIXDL include dialect subdir was selected for default build")',
                "endif()",
                'if(FLYDSL_BACKENDS STREQUAL "iluvatar" AND _flyixdl_include_idx EQUAL -1)',
                '  message(FATAL_ERROR "FlyIXDL include dialect subdir was not selected")',
                "endif()",
                'if(FLYDSL_BACKENDS STREQUAL "iluvatar" AND _flyixdl_lib_idx EQUAL -1)',
                '  message(FATAL_ERROR "FlyIXDL lib dialect subdir was not selected")',
                "endif()",
                'if(FLYDSL_BACKENDS STREQUAL "iluvatar" AND _flyixdl_capi_idx EQUAL -1)',
                '  message(FATAL_ERROR "FlyIXDL CAPI subdir was not selected")',
                "endif()",
                'if(FLYDSL_BACKENDS STREQUAL "iluvatar" AND _flyixdl_embed_idx EQUAL -1)',
                '  message(FATAL_ERROR "MLIRCPIFlyIXDL embed CAPI lib was not selected")',
                "endif()",
                'if(FLYDSL_BACKENDS STREQUAL "iluvatar" AND _flyixdl_flyopt_idx EQUAL -1)',
                '  message(FATAL_ERROR "MLIRCPIFlyIXDL fly-opt lib was not selected")',
                "endif()",
                "",
            ]
        )
    )

    default_build = tmp_path / "build-default"
    subprocess.run(
        [cmake, "-S", str(tmp_path), "-B", str(default_build)],
        check=True,
        text=True,
        capture_output=True,
    )
    default_cache = (default_build / "CMakeCache.txt").read_text()
    assert "FLYDSL_BACKENDS:STRING=rocdl" in default_cache
    assert "GUARDRAIL_SELECTED_ROCDL:BOOL=ON" in default_cache
    assert "GUARDRAIL_ENTERED_ROCM_RUNTIME:BOOL=ON" in default_cache
    assert "GUARDRAIL_SELECTED_ILUVATAR" not in default_cache

    iluvatar_build = tmp_path / "build-iluvatar"
    subprocess.run(
        [cmake, "-S", str(tmp_path), "-B", str(iluvatar_build), "-DFLYDSL_BACKENDS=iluvatar"],
        check=True,
        text=True,
        capture_output=True,
    )
    iluvatar_cache = (iluvatar_build / "CMakeCache.txt").read_text()
    assert "FLYDSL_BACKENDS:STRING=iluvatar" in iluvatar_cache
    assert "GUARDRAIL_SELECTED_ILUVATAR:BOOL=ON" in iluvatar_cache
    assert "GUARDRAIL_SELECTED_ROCDL" not in iluvatar_cache
    assert "GUARDRAIL_ENTERED_ROCM_RUNTIME" not in iluvatar_cache
