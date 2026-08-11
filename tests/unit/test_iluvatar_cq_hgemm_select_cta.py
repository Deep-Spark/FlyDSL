# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Unit tests for CQ HGEMM CTA auto-select and CQ device gating (no GPU required)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.l0_backend_agnostic]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _require_select():
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    try:
        from kernels.gemm.iluvatar.cq.hgemm import select_swizzle_cta
    except ModuleNotFoundError as exc:
        pytest.fail(f"failed to import CQ HGEMM select_swizzle_cta: {exc}")
    return select_swizzle_cta


def _require_cq_device_helpers():
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from tests.iluvatar_cq_device import is_iluvatar_cq_cuda_device

    return is_iluvatar_cq_cuda_device


@pytest.mark.parametrize(
    "m,n,k,expected",
    [
        (64, 64, 64, "256"),
        (256, 256, 256, "1024"),
        (512, 512, 512, "1024"),
        (1024, 1024, 1024, "1024"),
        (2048, 2048, 2048, "2048"),
        (3072, 3072, 3072, "2048"),
        (4096, 4096, 4096, "2048"),
        # M not multiple of 512 -> cannot use 2048 preset
        (2304, 2304, 2048, "1024"),
        # min(M,N) < 2048 band -> stay on 1024 even if divisible
        (4096, 256, 4096, "1024"),
    ],
)
def test_select_swizzle_cta_by_shape(m, n, k, expected):
    select = _require_select()
    assert select(m, n, k).name == expected


def test_select_swizzle_cta_rejects_undividable_shape():
    select = _require_select()
    with pytest.raises(ValueError, match="no CQ HGEMM CTA preset"):
        select(48, 48, 48)


def test_is_iluvatar_cq_cuda_device_name_and_smem():
    is_cq = _require_cq_device_helpers()

    cq = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_properties=lambda _i: SimpleNamespace(
                name="Iluvatar TG-V300X-72-A2",
                shared_memory_per_block=196608,
            ),
        )
    )
    mr = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_properties=lambda _i: SimpleNamespace(
                name="Iluvatar BI-V150S",
                shared_memory_per_block=131072,
            ),
        )
    )
    none = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    )
    empty = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 0)
    )

    assert is_cq(cq) is True
    assert is_cq(mr) is False
    assert is_cq(none) is False
    assert is_cq(empty) is False
