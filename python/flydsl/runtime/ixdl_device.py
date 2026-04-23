# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Iluvatar IXDL device helpers.

Arch detection is minimal on purpose:
* ``FLYDSL_GPU_ARCH`` overrides everything (same env as ROCm path);
* otherwise default to ``ivcore11`` — covers both MR-V100 and MR-V50.

No ``ixsmi`` probe here — we do not need a runtime shell-out just to pick a
chip name, and Iluvatar's ``#ixdl.target`` already defaults to ``ivcore11``.
"""

from __future__ import annotations

import functools
import os
import subprocess
from typing import Optional


_IXDL_DEFAULT_ARCH = "ivcore11"


@functools.lru_cache(maxsize=None)
def get_ixdl_arch() -> str:
    """Best-effort Iluvatar GPU arch string (e.g. ``ivcore11``)."""
    override = os.environ.get("FLYDSL_GPU_ARCH")
    if override:
        return override
    return _IXDL_DEFAULT_ARCH


@functools.lru_cache(maxsize=None)
def get_ixdl_device_count() -> int:
    """Best-effort visible Iluvatar GPU count via ``ixsmi -L``.

    Returns 0 when ``ixsmi`` is unavailable or produces no GPU lines. Call
    sites must not treat 0 as "no GPU present"; it only means "can't probe".
    """
    try:
        out = subprocess.check_output(
            ["ixsmi", "-L"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return 0

    count = 0
    for line in out.splitlines():
        if line.strip().lower().startswith("gpu "):
            count += 1
    return count


def get_visible_ixdl_device_index() -> Optional[int]:
    """Return the first visible device index (respects ``CUDA_VISIBLE_DEVICES``).

    Iluvatar honors ``CUDA_VISIBLE_DEVICES`` for device selection; downstream
    code that needs "which physical card will we run on" should consult this.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None or cvd.strip() == "":
        return 0
    first = cvd.split(",")[0].strip()
    if not first:
        return None
    try:
        return int(first)
    except ValueError:
        return None
