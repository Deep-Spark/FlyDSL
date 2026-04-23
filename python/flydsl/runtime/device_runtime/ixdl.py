# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Iluvatar IXDL device runtime (CUDA-compatible host runtime).

Pairs with the ``ixdl`` compile backend. On Iluvatar the host-side runtime
is CUDA-compatible, so ``torch.cuda`` (Iluvatar build) is the device probe
of record here.
"""

from __future__ import annotations

from typing import ClassVar

from ..ixdl_device import get_ixdl_device_count
from .base import DeviceRuntime


class IxdlDeviceRuntime(DeviceRuntime):
    """Iluvatar ``ixdl`` runtime; matches compile backend ``ixdl``.

    ``device_count()`` prefers ``torch.cuda.device_count()`` when the
    Iluvatar-compatible PyTorch is importable (no subprocess), and falls back
    to ``ixsmi -L`` (see :func:`get_ixdl_device_count`). Both paths respect
    ``CUDA_VISIBLE_DEVICES``; ``ixsmi -L`` does not, which is fine since we
    only use it as a last-resort probe.
    """

    kind: ClassVar[str] = "ixdl"

    def device_count(self) -> int:
        try:
            import torch
        except Exception:
            return get_ixdl_device_count()
        if not torch.cuda.is_available():
            return get_ixdl_device_count()
        return torch.cuda.device_count()
