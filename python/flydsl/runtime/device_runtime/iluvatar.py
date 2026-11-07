# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Iluvatar device runtime."""

from __future__ import annotations

from typing import ClassVar

from .base import DeviceRuntime


class IluvatarDeviceRuntime(DeviceRuntime):
    """Iluvatar runtime; matches compile backend ``iluvatar``."""

    kind: ClassVar[str] = "iluvatar"

    def device_count(self) -> int:
        return 0

    def current_device_id(self) -> int:
        # TODO: Implement this via `ixsmi`
        return 0
