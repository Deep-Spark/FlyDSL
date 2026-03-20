# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Iluvatar Corex device runtime stub (compile backend ``iluvatar``)."""

from __future__ import annotations

import os
from typing import ClassVar

from .base import DeviceRuntime


class IluvatarDeviceRuntime(DeviceRuntime):
    """Placeholder device count for Iluvatar; extend with vendor APIs as needed."""

    kind: ClassVar[str] = "iluvatar"

    def device_count(self) -> int:
        if os.environ.get("SW_HOME") or os.path.exists("/usr/local/corex"):
            return 1
        return 0
