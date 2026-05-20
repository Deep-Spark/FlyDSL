# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Iluvatar helpers aligned with ``flydsl.expr.rocdl.universal`` patterns."""

from ..primitive import get_iter, get_layout, make_view
from ..typing import Tensor


def make_global_tensor(tensor: Tensor) -> Tensor:
    """Prepare a global tensor view for IXDL async copy.

    ROCm kernels typically call ``fx.rocdl.make_buffer_tensor`` to attach a
    buffer descriptor while preserving layout. Iluvatar MR async copy uses
    plain global pointers; this helper is the layout-preserving analog.

    Row pitch and tile origin still come from the usual layout API:
    ``make_layout`` / ``add_offset`` / ``slice`` / ``zipped_divide``.
    For dynamic pitch, set ``stride_byte`` on the copy atom.
    """
    return make_view(get_iter(tensor), get_layout(tensor))
