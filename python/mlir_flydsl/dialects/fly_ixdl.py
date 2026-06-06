# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

# isort: skip_file
# ruff: noqa: F401,F403
from ._fly_ixdl_ops_gen import *
from ._fly_ixdl_ops_gen import _Dialect

from .._mlir_libs._mlirDialectsFlyIXDL import *


class _TargetAddressSpace:
    @property
    def SmeGmem(self):
        from .. import ir

        return ir.Attribute.parse("#fly_ixdl.sme_gmem")


TargetAddressSpace = _TargetAddressSpace()
