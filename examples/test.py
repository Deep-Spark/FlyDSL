#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from flydsl._mlir.ir import FunctionType, InsertionPoint
from flydsl._mlir.dialects import func
import flydsl.expr as fx


def test_sme_swizzle_recast_preserves_mask_base_order(ctx):
    with InsertionPoint(ctx.module.body):
        row_bits = fx.make_composed_layout(
            fx.static(fx.SwizzleType.get(1, 7, 2)),
            fx.make_int_tuple(0),
            fx.make_layout(((2, 8), (16, 16, 2)), ((16, 512), (1, 32, 4096))),
        )
        col_bits = fx.make_composed_layout(
            fx.static(fx.SwizzleType.get(2, 4, 4)),
            fx.make_int_tuple(0),
            fx.make_layout(((4, 4), (32, 4, 4)), ((32, 2048), (1, 512, 128))),
        )

        row_f16 = fx.recast_layout(row_bits, 1, 16)
        col_f16 = fx.recast_layout(col_bits, 1, 16)

    print(row_f16.type)
    print(col_f16.type)
    assert str(row_f16.type).startswith("!fly.composed_layout<S<1,3,2>")
    assert str(col_f16.type).startswith("!fly.composed_layout<S<2,0,4>")
