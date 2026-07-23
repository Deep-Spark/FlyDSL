# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Shared Iluvatar MMAD geometry and SME view construction."""

import flydsl.expr as fx
import flydsl.expr.ixdl as ixdl

ATOM_M = 16
ATOM_N = 16
ATOM_K = 16
SME_ROWS = 16
SME_BF16_PER_ROW = 32
WARP_SIZE = 64
FRAG_ELEMS = 4
BRICK_ELEMS = SME_ROWS * SME_BF16_PER_ROW
HEAD_DIM = 128
_LOG2E = 1.4426950408889634


def sme_view(base_ptr, elem_type, elem_offset, transpose=False):
    elem_ir_type = elem_type.ir_type if hasattr(elem_type, "ir_type") else elem_type
    smem_ptr = fx.recast_iter(fx.PointerType.get(elem_ir_type, fx.AddressSpace.Shared), base_ptr)
    smem_ptr = fx.add_offset(smem_ptr, fx.make_int_tuple(elem_offset))
    return fx.make_view(
        smem_ptr,
        ixdl.make_sme_shared_layout(
            ixdl.SMESwizzle.Col if transpose else ixdl.SMESwizzle.Row16b,
            elem_type,
            major=ixdl.SMEMajor.K,
        ),
    )
