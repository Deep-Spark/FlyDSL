# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Temporary visual checker for Iluvatar MR async-copy swizzle behavior.

Usage examples:
  .venv/bin/python examples/iluvatar/99-temp-swizzle-visual-check.py --case b8_row_4x64
  .venv/bin/python examples/iluvatar/99-temp-swizzle-visual-check.py --case b16_row_16x32
  .venv/bin/python examples/iluvatar/99-temp-swizzle-visual-check.py --case b32_col_16x16
  .venv/bin/python examples/iluvatar/99-temp-swizzle-visual-check.py --case all

What this script prints per case:
  1) source tile (logical order)
  2) shared linear dump (physical linear readback from smem)
  3) decoded dump (case-specific inverse mapping if available)

This is intended for manual/eyeball verification while bringing up swizzle paths.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Callable

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.iluvatar as ix


@dataclass(frozen=True)
class SwizzleCase:
    name: str
    copy_ctor_name: str
    elem_bits: int
    tile_m: int
    tile_n: int
    torch_dtype: torch.dtype
    fx_dtype: object
    decode_mode: str


CASES = {
    "b8_row_4x64": SwizzleCase(
        name="b8_row_4x64",
        copy_ctor_name="AsyncCopy4x64B8Row",
        elem_bits=8,
        tile_m=4,
        tile_n=64,
        torch_dtype=torch.int8,
        fx_dtype=fx.Int8,
        decode_mode="mr4x64_b8_row_perm",
    ),
    "b16_row_16x32": SwizzleCase(
        name="b16_row_16x32",
        copy_ctor_name="AsyncCopy16x32B16Row",
        elem_bits=16,
        tile_m=16,
        tile_n=32,
        torch_dtype=torch.int16,
        fx_dtype=fx.Int16,
        decode_mode="mr16x32_b16_row_perm",
    ),
    "b32_col_16x16": SwizzleCase(
        name="b32_col_16x16",
        copy_ctor_name="AsyncCopy16x16B32Col",
        elem_bits=32,
        tile_m=16,
        tile_n=16,
        torch_dtype=torch.int32,
        fx_dtype=fx.Int32,
        decode_mode="mr16x16_b32_col_perm",
    ),
}


def _configure_iluvatar_env() -> None:
    os.environ.setdefault("FLYDSL_COMPILE_BACKEND", "iluvatar")
    os.environ.setdefault("FLYDSL_RUNTIME_KIND", "iluvatar")
    os.environ.setdefault("ARCH", "ivcore11")
    os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    os.environ.pop("COMPILE_ONLY", None)


def _make_source_tile(case: SwizzleCase) -> torch.Tensor:
    total = case.tile_m * case.tile_n
    # Use int64 for generation then cast to keep deterministic pattern in all dtypes.
    vals = torch.arange(total, dtype=torch.int64, device="cuda").reshape(case.tile_m, case.tile_n)
    return vals.to(case.torch_dtype)


def _print_slice(title: str, tensor: torch.Tensor, rows: int = 6, cols: int = 16) -> None:
    r = min(rows, tensor.shape[0])
    c = min(cols, tensor.shape[1])
    print(f"{title} (showing {r}x{c}):")
    print(tensor[:r, :c].cpu())
    print()


def _build_runner(case: SwizzleCase) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int, int], None]:
    tile_m = case.tile_m
    tile_n = case.tile_n
    tile_elems = tile_m * tile_n
    assert tile_elems % 64 == 0
    elems_per_lane = tile_elems // 64
    tile_bits = tile_elems * case.elem_bits
    smem_bytes = tile_elems * (case.elem_bits // 8)
    copy_ctor = getattr(ix, case.copy_ctor_name)

    @flyc.kernel
    def _kernel(
        A: fx.Tensor,
        B_linear: fx.Tensor,
        B_decoded: fx.Tensor,
        pitch_elems: fx.Int32,
        pitch_bytes: fx.Int32,
    ):
        tid = fx.thread_idx.x
        g_src_layout = fx.make_layout((tile_m, tile_n), (pitch_elems, 1))
        s_layout = fx.make_layout((tile_m, tile_n), (tile_n, 1))

        s_base = fx.get_dyn_shared(case.fx_dtype)
        sA = fx.make_view(s_base, s_layout)

        atom = fx.make_copy_atom(copy_ctor(), tile_bits)
        atom = atom.set_value("stride_byte", pitch_bytes)

        gA = fx.make_view(fx.get_iter(A), g_src_layout)
        gB_linear = fx.get_iter(B_linear)
        gB_decoded = fx.get_iter(B_decoded)

        fx.copy(atom, gA, sA)
        ix.cp_async_commit_group()
        ix.cp_async_wait_group(0)

        lane_base = tid * elems_per_lane
        for i in fx.range_constexpr(elems_per_lane):
            elem_idx = lane_base + i

            # 1) Physical linear readback from shared memory.
            src_linear_ptr = fx.add_offset(s_base, fx.make_int_tuple(elem_idx))
            dst_linear_ptr = fx.add_offset(gB_linear, fx.make_int_tuple(elem_idx))
            fx.ptr_store(fx.ptr_load(src_linear_ptr), dst_linear_ptr)

            # 2) Decoded readback (identity unless a case-specific inverse map is provided).
            decoded_idx = elem_idx
            if case.decode_mode == "mr4x64_b8_row_perm":
                # For 4x64.b8.row, the observed smem physical index is:
                #   p = col * 4 + row, with logical idx l = row * 64 + col.
                row = elem_idx // tile_n
                col = elem_idx % tile_n
                decoded_idx = col * tile_m + row
            elif case.decode_mode == "mr16x32_b16_row_perm":
                # For 16x32.b16.row, inverse mapping from logical (row, col) to
                # physical linear shared index:
                #   rg = row // 2, row_lo = row % 2
                #   col_hi = col // 16, col_lo = col % 16
                #   pr = (col_hi * 8) + (rg xor (col_hi * 2))
                #   pc = col_lo * 2 + row_lo
                #   p = pr * 32 + pc
                row = elem_idx // tile_n
                col = elem_idx % tile_n
                rg = row // 2
                row_lo = row % 2
                col_hi = col // 16
                col_lo = col % 16
                pr = col_hi * 8 + (rg ^ (col_hi * 2))
                pc = col_lo * 2 + row_lo
                decoded_idx = pr * tile_n + pc
            elif case.decode_mode == "mr16x16_b32_col_perm":
                # For 16x16.b32.col, inverse mapping from logical (row, col) to
                # physical linear shared index:
                #   rg = row // 4, row_lo = row % 4
                #   cg = col // 4, col_lo = col % 4
                #   pr = rg * 4 + col_lo
                #   pc = (cg xor rg) * 4 + row_lo
                #   p = pr * 16 + pc
                row = elem_idx // tile_n
                col = elem_idx % tile_n
                rg = row // 4
                row_lo = row % 4
                cg = col // 4
                col_lo = col % 4
                pr = rg * 4 + col_lo
                pc = (cg ^ rg) * 4 + row_lo
                decoded_idx = pr * tile_n + pc

            src_decoded_ptr = fx.add_offset(s_base, fx.make_int_tuple(decoded_idx))
            dst_decoded_ptr = fx.add_offset(gB_decoded, fx.make_int_tuple(elem_idx))
            fx.ptr_store(fx.ptr_load(src_decoded_ptr), dst_decoded_ptr)

    @flyc.jit
    def _run(
        A: fx.Tensor,
        B_linear: fx.Tensor,
        B_decoded: fx.Tensor,
        pitch_elems: fx.Int32,
        pitch_bytes: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        _kernel(A, B_linear, B_decoded, pitch_elems, pitch_bytes).launch(
            grid=(1, 1, 1), block=(64, 1, 1), smem=smem_bytes, stream=stream
        )

    return _run


def _run_one_case(case: SwizzleCase) -> None:
    print("=" * 100)
    print(f"[CASE] {case.name} | op={case.copy_ctor_name} | tile={case.tile_m}x{case.tile_n} | bits={case.elem_bits}")
    print("=" * 100)

    pitch_elems = case.tile_n
    pitch_bytes = pitch_elems * (case.elem_bits // 8)
    if pitch_bytes % 64 != 0:
        raise RuntimeError(f"pitch_bytes must be 64B aligned, got {pitch_bytes}")

    src_tile = _make_source_tile(case)
    A = src_tile.contiguous()
    B_linear = torch.zeros(case.tile_m * case.tile_n, dtype=case.torch_dtype, device="cuda")
    B_decoded = torch.zeros(case.tile_m * case.tile_n, dtype=case.torch_dtype, device="cuda")

    runner = _build_runner(case)
    stream = torch.cuda.Stream()
    runner(A, B_linear, B_decoded, pitch_elems, pitch_bytes, stream=stream)
    torch.cuda.synchronize()

    linear_mat = B_linear.reshape(case.tile_m, case.tile_n)
    decoded_mat = B_decoded.reshape(case.tile_m, case.tile_n)

    _print_slice("source(logical)", src_tile)
    _print_slice("smem_linear_dump(physical linear)", linear_mat)
    _print_slice("smem_decoded_dump(case inverse map)", decoded_mat)

    linear_match = torch.equal(src_tile, linear_mat)
    decoded_match = torch.equal(src_tile, decoded_mat)
    print(f"linear_match_source = {linear_match}")
    print(f"decoded_match_source = {decoded_match}")

    if not linear_match:
        diff = (src_tile.to(torch.int64) - linear_mat.to(torch.int64)).abs().max().item()
        print(f"max_abs_diff(source vs linear) = {diff}")
    if not decoded_match:
        diff = (src_tile.to(torch.int64) - decoded_mat.to(torch.int64)).abs().max().item()
        print(f"max_abs_diff(source vs decoded) = {diff}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporary visual checker for Iluvatar MR async-copy swizzle")
    parser.add_argument(
        "--case",
        default="all",
        choices=["all"] + list(CASES.keys()),
        help="Which case to run",
    )
    args = parser.parse_args()

    _configure_iluvatar_env()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available")

    if args.case == "all":
        for key in CASES:
            _run_one_case(CASES[key])
    else:
        _run_one_case(CASES[args.case])


if __name__ == "__main__":
    main()

