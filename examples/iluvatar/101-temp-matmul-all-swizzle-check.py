# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Temporary matmul checker for MR async-copy swizzle combinations.

Covers 6 combinations:
  - b8  row / col
  - b16 row / col
  - b32 row / col

Approach per case:
  1) Auto-discover logical->physical shared permutation by copying a known tile
     and dumping shared linear order.
  2) Run a CUDA-core-style scalar-FMA matmul kernel (no tensor core op), where
     A/B are loaded to shared with the target MR async-copy instruction and then
     read back using the discovered permutation.
  3) Compare against torch.matmul reference.
"""

import argparse
import os
from dataclasses import dataclass

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.iluvatar as ix


@dataclass(frozen=True)
class MatmulCase:
    name: str
    copy_ctor_name: str
    elem_bits: int
    tile_m: int
    tile_k: int
    tile_n: int
    torch_dtype: torch.dtype
    fx_dtype: object


CASES = {
    "b8_row": MatmulCase(
        name="b8_row",
        copy_ctor_name="AsyncCopy16x64B8Row",
        elem_bits=8,
        tile_m=16,
        tile_k=64,
        tile_n=64,
        torch_dtype=torch.int8,
        fx_dtype=fx.Int8,
    ),
    "b8_col": MatmulCase(
        name="b8_col",
        copy_ctor_name="AsyncCopy16x64B8Col",
        elem_bits=8,
        tile_m=16,
        tile_k=64,
        tile_n=64,
        torch_dtype=torch.int8,
        fx_dtype=fx.Int8,
    ),
    "b16_row": MatmulCase(
        name="b16_row",
        copy_ctor_name="AsyncCopy16x32B16Row",
        elem_bits=16,
        tile_m=16,
        tile_k=32,
        tile_n=32,
        torch_dtype=torch.bfloat16,
        fx_dtype=fx.BFloat16,
    ),
    "b16_col": MatmulCase(
        name="b16_col",
        copy_ctor_name="AsyncCopy16x32B16Col",
        elem_bits=16,
        tile_m=16,
        tile_k=32,
        tile_n=32,
        torch_dtype=torch.bfloat16,
        fx_dtype=fx.BFloat16,
    ),
    "b32_row": MatmulCase(
        name="b32_row",
        copy_ctor_name="AsyncCopy16x16B32Row",
        elem_bits=32,
        tile_m=16,
        tile_k=16,
        tile_n=16,
        torch_dtype=torch.float32,
        fx_dtype=fx.Float32,
    ),
    "b32_col": MatmulCase(
        name="b32_col",
        copy_ctor_name="AsyncCopy16x16B32Col",
        elem_bits=32,
        tile_m=16,
        tile_k=16,
        tile_n=16,
        torch_dtype=torch.float32,
        fx_dtype=fx.Float32,
    ),
}


def _configure_iluvatar_env() -> None:
    os.environ.setdefault("FLYDSL_COMPILE_BACKEND", "iluvatar")
    os.environ.setdefault("FLYDSL_RUNTIME_KIND", "iluvatar")
    os.environ.setdefault("ARCH", "ivcore11")
    os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    os.environ.pop("COMPILE_ONLY", None)


def _build_discovery_runner(case: MatmulCase):
    tile_m = case.tile_m
    tile_k = case.tile_k
    tile_elems = tile_m * tile_k
    assert tile_elems % 64 == 0
    elems_per_lane = tile_elems // 64
    val_bits = tile_elems * case.elem_bits
    copy_ctor = getattr(ix, case.copy_ctor_name)

    @flyc.kernel
    def _discovery_kernel(
        Src: fx.Tensor,
        Dump: fx.Tensor,
        pitch_elems: fx.Int32,
        pitch_bytes: fx.Int32,
        smem_offset_elems: fx.Int32,
    ):
        tid = fx.thread_idx.x
        g_layout = fx.make_layout((tile_m, tile_k), (pitch_elems, 1))
        s_layout = fx.make_layout((tile_m, tile_k), (tile_k, 1))

        s_base = fx.get_dyn_shared(case.fx_dtype)
        s_ptr = fx.add_offset(s_base, fx.make_int_tuple(smem_offset_elems))
        s_view = fx.make_view(s_ptr, s_layout)
        g_view = fx.make_view(fx.get_iter(Src), g_layout)

        atom = fx.make_copy_atom(copy_ctor(), val_bits)
        atom = atom.set_value("stride_byte", pitch_bytes)
        fx.copy(atom, g_view, s_view)
        ix.cp_async_commit_group()
        ix.cp_async_wait_group(0)

        lane_base = tid * elems_per_lane
        dump_base = fx.get_iter(Dump)
        for i in fx.range_constexpr(elems_per_lane):
            idx = lane_base + i
            src_ptr = fx.add_offset(s_ptr, fx.make_int_tuple(idx))
            dst_ptr = fx.add_offset(dump_base, fx.make_int_tuple(idx))
            fx.ptr_store(fx.ptr_load(src_ptr), dst_ptr)

    @flyc.jit
    def _run(
        Src: fx.Tensor,
        Dump: fx.Tensor,
        pitch_elems: fx.Int32,
        pitch_bytes: fx.Int32,
        smem_offset_elems: fx.Int32,
        smem_total_elems: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        _discovery_kernel(Src, Dump, pitch_elems, pitch_bytes, smem_offset_elems).launch(
            grid=(1, 1, 1),
            block=(64, 1, 1),
            smem=smem_total_elems * (case.elem_bits // 8),
            stream=stream,
        )

    return _run


def _discover_perm(case: MatmulCase, smem_offset_elems: int, smem_total_elems: int) -> torch.Tensor:
    tile_elems = case.tile_m * case.tile_k
    pitch_elems = case.tile_k
    pitch_bytes = pitch_elems * (case.elem_bits // 8)
    assert pitch_bytes % 64 == 0

    run = _build_discovery_runner(case)
    stream = torch.cuda.Stream()

    if case.elem_bits == 8:
        # Use two 8-bit channels to encode logical idx uniquely in [0, 1023].
        logical = torch.arange(tile_elems, dtype=torch.int32, device="cuda")
        low_u8 = logical & 0xFF
        high_u8 = (logical >> 8) & 0xFF
        low_i8 = (low_u8 - 128).to(torch.int8)
        high_i8 = high_u8.to(torch.int8)

        src_low = low_i8.reshape(case.tile_m, case.tile_k).contiguous()
        src_high = high_i8.reshape(case.tile_m, case.tile_k).contiguous()
        dump_low = torch.zeros(tile_elems, dtype=torch.int8, device="cuda")
        dump_high = torch.zeros(tile_elems, dtype=torch.int8, device="cuda")

        run(src_low, dump_low, pitch_elems, pitch_bytes, smem_offset_elems, smem_total_elems, stream=stream)
        run(src_high, dump_high, pitch_elems, pitch_bytes, smem_offset_elems, smem_total_elems, stream=stream)
        torch.cuda.synchronize()

        src_low_cpu = src_low.flatten().cpu().tolist()
        src_high_cpu = src_high.flatten().cpu().tolist()
        dump_low_cpu = dump_low.cpu().tolist()
        dump_high_cpu = dump_high.cpu().tolist()

        pair_to_logical = {}
        for logical_idx in range(tile_elems):
            pair_to_logical[(int(src_low_cpu[logical_idx]), int(src_high_cpu[logical_idx]))] = logical_idx

        perm = torch.full((tile_elems,), -1, dtype=torch.int32, device="cuda")
        perm_cpu = perm.cpu()
        for phys_idx in range(tile_elems):
            pair = (int(dump_low_cpu[phys_idx]), int(dump_high_cpu[phys_idx]))
            logical_idx = pair_to_logical[pair]
            perm_cpu[logical_idx] = phys_idx
        perm = perm_cpu.to("cuda")
    elif case.elem_bits == 16:
        # BF16 cannot represent every integer in [0, 511] exactly, so use 2 channels.
        logical = torch.arange(tile_elems, dtype=torch.int32, device="cuda")
        low = (logical & 0x7F).to(case.torch_dtype)
        high = (logical >> 7).to(case.torch_dtype)
        src_low = low.reshape(case.tile_m, case.tile_k).contiguous()
        src_high = high.reshape(case.tile_m, case.tile_k).contiguous()
        dump_low = torch.zeros(tile_elems, dtype=case.torch_dtype, device="cuda")
        dump_high = torch.zeros(tile_elems, dtype=case.torch_dtype, device="cuda")

        run(src_low, dump_low, pitch_elems, pitch_bytes, smem_offset_elems, smem_total_elems, stream=stream)
        run(src_high, dump_high, pitch_elems, pitch_bytes, smem_offset_elems, smem_total_elems, stream=stream)
        torch.cuda.synchronize()

        src_low_cpu = src_low.flatten().cpu().tolist()
        src_high_cpu = src_high.flatten().cpu().tolist()
        dump_low_cpu = dump_low.cpu().tolist()
        dump_high_cpu = dump_high.cpu().tolist()

        pair_to_logical = {}
        for logical_idx in range(tile_elems):
            pair_to_logical[(float(src_low_cpu[logical_idx]), float(src_high_cpu[logical_idx]))] = logical_idx

        perm = torch.full((tile_elems,), -1, dtype=torch.int32)
        for phys_idx in range(tile_elems):
            pair = (float(dump_low_cpu[phys_idx]), float(dump_high_cpu[phys_idx]))
            logical_idx = pair_to_logical[pair]
            perm[logical_idx] = phys_idx
        perm = perm.to("cuda")
    else:
        src = torch.arange(tile_elems, dtype=case.torch_dtype, device="cuda").reshape(case.tile_m, case.tile_k)
        dump = torch.zeros(tile_elems, dtype=case.torch_dtype, device="cuda")

        run(src, dump, pitch_elems, pitch_bytes, smem_offset_elems, smem_total_elems, stream=stream)
        torch.cuda.synchronize()
        dump_cpu = dump.cpu().tolist()
        perm = torch.full((tile_elems,), -1, dtype=torch.int32)
        for phys_idx, logical_val in enumerate(dump_cpu):
            perm[int(logical_val)] = phys_idx
        perm = perm.to("cuda")

    if (perm < 0).any():
        raise RuntimeError(f"failed to discover full permutation for case {case.name}")
    return perm.contiguous()


def _build_matmul_runner(case: MatmulCase):
    tile_m = case.tile_m
    tile_k = case.tile_k
    tile_n = case.tile_n
    assert tile_k % tile_m == 0
    num_b_chunks = tile_k // tile_m

    a_tile_elems = tile_m * tile_k
    b_chunk_elems = tile_m * tile_n
    out_elems = tile_m * tile_n
    assert out_elems % 64 == 0
    out_per_lane = out_elems // 64

    val_bits_a = a_tile_elems * case.elem_bits
    val_bits_b = b_chunk_elems * case.elem_bits
    smem_elems = a_tile_elems + num_b_chunks * b_chunk_elems
    smem_bytes = smem_elems * (case.elem_bits // 8)
    copy_ctor = getattr(ix, case.copy_ctor_name)
    if case.elem_bits == 8:
        def _load_as_f32(ptr):
            return fx.Float32(fx.Int8(fx.ptr_load(ptr)))
    else:
        def _load_as_f32(ptr):
            return fx.Float32(fx.ptr_load(ptr))

    @flyc.kernel
    def _matmul_kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        PermA: fx.Tensor,
        PermBAll: fx.Tensor,
        pitch_a_elems: fx.Int32,
        pitch_b_elems: fx.Int32,
        pitch_a_bytes: fx.Int32,
        pitch_b_bytes: fx.Int32,
    ):
        tid = fx.thread_idx.x

        s_base = fx.get_dyn_shared(case.fx_dtype)
        sA_ptr = s_base
        sB_ptr = fx.add_offset(s_base, fx.make_int_tuple(a_tile_elems))

        a_layout = fx.make_layout((tile_m, tile_k), (pitch_a_elems, 1))
        sA_layout = fx.make_layout((tile_m, tile_k), (tile_k, 1))
        b_layout = fx.make_layout((tile_m, tile_n), (pitch_b_elems, 1))
        sB_layout = fx.make_layout((tile_m, tile_n), (tile_n, 1))

        gA = fx.make_view(fx.get_iter(A), a_layout)
        sA = fx.make_view(sA_ptr, sA_layout)
        atomA = fx.make_copy_atom(copy_ctor(), val_bits_a).set_value("stride_byte", pitch_a_bytes)
        fx.copy(atomA, gA, sA)

        atomB = fx.make_copy_atom(copy_ctor(), val_bits_b).set_value("stride_byte", pitch_b_bytes)
        b_base = fx.get_iter(B)
        for chunk in fx.range_constexpr(num_b_chunks):
            gB_chunk = fx.make_view(
                fx.add_offset(b_base, fx.make_int_tuple(chunk * tile_m * pitch_b_elems)),
                b_layout,
            )
            sB_chunk_ptr = fx.add_offset(sB_ptr, fx.make_int_tuple(chunk * b_chunk_elems))
            sB_chunk = fx.make_view(sB_chunk_ptr, sB_layout)
            fx.copy(atomB, gB_chunk, sB_chunk)

        ix.cp_async_commit_group()
        ix.cp_async_wait_group(0)

        permA_base = fx.get_iter(PermA)
        permB_base = fx.get_iter(PermBAll)
        c_base = fx.get_iter(C)

        lane_base = tid * out_per_lane
        for i in fx.range_constexpr(out_per_lane):
            out_idx = lane_base + i
            row = out_idx // tile_n
            col = out_idx % tile_n

            acc = fx.Float32(0.0)
            for k in fx.range_constexpr(tile_k):
                a_logical = row * tile_k + k
                a_perm_ptr = fx.add_offset(permA_base, fx.make_int_tuple(a_logical))
                a_phys = fx.ptr_load(a_perm_ptr)
                a_ptr = fx.add_offset(sA_ptr, fx.make_int_tuple(a_phys))

                b_chunk = k // tile_m
                b_row = k % tile_m
                b_logical = b_row * tile_n + col
                b_perm_idx = b_chunk * b_chunk_elems + b_logical
                b_perm_ptr = fx.add_offset(permB_base, fx.make_int_tuple(b_perm_idx))
                b_phys_local = fx.ptr_load(b_perm_ptr)
                b_phys = b_chunk * b_chunk_elems + b_phys_local
                b_ptr = fx.add_offset(sB_ptr, fx.make_int_tuple(b_phys))

                a_val = _load_as_f32(a_ptr)
                b_val = _load_as_f32(b_ptr)
                acc = acc + a_val * b_val

            c_ptr = fx.add_offset(c_base, fx.make_int_tuple(out_idx))
            fx.ptr_store(acc, c_ptr)

    @flyc.jit
    def _run(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        PermA: fx.Tensor,
        PermBAll: fx.Tensor,
        pitch_a_elems: fx.Int32,
        pitch_b_elems: fx.Int32,
        pitch_a_bytes: fx.Int32,
        pitch_b_bytes: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        _matmul_kernel(
            A,
            B,
            C,
            PermA,
            PermBAll,
            pitch_a_elems,
            pitch_b_elems,
            pitch_a_bytes,
            pitch_b_bytes,
        ).launch(grid=(1, 1, 1), block=(64, 1, 1), smem=smem_bytes, stream=stream)

    return _run


def _make_inputs(case: MatmulCase):
    if case.torch_dtype == torch.int8:
        A = torch.randint(-8, 9, (case.tile_m, case.tile_k), dtype=torch.int8, device="cuda")
        B = torch.randint(-8, 9, (case.tile_k, case.tile_n), dtype=torch.int8, device="cuda")
    else:
        A = torch.randn((case.tile_m, case.tile_k), dtype=case.torch_dtype, device="cuda")
        B = torch.randn((case.tile_k, case.tile_n), dtype=case.torch_dtype, device="cuda")
    C = torch.zeros((case.tile_m, case.tile_n), dtype=torch.float32, device="cuda")
    return A, B, C


def _run_case(case: MatmulCase) -> bool:
    print(f"\n[CASE] {case.name} | op={case.copy_ctor_name} | tile=({case.tile_m},{case.tile_k},{case.tile_n})")
    try:
        if case.tile_k != case.tile_n:
            raise RuntimeError("this temporary checker currently expects tile_k == tile_n")

        a_tile_elems = case.tile_m * case.tile_k
        b_chunk_elems = case.tile_m * case.tile_n
        num_b_chunks = case.tile_k // case.tile_m
        smem_total_elems = a_tile_elems + num_b_chunks * b_chunk_elems

        perm_a = _discover_perm(case, smem_offset_elems=0, smem_total_elems=smem_total_elems)
        perm_b_chunks = []
        for chunk in range(num_b_chunks):
            b_off = a_tile_elems + chunk * b_chunk_elems
            perm_b_chunks.append(_discover_perm(case, smem_offset_elems=b_off, smem_total_elems=smem_total_elems))
        perm_b_all = torch.stack(perm_b_chunks, dim=0).reshape(-1).contiguous()

        A, B, C = _make_inputs(case)
        pitch_a_elems = int(A.stride(0))
        pitch_b_elems = int(B.stride(0))
        pitch_a_bytes = pitch_a_elems * (case.elem_bits // 8)
        pitch_b_bytes = pitch_b_elems * (case.elem_bits // 8)
        assert pitch_a_bytes % 64 == 0
        assert pitch_b_bytes % 64 == 0

        run = _build_matmul_runner(case)
        stream = torch.cuda.Stream()
        run(
            A,
            B,
            C,
            perm_a,
            perm_b_all,
            pitch_a_elems,
            pitch_b_elems,
            pitch_a_bytes,
            pitch_b_bytes,
            stream=stream,
        )
        torch.cuda.synchronize()

        ref = torch.matmul(A.float(), B.float())
        max_abs_diff = (C - ref).abs().max().item()
        ok = torch.allclose(C, ref, rtol=1e-3, atol=1e-3)
        print(f"  result_ok={ok} max_abs_diff={max_abs_diff:.6f}")
        if not ok:
            print("  C[:4, :8]:\n", C[:4, :8].cpu())
            print("  Ref[:4, :8]:\n", ref[:4, :8].cpu())
        return ok
    except Exception as exc:
        print(f"  result_ok=False error={exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporary MR async-copy swizzle matmul checker")
    parser.add_argument(
        "--case",
        default="all",
        choices=["all"] + list(CASES.keys()),
        help="Run one case or all",
    )
    args = parser.parse_args()

    _configure_iluvatar_env()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available")

    names = list(CASES.keys()) if args.case == "all" else [args.case]
    all_ok = True
    for name in names:
        all_ok = _run_case(CASES[name]) and all_ok

    print("\n[SUMMARY]")
    print("  all_ok =", all_ok)
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

