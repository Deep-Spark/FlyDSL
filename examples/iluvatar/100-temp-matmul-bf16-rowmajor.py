# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Temporary BF16 row-major GEMM checker using MR async copy.

Goal:
  - Validate MR async-copy correctness in a real compute flow (matmul),
    while computing with scalar CUDA-core-style FMA (no tensor core MMA op).
  - A/B are BF16 row-major. C is FP32 accumulation/output.

Current shape:
  - A: [16, 64] (bf16, row-major)
  - B: [64, 32] (bf16, row-major)
  - C: [16, 32] (fp32)
"""

import os

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.iluvatar as ix

M = 16
K = 64
N = 32

TILE_M = 16
TILE_K = 32
TILE_N = 32
TILE_ELEMS_B16 = TILE_M * TILE_K
TILE_BITS_B16 = TILE_ELEMS_B16 * 16

# Shared-memory layout:
#   sA  : 16x32 bf16 (512 elems)
#   sB0 : 16x32 bf16 (512 elems)
#   sB1 : 16x32 bf16 (512 elems)
S_A_OFFSET = 0
S_B0_OFFSET = TILE_ELEMS_B16
S_B1_OFFSET = TILE_ELEMS_B16 * 2
SMEM_ELEMS_TOTAL = TILE_ELEMS_B16 * 3
SMEM_BYTES = SMEM_ELEMS_TOTAL * 2

# 64-thread warp-cooperative launch. Each lane computes 8 output elements.
OUT_ELEMS = M * N
assert OUT_ELEMS % 64 == 0
OUT_PER_LANE = OUT_ELEMS // 64


@flyc.kernel
def bf16_rowmajor_matmul_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    pitch_a_elems: fx.Int32,
    pitch_b_elems: fx.Int32,
    pitch_a_bytes: fx.Int32,
    pitch_b_bytes: fx.Int32,
    k_runtime: fx.Int32,
):
    tid = fx.thread_idx.x

    s_base = fx.get_dyn_shared(fx.BFloat16)
    sA_ptr = fx.add_offset(s_base, fx.make_int_tuple(S_A_OFFSET))
    sB0_ptr = fx.add_offset(s_base, fx.make_int_tuple(S_B0_OFFSET))
    sB1_ptr = fx.add_offset(s_base, fx.make_int_tuple(S_B1_OFFSET))

    tile_layout = fx.make_layout((TILE_M, TILE_K), (TILE_K, 1))
    sA = fx.make_view(sA_ptr, tile_layout)
    sB0 = fx.make_view(sB0_ptr, tile_layout)
    sB1 = fx.make_view(sB1_ptr, tile_layout)

    a_layout = fx.make_layout((TILE_M, TILE_K), (pitch_a_elems, 1))
    b_layout = fx.make_layout((TILE_M, TILE_N), (pitch_b_elems, 1))

    a_base = fx.get_iter(A)
    b_base = fx.get_iter(B)
    c_base = fx.get_iter(C)

    g2s_a = fx.make_copy_atom(ix.AsyncCopy16x32B16Row(), TILE_BITS_B16)
    g2s_b = fx.make_copy_atom(ix.AsyncCopy16x32B16Row(), TILE_BITS_B16)
    g2s_a = g2s_a.set_value("stride_byte", pitch_a_bytes)
    g2s_b = g2s_b.set_value("stride_byte", pitch_b_bytes)

    lane_base = tid * OUT_PER_LANE
    init_state = [fx.Float32(0.0) for _ in fx.range_constexpr(OUT_PER_LANE)]
    results = init_state
    for k_block, state in range(0, k_runtime, TILE_K, init=init_state):
        accs = list(state)
        k_block_i32 = fx.Int32(k_block)
        a_block_base = fx.add_offset(a_base, fx.make_int_tuple(k_block_i32))
        gA_block = fx.make_view(a_block_base, a_layout)

        b_block_base = fx.add_offset(b_base, fx.make_int_tuple(k_block_i32 * pitch_b_elems))
        gB0_block = fx.make_view(b_block_base, b_layout)
        gB1_block = fx.make_view(fx.add_offset(b_block_base, fx.make_int_tuple(TILE_M * pitch_b_elems)), b_layout)

        # Async copy is intentionally issued inside the K-loop per tile.
        fx.copy(g2s_a, gA_block, sA)
        fx.copy(g2s_b, gB0_block, sB0)
        fx.copy(g2s_b, gB1_block, sB1)
        ix.cp_async_commit_group()
        ix.cp_async_wait_group(0)

        for k in fx.range_constexpr(TILE_K):
            # 16x32.b16.row logical(row, col) -> physical linear index:
            #   rg = row // 2, row_lo = row % 2
            #   col_hi = col // 16, col_lo = col % 16
            #   pr = col_hi * 8 + (rg xor (col_hi * 2))
            #   pc = col_lo * 2 + row_lo
            #   p = pr * 32 + pc
            a_col_hi = k // 16
            a_col_lo = k % 16
            b_chunk = k // TILE_M
            b_row = k % TILE_M

            for i in fx.range_constexpr(OUT_PER_LANE):
                out_idx = lane_base + i
                row = out_idx // N
                col = out_idx % N

                a_rg = row // 2
                a_row_lo = row % 2
                a_pr = a_col_hi * 8 + (a_rg ^ (a_col_hi * 2))
                a_pc = a_col_lo * 2 + a_row_lo
                a_idx = S_A_OFFSET + a_pr * TILE_K + a_pc
                a_ptr = fx.add_offset(s_base, fx.make_int_tuple(a_idx))

                b_rg = b_row // 2
                b_row_lo = b_row % 2
                b_col_hi = col // 16
                b_col_lo = col % 16
                b_pr = b_col_hi * 8 + (b_rg ^ (b_col_hi * 2))
                b_pc = b_col_lo * 2 + b_row_lo
                b_idx = S_B0_OFFSET + b_chunk * (TILE_M * TILE_N) + b_pr * TILE_N + b_pc
                b_ptr = fx.add_offset(s_base, fx.make_int_tuple(b_idx))

                a_val = fx.Float32(fx.ptr_load(a_ptr))
                b_val = fx.Float32(fx.ptr_load(b_ptr))
                accs[i] = accs[i] + a_val * b_val
        results = yield list(accs)

    for i in fx.range_constexpr(OUT_PER_LANE):
        out_idx = lane_base + i
        row = out_idx // N
        col = out_idx % N
        c_idx = row * N + col
        c_ptr = fx.add_offset(c_base, fx.make_int_tuple(c_idx))
        fx.ptr_store(results[i], c_ptr)


@flyc.jit
def bf16_rowmajor_matmul(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    pitch_a_elems: fx.Int32,
    pitch_b_elems: fx.Int32,
    pitch_a_bytes: fx.Int32,
    pitch_b_bytes: fx.Int32,
    k_runtime: fx.Int32,
    stream: fx.Stream = fx.Stream(None),
):
    bf16_rowmajor_matmul_kernel(
        A, B, C, pitch_a_elems, pitch_b_elems, pitch_a_bytes, pitch_b_bytes, k_runtime
    ).launch(grid=(1, 1, 1), block=(64, 1, 1), smem=SMEM_BYTES, stream=stream)


def _configure_iluvatar_env() -> None:
    os.environ.setdefault("FLYDSL_COMPILE_BACKEND", "iluvatar")
    os.environ.setdefault("FLYDSL_RUNTIME_KIND", "iluvatar")
    os.environ.setdefault("ARCH", "ivcore11")
    os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")


if __name__ == "__main__":
    _configure_iluvatar_env()
    torch.manual_seed(0)

    A = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    B = torch.randn((K, N), dtype=torch.bfloat16, device="cuda")
    C = torch.zeros((M, N), dtype=torch.float32, device="cuda")

    pitch_a_elems = int(A.stride(0))
    pitch_b_elems = int(B.stride(0))
    pitch_a_bytes = pitch_a_elems * 2
    pitch_b_bytes = pitch_b_elems * 2
    assert pitch_a_bytes % 64 == 0, f"A pitch must be 64B aligned, got {pitch_a_bytes}"
    assert pitch_b_bytes % 64 == 0, f"B pitch must be 64B aligned, got {pitch_b_bytes}"
    assert K % TILE_K == 0, f"K must be multiple of TILE_K ({TILE_K})"

    stream = torch.cuda.Stream()
    bf16_rowmajor_matmul(
        A,
        B,
        C,
        pitch_a_elems=pitch_a_elems,
        pitch_b_elems=pitch_b_elems,
        pitch_a_bytes=pitch_a_bytes,
        pitch_b_bytes=pitch_b_bytes,
        k_runtime=K,
        stream=stream,
    )
    torch.cuda.synchronize()

    ref = torch.matmul(A.float(), B.float())
    max_abs_diff = (C - ref).abs().max().item()
    ok = torch.allclose(C, ref, rtol=1e-2, atol=1e-2)

    print("Result correct:", ok)
    print("max_abs_diff:", max_abs_diff)
    if not ok:
        print("C[:4, :8]:\n", C[:4, :8].cpu())
        print("Ref[:4, :8]:\n", ref[:4, :8].cpu())

