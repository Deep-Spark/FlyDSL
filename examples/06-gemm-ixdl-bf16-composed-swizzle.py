# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""bf16 IXDL GEMM using UniversalCopy plus composed-layout shared swizzle.

This is the cleaned-up non-SME version of the software-swizzle GEMM path:

* no SME copy atom or SME layout is used;
* global -> shared uses ordinary ``UniversalCopy32b``;
* shared memory uses ``make_composed_layout(S<2,1,5>, ordered_layout)``;
* shared -> register uses ``UniversalCopy16b`` for A and ``UniversalCopy32b``
  for B, matching the actual MMAD S2R contiguity;
* one CTA computes a 256x256 tile by assigning a 64x64 tile to each warp.

The default benchmark shape is 4096^3. The correctness check defaults to a
smaller shape to avoid spending most of the run in the reference matmul.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

os.environ.setdefault("FLYDSL_COMPILE_BACKEND", "ixdl")
os.environ.setdefault("FLYDSL_RUNTIME_KIND", "ixdl")

import torch  # noqa: E402

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402

ATOM_M = 16
ATOM_N = 16
ATOM_K = 16
SMEM_ROWS = 16
SMEM_BF16_PER_ROW = 32
WARP_SIZE = 64

WARP_ATOMS_M = 4
WARP_ATOMS_N = 4
DEFAULT_WARPS_M = 4
DEFAULT_WARPS_N = 4
DEFAULT_K_REP = 2
DEFAULT_SWIZZLE = (2, 1, 5)
# DEFAULT_SWIZZLE = (0, 0, 0)


def _warn_if_card_busy() -> None:
    if shutil.which("ixsmi") is None:
        return
    try:
        out = subprocess.check_output(
            ["ixsmi"], text=True, timeout=5, stderr=subprocess.DEVNULL
        )
    except Exception:
        return
    for line in out.splitlines():
        ls = line.strip()
        if ls.startswith("|") and ("MiB" in ls) and any(c.isdigit() for c in ls):
            if "python" in ls or ("MiB /" in ls and "0MiB" not in ls.split("/")[0]):
                print(
                    "[WARN] ixsmi suggests this GPU may already be busy.",
                    file=sys.stderr,
                )
                return


def _reference(A, B):
    return A.to(torch.float32) @ B.to(torch.float32).T


def _make_g2s_tiled_copy(copy_atom):
    # 64 lanes cover a 16x32 bf16 brick; each lane copies 8 bf16 values.
    thr_layout = fx.make_layout((16, 4), (1, 16))
    val_layout = fx.make_layout((1, 8), (1, 1))
    tile_mn, layout_tv = fx.make_layout_tv(thr_layout, val_layout)
    return fx.make_tiled_copy(copy_atom, layout_tv, tile_mn)


def _build_kernel(M, N, K, warps_m, warps_n, k_rep, swizzle_params):
    WARP_M = ATOM_M * WARP_ATOMS_M
    WARP_N = ATOM_N * WARP_ATOMS_N
    BM = WARP_M * warps_m
    BN = WARP_N * warps_n
    BK = ATOM_K * k_rep
    num_warps = warps_m * warps_n
    threads = num_warps * WARP_SIZE

    assert K % BK == 0
    assert M % BM == 0 and N % BN == 0
    assert BK % SMEM_BF16_PER_ROW == 0

    cta_atoms_m = BM // SMEM_ROWS
    cta_atoms_n = BN // SMEM_ROWS
    cta_atoms_k = BK // SMEM_BF16_PER_ROW

    a_atoms_total = cta_atoms_m * cta_atoms_k
    b_atoms_total = cta_atoms_n * cta_atoms_k
    a_per_warp = a_atoms_total // num_warps
    b_per_warp = b_atoms_total // num_warps
    assert a_atoms_total % num_warps == 0
    assert b_atoms_total % num_warps == 0

    brick_elems = SMEM_ROWS * SMEM_BF16_PER_ROW
    stage_elems = (BM + BN) * BK

    @flyc.kernel
    def gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x
        bid_x, bid_y, _ = fx.block_idx
        warp_id = tid // WARP_SIZE
        lane_id = tid % WARP_SIZE
        warp_m = warp_id // warps_n
        warp_n = warp_id % warps_n

        gA = fx.slice(fx.flat_divide(A, (BM, BK)), (None, None, bid_x, None))
        gB = fx.slice(fx.flat_divide(B, (BN, BK)), (None, None, bid_y, None))
        gC = fx.slice(fx.flat_divide(C, (BM, BN)), (None, None, bid_x, bid_y))

        smem_ptr = fx.get_dyn_shared()
        smem_bf16 = fx.recast_iter(
            fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Shared, 128),
            smem_ptr,
        )

        mma_atom = fx.make_mma_atom(
            fx.ixdl.MMAD(ATOM_M, ATOM_N, ATOM_K, fx.BFloat16)
        )
        tiled_mma = fx.make_tiled_mma(
            mma_atom, fx.make_layout((1, 1, 1), (1, 1, 1))
        )
        thr_mma = tiled_mma.thr_slice(lane_id)

        copy_atom_g2s = fx.make_copy_atom(fx.UniversalCopy32b(), fx.BFloat16)
        copy_atom_a = fx.make_copy_atom(fx.UniversalCopy16b(), fx.BFloat16)
        copy_atom_b = fx.make_copy_atom(fx.UniversalCopy32b(), fx.BFloat16)
        copy_atom_c = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)

        tiled_copy_A = fx.make_tiled_copy_A(copy_atom_a, tiled_mma)
        tiled_copy_B = fx.make_tiled_copy_B(copy_atom_b, tiled_mma)
        tiled_copy_C = fx.make_tiled_copy_C(copy_atom_c, tiled_mma)
        tiled_copy_g2s = _make_g2s_tiled_copy(copy_atom_g2s)
        thr_copy_A = tiled_copy_A.get_slice(lane_id)
        thr_copy_B = tiled_copy_B.get_slice(lane_id)
        thr_copy_C = tiled_copy_C.get_slice(lane_id)
        thr_copy_g2s = tiled_copy_g2s.get_slice(lane_id)

        tile_smem = fx.make_tile(SMEM_ROWS, SMEM_BF16_PER_ROW)
        tile_atom_A = fx.make_tile(ATOM_M, ATOM_K)
        tile_atom_B = fx.make_tile(ATOM_N, ATOM_K)

        swizzle = fx.static(fx.SwizzleType.get(*swizzle_params))
        composed_tile_layout = fx.make_composed_layout(
            swizzle,
            fx.make_ordered_layout((SMEM_ROWS, SMEM_BF16_PER_ROW), (1, 0)),
        )

        def _shared_tile_view_dyn(elem_offset):
            ptr = fx.add_offset(smem_bf16, fx.make_int_tuple(elem_offset))
            return fx.make_view(ptr, composed_tile_layout)

        gC_warp = fx.slice(
            fx.flat_divide(gC, (WARP_M, WARP_N)), (None, None, warp_m, warp_n)
        )
        gC_atoms = fx.flat_divide(gC_warp, (ATOM_M, ATOM_N))

        accs = []
        for im in fx.range_constexpr(WARP_ATOMS_M):
            row = []
            for jn in fx.range_constexpr(WARP_ATOMS_N):
                c_tile = fx.slice(gC_atoms, (None, None, im, jn))
                frag = thr_mma.make_fragment_C(c_tile)
                frag.fill(0)
                row.append(frag)
            accs.append(row)

        warp_a_start = warp_id * a_per_warp
        warp_b_start = warp_id * b_per_warp
        warp_a_base = warp_m * fx.Int32(WARP_ATOMS_M * cta_atoms_k * brick_elems)
        warp_b_base = warp_n * fx.Int32(WARP_ATOMS_N * cta_atoms_k * brick_elems)

        g2s_a_tiles = []
        for t in fx.range_constexpr(a_per_warp):
            atom_idx = warp_a_start + t
            g2s_a_tiles.append(_shared_tile_view_dyn(fx.Int32(atom_idx * brick_elems)))

        g2s_b_tiles = []
        for t in fx.range_constexpr(b_per_warp):
            atom_idx = warp_b_start + t
            g2s_b_tiles.append(
                _shared_tile_view_dyn(fx.Int32(BM * BK + atom_idx * brick_elems))
            )

        def _make_warp_atoms():
            a_atoms = []
            b_atoms = []
            for im in fx.range_constexpr(WARP_ATOMS_M):
                row = []
                for ki in fx.range_constexpr(cta_atoms_k):
                    off = warp_a_base + fx.Int32((im * cta_atoms_k + ki) * brick_elems)
                    row.append(
                        fx.zipped_divide(_shared_tile_view_dyn(off), tile_atom_A)
                    )
                a_atoms.append(row)
            for jn in fx.range_constexpr(WARP_ATOMS_N):
                row = []
                for ki in fx.range_constexpr(cta_atoms_k):
                    off = (
                        warp_b_base
                        + fx.Int32(BM * BK + (jn * cta_atoms_k + ki) * brick_elems)
                    )
                    row.append(
                        fx.zipped_divide(_shared_tile_view_dyn(off), tile_atom_B)
                    )
                b_atoms.append(row)
            return a_atoms, b_atoms

        warp_a_atoms, warp_b_atoms = _make_warp_atoms()

        def issue_stage(k):
            k_A = gA[None, None, k]
            k_B = gB[None, None, k]
            g_A_div = fx.zipped_divide(k_A, tile_smem)
            g_B_div = fx.zipped_divide(k_B, tile_smem)

            for t in fx.range_constexpr(a_per_warp):
                atom_idx = warp_a_start + t
                mi = atom_idx // cta_atoms_k
                ki = atom_idx % cta_atoms_k
                g_tile = fx.slice(g_A_div, (None, (mi, ki)))
                fx.copy(
                    copy_atom_g2s,
                    thr_copy_g2s.partition_S(g_tile),
                    thr_copy_g2s.partition_D(g2s_a_tiles[t]),
                    pred=None,
                )

            for t in fx.range_constexpr(b_per_warp):
                atom_idx = warp_b_start + t
                ni = atom_idx // cta_atoms_k
                ki = atom_idx % cta_atoms_k
                g_tile = fx.slice(g_B_div, (None, (ni, ki)))
                fx.copy(
                    copy_atom_g2s,
                    thr_copy_g2s.partition_S(g_tile),
                    thr_copy_g2s.partition_D(g2s_b_tiles[t]),
                    pred=None,
                )

        def compute_stage(sA_stg, sB_stg):
            for kk in fx.range_constexpr(k_rep):
                ki = kk // 2
                kk_in_tile = kk % 2

                a_frags = []
                for im in fx.range_constexpr(WARP_ATOMS_M):
                    a_tile = fx.slice(sA_stg[im][ki], (None, kk_in_tile))
                    frag_A = thr_mma.make_fragment_A(a_tile)
                    fx.copy(
                        copy_atom_a,
                        thr_copy_A.partition_S(a_tile),
                        thr_copy_A.retile(frag_A),
                        pred=None,
                    )
                    a_frags.append(frag_A)

                b_frags = []
                for jn in fx.range_constexpr(WARP_ATOMS_N):
                    b_tile = fx.slice(sB_stg[jn][ki], (None, kk_in_tile))
                    frag_B = thr_mma.make_fragment_B(b_tile)
                    fx.copy(
                        copy_atom_b,
                        thr_copy_B.partition_S(b_tile),
                        thr_copy_B.retile(frag_B),
                        pred=None,
                    )
                    b_frags.append(frag_B)

                for jn in fx.range_constexpr(WARP_ATOMS_N):
                    for im in fx.range_constexpr(WARP_ATOMS_M):
                        fx.gemm(
                            mma_atom,
                            accs[im][jn],
                            a_frags[im],
                            b_frags[jn],
                            accs[im][jn],
                        )

        def store_acc_values(acc_vals):
            idx = 0
            for im in fx.range_constexpr(WARP_ATOMS_M):
                for jn in fx.range_constexpr(WARP_ATOMS_N):
                    accs[im][jn].store(acc_vals[idx])
                    idx += 1

        def load_acc_values():
            vals = []
            for im in fx.range_constexpr(WARP_ATOMS_M):
                for jn in fx.range_constexpr(WARP_ATOMS_N):
                    vals.append(accs[im][jn].load())
            return vals

        k_tiles = K // BK
        init_state = load_acc_values()
        results = init_state
        for k, state in range(0, k_tiles, 1, init=init_state):
            store_acc_values(list(state))
            issue_stage(fx.Int32(k))
            fx.gpu.barrier()
            compute_stage(warp_a_atoms, warp_b_atoms)
            fx.gpu.barrier()
            results = yield load_acc_values()

        store_acc_values(list(results))

        for im in fx.range_constexpr(WARP_ATOMS_M):
            for jn in fx.range_constexpr(WARP_ATOMS_N):
                c_tile = fx.slice(gC_atoms, (None, None, im, jn))
                fx.copy(
                    copy_atom_c,
                    thr_copy_C.retile(accs[im][jn]),
                    thr_copy_C.partition_S(c_tile),
                    pred=None,
                )

    return gemm_kernel, threads, stage_elems


def _build_launcher(M, N, K, warps_m, warps_n, k_rep, swizzle_params):
    gemm_kernel, threads, stage_elems = _build_kernel(
        M, N, K, warps_m, warps_n, k_rep, swizzle_params
    )
    WARP_M = ATOM_M * WARP_ATOMS_M
    WARP_N = ATOM_N * WARP_ATOMS_N
    BM = WARP_M * warps_m
    BN = WARP_N * warps_n
    grid = (M // BM, N // BN, 1)
    block = (threads, 1, 1)
    smem_bytes = stage_elems * 2

    @flyc.jit
    def gemm(A, B, C, stream=fx.Stream(None)):
        gemm_kernel(A, B, C).launch(
            grid=grid, block=block, smem=smem_bytes, stream=stream
        )

    return gemm, (grid, block, smem_bytes)


def _check(M, N, K, warps_m, warps_n, k_rep, swizzle_params):
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.bfloat16).cuda()
    B = torch.randn(N, K, dtype=torch.bfloat16).cuda()
    C = torch.zeros(M, N, dtype=torch.float32).cuda()
    launcher, (grid, block, smem) = _build_launcher(
        M, N, K, warps_m, warps_n, k_rep, swizzle_params
    )
    stream = torch.cuda.Stream()
    launcher(A, B, C, stream=stream)
    torch.cuda.synchronize()

    expected = _reference(A, B)
    diff = (C - expected).abs()
    atol = 2e-2 * max(1.0, (K / 16) ** 0.5)
    ok = torch.allclose(C, expected, atol=atol, rtol=2e-2)
    finite_ok = torch.isfinite(C).all().item()
    print(
        f"[check] M={M} N={N} K={K} swizzle=S<{swizzle_params[0]},"
        f"{swizzle_params[1]},{swizzle_params[2]}> "
        f"warps=({warps_m}x{warps_n}) k_rep={k_rep} "
        f"grid={grid} block={block} smem={smem} "
        f"ok={ok} finite={finite_ok} max_abs={diff.max().item():.3e} "
        f"mean_abs={diff.mean().item():.3e} atol={atol:.2e}"
    )
    return bool(ok and finite_ok)


def _bench(M, N, K, warps_m, warps_n, k_rep, swizzle_params, iters, warmup):
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.bfloat16).cuda()
    B = torch.randn(N, K, dtype=torch.bfloat16).cuda()
    C = torch.zeros(M, N, dtype=torch.float32).cuda()
    launcher, _ = _build_launcher(M, N, K, warps_m, warps_n, k_rep, swizzle_params)
    stream = torch.cuda.Stream()

    t0 = time.perf_counter()
    compiled = flyc.compile(launcher, A, B, C, fx.Stream(stream))
    torch.cuda.synchronize()
    print(f"[compile] {1e3 * (time.perf_counter() - t0):.1f} ms")

    for _ in range(warmup):
        compiled(A, B, C, fx.Stream(stream))
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record()
        for _ in range(iters):
            compiled(A, B, C, fx.Stream(stream))
        end.record()
    torch.cuda.synchronize()

    us = start.elapsed_time(end) * 1e3 / iters
    tflops = (2.0 * M * N * K) / (us * 1e-6) / 1e12
    print(f"[bench] {us:.1f} us/iter  {tflops:.2f} TFLOPS")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", nargs=3, type=int, default=[4096, 4096, 4096])
    p.add_argument("--check-shape", nargs=3, type=int, default=[256, 256, 64])
    p.add_argument("--warps-m", type=int, default=DEFAULT_WARPS_M)
    p.add_argument("--warps-n", type=int, default=DEFAULT_WARPS_N)
    p.add_argument("--k-rep", type=int, default=DEFAULT_K_REP)
    p.add_argument(
        "--swizzle",
        nargs=3,
        type=int,
        metavar=("MASK", "BASE", "SHIFT"),
        default=list(DEFAULT_SWIZZLE),
    )
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--check-only", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    _warn_if_card_busy()
    args = _parse_args()
    swizzle_params = tuple(args.swizzle)
    cm, cn, ck = args.check_shape
    ok = _check(
        cm,
        cn,
        ck,
        args.warps_m,
        args.warps_n,
        args.k_rep,
        swizzle_params,
    )
    if not ok:
        sys.exit(1)
    if args.check_only:
        sys.exit(0)

    M, N, K = args.shape
    _bench(
        M,
        N,
        K,
        args.warps_m,
        args.warps_n,
        args.k_rep,
        swizzle_params,
        args.iters,
        args.warmup,
    )
