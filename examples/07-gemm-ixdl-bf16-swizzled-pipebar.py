"""bf16 GEMM on MR-V100 with swizzled SME and ixmma-style pipebar sync.

This uses swizzled SME global -> shared staging plus per-tile MMA-aware LDS
reads. Stage synchronization uses the lighter IXDL primitives

* ``ixdl.sl_waitcnt``
* ``ixdl.pipebar.req``
* ``ixdl.pipebar.wait``

following the same ``SyncWait -> DMA(next)+MMA(current) -> SyncArrive``
structure used by ixmma on IVCore11.
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
SME_ROWS = 16
SME_BF16_PER_ROW = 32
WARP_SIZE = 64
STAGES = 2


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


def _sme_view_dyn(base_ptr, elem_type, elem_offset, transpose=False):
    elem_ir_type = elem_type.ir_type if hasattr(elem_type, "ir_type") else elem_type
    smem_ptr = fx.recast_iter(
        fx.PointerType.get(elem_ir_type, fx.AddressSpace.Shared),
        base_ptr,
    )
    smem_ptr = fx.add_offset(smem_ptr, fx.make_int_tuple(elem_offset))
    return fx.make_view(
        smem_ptr,
        fx.ixdl.SMELayout16x512b(elem_ir_type, transpose=transpose),
    )


def _build_kernel(M, N, K, warps_m, warps_n, k_rep, copy_bits=32):
    WARP_ATOMS_M = 4
    WARP_ATOMS_N = 4
    WARP_M = ATOM_M * WARP_ATOMS_M
    WARP_N = ATOM_N * WARP_ATOMS_N
    BM = WARP_M * warps_m
    BN = WARP_N * warps_n
    BK = ATOM_K * k_rep
    num_warps = warps_m * warps_n
    threads = num_warps * WARP_SIZE

    assert K % BK == 0
    assert M % BM == 0 and N % BN == 0
    assert BK % SME_BF16_PER_ROW == 0

    cta_atoms_m = BM // SME_ROWS
    cta_atoms_n = BN // SME_ROWS
    cta_atoms_k = BK // SME_BF16_PER_ROW

    stride_byte_A = K * 2
    stride_byte_B = K * 2

    a_atoms_total = cta_atoms_m * cta_atoms_k
    b_atoms_total = cta_atoms_n * cta_atoms_k
    a_per_warp = a_atoms_total // num_warps
    b_per_warp = b_atoms_total // num_warps
    assert a_atoms_total % num_warps == 0
    assert b_atoms_total % num_warps == 0

    brick_elems = SME_ROWS * SME_BF16_PER_ROW
    stage_elems = (BM + BN) * BK

    @flyc.kernel
    def gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x
        bid_x, bid_y, _ = fx.block_idx
        warp_id = tid // WARP_SIZE
        lane_id = tid % WARP_SIZE
        warp_m = warp_id // warps_n
        warp_n = warp_id % warps_n

        gA = fx.slice(
            fx.flat_divide(A, (BM, BK)),
            (None, None, bid_x, None),
        )
        gB = fx.slice(
            fx.flat_divide(B, (BN, BK)),
            (None, None, bid_y, None),
        )
        gC = fx.slice(
            fx.flat_divide(C, (BM, BN)),
            (None, None, bid_x, bid_y),
        )

        smem_ptr = fx.get_dyn_shared()

        mma_atom = fx.make_mma_atom(
            fx.ixdl.MMAD(ATOM_M, ATOM_N, ATOM_K, fx.BFloat16)
        )
        tiled_mma = fx.make_tiled_mma(
            mma_atom, fx.make_layout((1, 1, 1), (1, 1, 1))
        )
        thr_mma = tiled_mma.thr_slice(lane_id)

        sme_atom_A = fx.make_copy_atom(
            fx.ixdl.SMECopy(
                fx.BFloat16,
                (SME_ROWS, SME_BF16_PER_ROW),
                stride_byte=stride_byte_A,
                major="k",
                cache_op="cache_all",
                swizzle="row_xfb16",
            ),
            fx.BFloat16,
        )
        sme_atom_B = fx.make_copy_atom(
            fx.ixdl.SMECopy(
                fx.BFloat16,
                (SME_ROWS, SME_BF16_PER_ROW),
                stride_byte=stride_byte_B,
                major="mn",
                cache_op="cache_all",
                swizzle="col_xfb8",
            ),
            fx.BFloat16,
        )

        if copy_bits == 32:
            s2r_copy = fx.UniversalCopy32b()
        elif copy_bits == 64:
            s2r_copy = fx.UniversalCopy64b()
        elif copy_bits == 128:
            s2r_copy = fx.UniversalCopy128b()
        else:
            raise ValueError(f"unsupported copy_bits={copy_bits}")

        copy_atom_s2r = fx.make_copy_atom(s2r_copy, fx.BFloat16)
        copy_atom_r2g_c = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)

        tiled_copy_A = fx.make_tiled_copy_A(copy_atom_s2r, tiled_mma)
        tiled_copy_B = fx.make_tiled_copy_B(copy_atom_s2r, tiled_mma)
        tiled_copy_C = fx.make_tiled_copy_C(copy_atom_r2g_c, tiled_mma)
        thr_copy_A = tiled_copy_A.get_slice(lane_id)
        thr_copy_B = tiled_copy_B.get_slice(lane_id)
        thr_copy_C = tiled_copy_C.get_slice(lane_id)

        tile_sme = fx.make_tile(SME_ROWS, SME_BF16_PER_ROW)
        tile_atom_A = fx.make_tile(ATOM_M, ATOM_K)
        tile_atom_B = fx.make_tile(ATOM_N, ATOM_K)

        gC_warp = fx.slice(
            fx.flat_divide(gC, (WARP_M, WARP_N)),
            (None, None, warp_m, warp_n),
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

        def _make_warp_atoms(stage_base):
            a_atoms = []
            b_atoms = []
            for im in fx.range_constexpr(WARP_ATOMS_M):
                row = []
                for ki in fx.range_constexpr(cta_atoms_k):
                    off = (
                        warp_a_base
                        + fx.Int32(stage_base + (im * cta_atoms_k + ki) * brick_elems)
                    )
                    row.append(
                        fx.zipped_divide(
                            _sme_view_dyn(smem_ptr, fx.BFloat16, off, transpose=False),
                            tile_atom_A,
                        )
                    )
                a_atoms.append(row)
            for jn in fx.range_constexpr(WARP_ATOMS_N):
                row = []
                for ki in fx.range_constexpr(cta_atoms_k):
                    off = (
                        warp_b_base
                        + fx.Int32(
                            stage_base + BM * BK + (jn * cta_atoms_k + ki) * brick_elems
                        )
                    )
                    row.append(
                        fx.zipped_divide(
                            _sme_view_dyn(smem_ptr, fx.BFloat16, off, transpose=True),
                            tile_atom_B,
                        )
                    )
                b_atoms.append(row)
            return a_atoms, b_atoms

        warp_a0_atoms, warp_b0_atoms = _make_warp_atoms(0)
        warp_a1_atoms, warp_b1_atoms = _make_warp_atoms(stage_elems)

        def _sync_arrive():
            fx.ixdl.sl_waitcnt(g2s=True, g2s_cnt=0, lm=True, lm_cnt=0)
            fx.ixdl.pipebar_req(0)

        def _sync_wait():
            fx.ixdl.pipebar_wait(0)

        def issue_stage_swizzled(k, stage_base):
            k_A = gA[None, None, k]
            k_B = gB[None, None, k]
            g_A_div = fx.zipped_divide(k_A, tile_sme)
            g_B_div = fx.zipped_divide(k_B, tile_sme)
            for t in fx.range_constexpr(a_per_warp):
                atom_idx = warp_a_start + t
                mi = atom_idx // cta_atoms_k
                ki = atom_idx % cta_atoms_k
                a_off = fx.Int32(stage_base) + atom_idx * fx.Int32(brick_elems)
                fx.copy_atom_call(
                    sme_atom_A,
                    fx.slice(g_A_div, (None, (mi, ki))),
                    _sme_view_dyn(smem_ptr, fx.BFloat16, a_off, transpose=False),
                )
            for t in fx.range_constexpr(b_per_warp):
                atom_idx = warp_b_start + t
                ni = atom_idx // cta_atoms_k
                ki = atom_idx % cta_atoms_k
                b_off = (
                    fx.Int32(stage_base + BM * BK)
                    + atom_idx * fx.Int32(brick_elems)
                )
                fx.copy_atom_call(
                    sme_atom_B,
                    fx.slice(g_B_div, (None, (ni, ki))),
                    _sme_view_dyn(smem_ptr, fx.BFloat16, b_off, transpose=True),
                )
            fx.ixdl.cp_async_commit_group()

        def compute_stage(sA_stg, sB_stg):
            for kk in fx.range_constexpr(k_rep):
                a_frags = []
                ki = kk // 2
                kk_in_tile = kk % 2
                for im in fx.range_constexpr(WARP_ATOMS_M):
                    a_tile = fx.slice(sA_stg[im][ki], (None, kk_in_tile))
                    frag_A = thr_mma.make_fragment_A(a_tile)
                    fx.copy(
                        copy_atom_s2r,
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
                        copy_atom_s2r,
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

        k_tiles = K // BK
        issue_stage_swizzled(0, 0)
        _sync_arrive()

        for k in fx.range_constexpr(k_tiles - 1):
            stage_now = k % STAGES
            stage_next = (k + 1) % STAGES

            sA_now = warp_a0_atoms if stage_now == 0 else warp_a1_atoms
            sB_now = warp_b0_atoms if stage_now == 0 else warp_b1_atoms
            stage_base = 0 if stage_next == 0 else stage_elems

            _sync_wait()
            issue_stage_swizzled(k + 1, stage_base)
            compute_stage(sA_now, sB_now)
            _sync_arrive()

        _sync_wait()
        sA_last = warp_a0_atoms if (k_tiles - 1) % STAGES == 0 else warp_a1_atoms
        sB_last = warp_b0_atoms if (k_tiles - 1) % STAGES == 0 else warp_b1_atoms
        compute_stage(sA_last, sB_last)

        for im in fx.range_constexpr(WARP_ATOMS_M):
            for jn in fx.range_constexpr(WARP_ATOMS_N):
                c_tile = fx.slice(gC_atoms, (None, None, im, jn))
                fx.copy(
                    copy_atom_r2g_c,
                    thr_copy_C.retile(accs[im][jn]),
                    thr_copy_C.partition_S(c_tile),
                    pred=None,
                )

    return gemm_kernel, threads


def _build_launcher(M, N, K, warps_m, warps_n, k_rep, copy_bits=32):
    gemm_kernel, threads = _build_kernel(
        M, N, K, warps_m, warps_n, k_rep, copy_bits
    )
    WARP_M = ATOM_M * 4
    WARP_N = ATOM_N * 4
    BM = WARP_M * warps_m
    BN = WARP_N * warps_n
    BK = ATOM_K * k_rep
    grid = (M // BM, N // BN, 1)
    block = (threads, 1, 1)
    smem_bytes = STAGES * (BM + BN) * BK * 2

    @flyc.jit
    def gemm(A, B, C, stream=fx.Stream(None)):
        gemm_kernel(A, B, C).launch(
            grid=grid, block=block, smem=smem_bytes, stream=stream
        )

    return gemm, (grid, block, smem_bytes)


def _structural_check(M, N, K, warps_m, warps_n, k_rep, copy_bits=32):
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.bfloat16).cuda()
    B = torch.randn(N, K, dtype=torch.bfloat16).cuda()
    C = torch.zeros(M, N, dtype=torch.float32).cuda()
    launcher, (grid, block, smem) = _build_launcher(
        M, N, K, warps_m, warps_n, k_rep, copy_bits
    )
    stream = torch.cuda.Stream()
    launcher(A, B, C, stream=stream)
    torch.cuda.synchronize()

    expected = _reference(A, B)
    diff = (C - expected).abs()
    finite_ok = torch.isfinite(C).all().item()
    max_abs = diff.max().item()
    print(
        f"[struct] M={M} N={N} K={K}  warps=({warps_m}x{warps_n}) "
        f"k_rep={k_rep} copy_bits={copy_bits}  "
        f"grid={grid} block={block} smem={smem}  "
        f"finite={finite_ok}  max_abs={max_abs:.3e}"
    )
    atol = 2e-2 * max(1.0, (K / 16) ** 0.5)
    ok = torch.allclose(C, expected, atol=atol, rtol=2e-2)
    print(f"[struct] numerical check: ok={ok}  atol={atol:.2e}")
    return bool(ok and finite_ok)


def _bench(M, N, K, warps_m, warps_n, k_rep, iters, warmup, copy_bits=32):
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.bfloat16).cuda()
    B = torch.randn(N, K, dtype=torch.bfloat16).cuda()
    C = torch.zeros(M, N, dtype=torch.float32).cuda()

    launcher, _ = _build_launcher(M, N, K, warps_m, warps_n, k_rep, copy_bits)
    stream = torch.cuda.Stream()

    t0 = time.perf_counter()
    compiled = flyc.compile(launcher, A, B, C, fx.Stream(stream))
    torch.cuda.synchronize()
    compile_s = time.perf_counter() - t0
    print(
        f"[compile] pipebar  warps=({warps_m}x{warps_n}) "
        f"k_rep={k_rep} copy_bits={copy_bits}  {compile_s*1e3:.1f} ms"
    )

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
    total_ms = start.elapsed_time(end)
    per_iter_us = total_ms * 1e3 / iters

    flops = 2.0 * M * N * K
    tflops = flops / (per_iter_us * 1e-6) / 1e12
    print(
        f"[bench]   pipebar  M={M} N={N} K={K}  "
        f"warps=({warps_m}x{warps_n}) k_rep={k_rep} copy_bits={copy_bits}  iters={iters}  "
        f"{per_iter_us:.1f} us/iter  {tflops:.2f} TFLOPS"
    )
    return per_iter_us, tflops


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", nargs=3, type=int, metavar=("M", "N", "K"),
                   default=[4096, 4096, 4096])
    p.add_argument("--check-shape", nargs=3, type=int, metavar=("M", "N", "K"),
                   default=[256, 256, 128])
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--warps", nargs=2, type=int, metavar=("WM", "WN"),
                   default=[4, 4])
    p.add_argument("--k-rep", type=int, default=2, choices=[2, 4, 8])
    p.add_argument("--copy-bits", type=int, default=32, choices=[32, 64, 128])
    p.add_argument("--check-only", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    _warn_if_card_busy()
    args = _parse_args()
    M, N, K = args.shape
    warps_m, warps_n = args.warps
    cm, cn, ck = args.check_shape

    _structural_check(cm, cn, ck, warps_m, warps_n, args.k_rep, args.copy_bits)
    if args.check_only:
        sys.exit(0)

    _bench(M, N, K, warps_m, warps_n, args.k_rep, args.iters, args.warmup,
           args.copy_bits)
