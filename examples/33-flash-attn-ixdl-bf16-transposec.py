"""FlashAttention on IXDL – MR-style TransposeCToB16 epilogue.

Based on example 32 with the epilogue replaced to match ixattention-backend
MR kernel's approach (VHdBlockSize > 32, i.e. HEAD_DIM=128):
  - TransposeCToB16: warp-private 512 i32 words (2KB/warp) in SMEM.
    Processes 4 head_dim atoms at once. No shuffle, uses byte_perm-equivalent
    (shift+mask) to recombine bf16 halves after transpose.
  - Output via i32 (bf16x2-packed) global stores. No cross-warp barrier.

PV MMAD unchanged from example 32: V=A-op, P=B-op → O^T[head_dim, seq_q].

C-accumulator layout (IXCore 16x16 MMAD, 32-bit):
  M = lane/16 + v*4  (head_dim_local for O^T)
  N = lane%16         (seq_q_local for O^T)
"""

import argparse
import math
import os
import sys
import time

os.environ.setdefault("FLYDSL_COMPILE_BACKEND", "ixdl")
os.environ.setdefault("FLYDSL_RUNTIME_KIND", "ixdl")

import torch  # noqa: E402

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from flydsl.expr import arith, gpu, vector  # noqa: E402
from flydsl.expr.typing import BFloat16, T, Vector as Vec  # noqa: E402
from flydsl.expr.utils.arith import _to_raw  # noqa: E402
from flydsl._mlir.dialects.arith import (  # noqa: E402
    TruncFOp, BitcastOp, ExtUIOp, ShLIOp, OrIOp, AndIOp,
    ShRUIOp, SelectOp, CmpIOp, CmpIPredicate,
)
from flydsl._mlir.dialects import scf, math as _math_dialect  # noqa: E402
from flydsl._mlir import ir  # noqa: E402

ATOM_M = 16
ATOM_N = 16
ATOM_K = 16
SME_ROWS = 16
SME_BF16_PER_ROW = 32
WARP_SIZE = 64
HEAD_DIM = 128
FRAG_ELEMS = 4
BRICK_ELEMS = SME_ROWS * SME_BF16_PER_ROW  # 512
_LOG2E = 1.4426950408889634


def _sme_view_dyn(base_ptr, elem_type, elem_offset, transpose=False):
    elem_ir_type = elem_type.ir_type if hasattr(elem_type, "ir_type") else elem_type
    smem_ptr = fx.recast_iter(
        fx.PointerType.get(elem_ir_type, fx.AddressSpace.Shared), base_ptr)
    smem_ptr = fx.add_offset(smem_ptr, fx.make_int_tuple(elem_offset))
    return fx.make_view(
        smem_ptr, fx.ixdl.SMELayout16x512b(elem_ir_type, transpose=transpose))


def _reference(q, k, v):
    _, hq, _, dim = q.shape
    hkv = k.shape[1]
    repeat = hq // hkv
    k_rep = k.repeat_interleave(repeat, dim=1)
    v_rep = v.repeat_interleave(repeat, dim=1)
    scores = torch.matmul(q.float(), k_rep.float().transpose(-1, -2))
    scores = scores * (1.0 / math.sqrt(dim))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v_rep.float()).to(torch.bfloat16)


def _build_kernel(batch, head_q, head_kv, seq_q, seq_k, k_rep,
                  num_warps=None):
    WARP_M = ATOM_M                                     # 16

    # Phase 3: auto-select BM=256/16-warp for long sequences
    if num_warps is None:
        num_warps = 16 if seq_q >= 256 else 8
    NUM_WARPS = num_warps
    BM = WARP_M * NUM_WARPS                              # 128 or 256
    BN = 128
    BK = ATOM_K * k_rep                                  # 128 with k_rep=8
    threads = NUM_WARPS * WARP_SIZE                       # 512 or 1024

    WARP_ATOMS_M = BN // ATOM_M                           # 8
    K_STEPS_QK = HEAD_DIM // ATOM_K                       # 8
    k_steps_pv = BN // ATOM_K                             # 8
    d_atoms = HEAD_DIM // ATOM_N                           # 8

    assert HEAD_DIM == 128
    assert seq_q % BM == 0 and seq_k % BN == 0
    assert head_q % head_kv == 0
    assert HEAD_DIM % BK == 0
    assert BK % SME_BF16_PER_ROW == 0

    repeat = head_q // head_kv
    scale_log2e = (1.0 / math.sqrt(HEAD_DIM)) * _LOG2E

    cta_atoms_k = BK // SME_BF16_PER_ROW                  # 4 (BK=128)
    cta_atoms_k_q = HEAD_DIM // SME_BF16_PER_ROW          # 4

    k_atoms_m = BN // SME_ROWS                             # 8
    k_atoms_total = k_atoms_m * cta_atoms_k                # 32 (BK=128)
    q_atoms_m = BM // SME_ROWS                             # 8 or 16
    q_atoms_total = q_atoms_m * cta_atoms_k_q              # 32 or 64
    assert k_atoms_total % NUM_WARPS == 0
    assert q_atoms_total % NUM_WARPS == 0
    k_per_warp = k_atoms_total // NUM_WARPS
    q_per_warp = q_atoms_total // NUM_WARPS

    # Phase 5: SMEM layout – V at [0, 32KB), K at [32KB, 64KB)
    v_smem_elems = HEAD_DIM * BN                           # 16384 bf16 = 32 KB
    k_smem_offset = v_smem_elems                           # 16384 bf16 = 32 KB offset
    k_stage_elems = BN * BK                                # bf16 per K stage
    k_tiles = HEAD_DIM // BK                               # 1 (BK=128) or 2 (BK=64)
    smem_bytes = (v_smem_elems + k_tiles * k_stage_elems) * 2  # 64 KB

    # V chunk bases within [0, 32KB): one base per k_tile
    v_chunk_bases = [kt * k_stage_elems for kt in range(k_tiles)]

    # Phase 2: Epilogue uses FULL SMEM (V+K both free after PV MMAD).
    # XOR swizzle on column index eliminates bank conflicts without padding.
    # TransposeCToB16 epilogue: 512 i32 words per warp (2 KB/warp), warp-private.
    total_smem_elems = v_smem_elems + k_tiles * k_stage_elems  # 32768 bf16 = 64 KB
    TC_WORDS_PER_WARP = 512
    tc_total_bytes = NUM_WARPS * TC_WORDS_PER_WARP * 4
    assert tc_total_bytes <= total_smem_elems * 2, (
        f"TransposeCToB16 needs {tc_total_bytes}B but only {total_smem_elems*2}B available")

    @flyc.kernel
    def flash_attn_opt_kernel(
        Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor,
    ):
        tid = fx.thread_idx.x
        q_tile = fx.Int32(fx.block_idx.x)
        b = fx.Int32(fx.block_idx.y)
        hq = fx.Int32(fx.block_idx.z)
        warp_id = tid // WARP_SIZE
        lane_id = tid % WARP_SIZE

        c_head_q = fx.Int32(head_q)
        c_head_kv = fx.Int32(head_kv)
        c_repeat = fx.Int32(repeat)
        hkv = hq // c_repeat

        c_q_tiles = fx.Int32(seq_q // BM)
        c_kv_tiles = fx.Int32(seq_k // BN)
        q_tile_id = (b * c_head_q + hq) * c_q_tiles + q_tile
        k_tile_base = (b * c_head_kv + hkv) * c_kv_tiles

        # ---- global views ----
        q_2d = fx.Tensor(fx.make_view(
            fx.get_iter(Q),
            fx.make_layout(
                (batch * head_q * seq_q, HEAD_DIM), (HEAD_DIM, 1))))
        k_2d = fx.Tensor(fx.make_view(
            fx.get_iter(K),
            fx.make_layout(
                (batch * head_kv * seq_k, HEAD_DIM), (HEAD_DIM, 1))))
        v_2d = fx.Tensor(fx.make_view(
            fx.get_iter(V),
            fx.make_layout(
                (batch * head_kv * seq_k, HEAD_DIM), (HEAD_DIM, 1))))

        q_tiles_bk = fx.flat_divide(q_2d, (BM, HEAD_DIM))
        k_tiles_all = fx.flat_divide(k_2d, (BN, BK))
        v_tiles_all = fx.flat_divide(v_2d, (BN, BK))

        smem_ptr = fx.get_dyn_shared()

        # ---- MMA (per-warp) ----
        mma_atom = fx.make_mma_atom(
            fx.ixdl.MMAD(ATOM_M, ATOM_N, ATOM_K, fx.BFloat16))
        tiled_mma = fx.make_tiled_mma(
            mma_atom, fx.make_layout((1, 1, 1), (1, 1, 1)))
        thr_mma = tiled_mma.thr_slice(lane_id)

        # ---- SME copy atoms ----
        sme_atom_K = fx.make_copy_atom(fx.ixdl.SMECopy(
            fx.BFloat16, (SME_ROWS, SME_BF16_PER_ROW),
            stride_byte=HEAD_DIM * 2, major="k",
            cache_op="cache_all", swizzle="row_xfb16",
        ), fx.BFloat16)
        sme_atom_col = fx.make_copy_atom(fx.ixdl.SMECopy(
            fx.BFloat16, (SME_ROWS, SME_BF16_PER_ROW),
            stride_byte=HEAD_DIM * 2, major="mn",
            cache_op="cache_all", swizzle="col_xfb8",
        ), fx.BFloat16)

        # ---- S2R copy atoms ----
        copy_atom_s2r = fx.make_copy_atom(
            fx.UniversalCopy32b(), fx.BFloat16)
        tiled_copy_A = fx.make_tiled_copy_A(copy_atom_s2r, tiled_mma)
        tiled_copy_B = fx.make_tiled_copy_B(copy_atom_s2r, tiled_mma)
        thr_copy_A = tiled_copy_A.get_slice(lane_id)
        thr_copy_B = tiled_copy_B.get_slice(lane_id)

        tile_sme = fx.make_tile(SME_ROWS, SME_BF16_PER_ROW)
        tile_atom_A = fx.make_tile(ATOM_M, ATOM_K)
        tile_atom_B = fx.make_tile(ATOM_N, ATOM_K)

        # ---- QK accumulators: 8 m-atoms (seq_k) per warp ----
        dummy_ptr = fx.recast_iter(
            fx.PointerType.get(
                fx.Float32.ir_type, fx.AddressSpace.Shared),
            smem_ptr)
        dummy_tile = fx.Tensor(fx.make_view(
            dummy_ptr, fx.make_layout((WARP_M, BN), (BN, 1))))
        dummy_atoms = fx.flat_divide(dummy_tile, (ATOM_M, ATOM_N))

        accs = []
        for jm in fx.range_constexpr(WARP_ATOMS_M):
            c_tile = fx.slice(dummy_atoms, (None, None, 0, jm))
            frag = thr_mma.make_fragment_C(c_tile)
            frag.fill(0)
            accs.append(frag)

        # ---- PV accumulators (8 output d-atoms) ----
        dummy_pv = fx.Tensor(fx.make_view(
            dummy_ptr,
            fx.make_layout((ATOM_M, ATOM_N), (ATOM_N, 1))))
        pv_accs = []
        for nt in fx.range_constexpr(d_atoms):
            frag = thr_mma.make_fragment_C(dummy_pv)
            frag.fill(0)
            pv_accs.append(frag)

        # ---- softmax state ----
        c_zero = arith.constant(0.0, type=T.f32)
        c_neg_inf = arith.constant(float("-inf"), type=T.f32)
        c_scale_log2e = arith.constant(scale_log2e, type=T.f32)
        m_running = c_neg_inf
        l_running = c_zero

        # ---- warp work distribution ----
        warp_q_start = warp_id * q_per_warp
        warp_k_start = warp_id * k_per_warp

        def _sync_arrive():
            fx.ixdl.sl_waitcnt(g2s=True, g2s_cnt=0)
            fx.ixdl.pipebar_req(0)

        def _sync_wait():
            fx.ixdl.pipebar_wait(0)

        # ---- K smem views at [32KB, 64KB) ----
        def _make_k_atoms(stage_base):
            k_atoms_s = []
            for jm in fx.range_constexpr(WARP_ATOMS_M):
                row = []
                for ki in fx.range_constexpr(cta_atoms_k):
                    off = fx.Int32(
                        k_smem_offset + stage_base
                        + (jm * cta_atoms_k + ki) * BRICK_ELEMS)
                    row.append(fx.zipped_divide(
                        _sme_view_dyn(
                            smem_ptr, fx.BFloat16, off, False),
                        tile_atom_A))
                k_atoms_s.append(row)
            return k_atoms_s

        k_atoms_list = [_make_k_atoms(kt * k_stage_elems)
                        for kt in range(k_tiles)]

        # ============================================================
        #  Phase 0: Load Q to SMEM [0, 32KB) with col_xfb8, copy to
        #           registers, then free SMEM for V
        # ============================================================
        q_full = fx.slice(
            q_tiles_bk, (None, None, q_tile_id, 0))
        q_div = fx.zipped_divide(q_full, tile_sme)
        for t in fx.range_constexpr(q_per_warp):
            atom_idx = warp_q_start + t
            mi = atom_idx // cta_atoms_k_q
            ki_q = atom_idx % cta_atoms_k_q
            q_off = atom_idx * fx.Int32(BRICK_ELEMS)
            fx.copy_atom_call(
                sme_atom_col,
                fx.slice(q_div, (None, (mi, ki_q))),
                _sme_view_dyn(
                    smem_ptr, fx.BFloat16, q_off, True))
        fx.ixdl.cp_async_commit_group()
        _sync_arrive()
        _sync_wait()

        q_reg_frags = []
        for k_step in fx.range_constexpr(K_STEPS_QK):
            ki_q = k_step // 2
            kq_sub = k_step % 2
            q_off = (warp_id * fx.Int32(cta_atoms_k_q)
                     + fx.Int32(ki_q)) * fx.Int32(BRICK_ELEMS)
            q_view = _sme_view_dyn(
                smem_ptr, fx.BFloat16, q_off, True)
            q_atoms = fx.zipped_divide(q_view, tile_atom_B)
            q_tile_s = fx.slice(q_atoms, (None, kq_sub))
            frag_Q = thr_mma.make_fragment_B(q_tile_s)
            fx.copy(copy_atom_s2r,
                    thr_copy_B.partition_S(q_tile_s),
                    thr_copy_B.retile(frag_Q), pred=None)
            q_reg_frags.append(frag_Q)

        gpu.barrier()

        # ---- output global view (i32 for bf16x2 packed stores) ----
        out_i32_ptr = fx.recast_iter(
            fx.PointerType.get(T.i32, fx.AddressSpace.Global),
            fx.get_iter(O))
        out_i32_2d = fx.Tensor(fx.make_view(
            out_i32_ptr,
            fx.make_layout(
                (batch * head_q * seq_q, HEAD_DIM // 2),
                (HEAD_DIM // 2, 1))))
        out_i32_tile = fx.slice(
            fx.flat_divide(out_i32_2d, (BM, HEAD_DIM // 2)),
            (None, None, q_tile_id, 0))

        # ---- pipeline helpers ----
        def issue_k_stage(kv_tile, k_tile_idx, stage_base):
            k_k = fx.slice(
                k_tiles_all,
                (None, None,
                 k_tile_base + fx.Int32(kv_tile), k_tile_idx))
            k_div = fx.zipped_divide(k_k, tile_sme)
            for t in fx.range_constexpr(k_per_warp):
                atom_idx = warp_k_start + t
                ni = atom_idx // cta_atoms_k
                ki = atom_idx % cta_atoms_k
                k_off = (fx.Int32(k_smem_offset + stage_base)
                         + atom_idx * fx.Int32(BRICK_ELEMS))
                fx.copy_atom_call(
                    sme_atom_K,
                    fx.slice(k_div, (None, (ni, ki))),
                    _sme_view_dyn(
                        smem_ptr, fx.BFloat16, k_off, False))
            fx.ixdl.cp_async_commit_group()

        def issue_v_sme(kv_tile, chunk_idx, chunk_base):
            v_k = fx.slice(
                v_tiles_all,
                (None, None,
                 k_tile_base + fx.Int32(kv_tile), chunk_idx))
            v_div = fx.zipped_divide(v_k, tile_sme)
            for t in fx.range_constexpr(k_per_warp):
                atom_idx = warp_k_start + t
                ni = atom_idx // cta_atoms_k
                ki = atom_idx % cta_atoms_k
                v_off = (fx.Int32(chunk_base)
                         + atom_idx * fx.Int32(BRICK_ELEMS))
                fx.copy_atom_call(
                    sme_atom_col,
                    fx.slice(v_div, (None, (ni, ki))),
                    _sme_view_dyn(
                        smem_ptr, fx.BFloat16, v_off, True))
            fx.ixdl.cp_async_commit_group()

        pending_q_frag = None
        pending_k_frags = None
        pending_valid = False

        def load_k_frags(sK, kk, k_tile_idx):
            ki = kk // 2
            kk_in = kk % 2
            k_step_global = k_tile_idx * k_rep + kk
            fQ = q_reg_frags[k_step_global]
            kFs = []
            for jm in fx.range_constexpr(WARP_ATOMS_M):
                k_tile_s = fx.slice(
                    sK[jm][ki], (None, kk_in))
                frag_K = thr_mma.make_fragment_A(k_tile_s)
                fx.copy(copy_atom_s2r,
                        thr_copy_A.partition_S(k_tile_s),
                        thr_copy_A.retile(frag_K), pred=None)
                kFs.append(frag_K)
            return fQ, kFs

        def mma_frags(frag_Q, k_frags):
            for jm in fx.range_constexpr(WARP_ATOMS_M):
                fx.gemm(mma_atom, accs[jm],
                        k_frags[jm], frag_Q, accs[jm])

        def compute_qk_stage(sK, k_tile_idx, first=False, last=False):
            nonlocal pending_q_frag, pending_k_frags, pending_valid
            if first and pending_valid:
                mma_frags(pending_q_frag, pending_k_frags)
                pending_valid = False
            # Prefetch first K slice before the MMAD loop
            fQ_cur, kFs_cur = load_k_frags(sK, 0, k_tile_idx)
            for kk in fx.range_constexpr(k_rep - 1):
                # MMAD on current slice; prefetch next (load-execute overlap)
                mma_frags(fQ_cur, kFs_cur)
                fQ_cur, kFs_cur = load_k_frags(sK, kk + 1, k_tile_idx)
            if last:
                mma_frags(fQ_cur, kFs_cur)
            else:
                pending_q_frag = fQ_cur
                pending_k_frags = kFs_cur
                pending_valid = True

        # ---- Dummy B-tile for P fragment creation ----
        dummy_bf16_ptr = fx.recast_iter(
            fx.PointerType.get(
                fx.BFloat16.ir_type, fx.AddressSpace.Shared),
            smem_ptr)
        dummy_b_tile = fx.Tensor(fx.make_view(
            dummy_bf16_ptr,
            fx.make_layout((ATOM_N, ATOM_K), (ATOM_K, 1))))

        # ============================================================
        #  Main KV-tile loop (runtime scf.for_, MR-style pipeline)
        #
        #  Runtime loop eliminates i-cache pressure for long sequences.
        #  BK=128 eliminates K0/K1 split (single K load per KV tile).
        #
        #  Prologue: Load K for tile 0, wait, barrier.
        #  Per-tile flow:
        #    Issue V (async) →
        #    QK [K already in SMEM] →
        #    Softmax [V streams in background] →
        #    waitcnt V + barrier →
        #    Issue K_next (async, conditional) →
        #    PV MMAD [K_next streams in background] →
        #    waitcnt K_next + barrier (conditional)
        # ============================================================
        num_kv_tiles = seq_k // BN

        # ---- Prologue: load K for tile 0, fine-grained drain ----
        for kt in fx.range_constexpr(k_tiles):
            issue_k_stage(0, kt, kt * k_stage_elems)
        _sync_arrive()
        _sync_wait()

        for kv_tile, _iter_args, _loop_results in scf.for_(
                0, num_kv_tiles, 1,
                iter_args=[m_running, l_running]):
            m_running = _iter_args[0]
            l_running = _iter_args[1]

            for jm in fx.range_constexpr(WARP_ATOMS_M):
                accs[jm].fill(0)

            pending_valid = False

            # Issue V at the start so V streams during QK+softmax
            for kt in fx.range_constexpr(k_tiles):
                issue_v_sme(kv_tile, kt, v_chunk_bases[kt])

            # QK on K (already in SMEM from prologue/prefetch)
            for kt in fx.range_constexpr(k_tiles):
                compute_qk_stage(
                    k_atoms_list[kt], kt,
                    first=(kt > 0), last=(kt == k_tiles - 1))

            # ============================================================
            #  Online softmax (V0,V1 still streaming in background)
            # ============================================================
            all_vals = []
            for jm in fx.range_constexpr(WARP_ATOMS_M):
                all_vals.append(Vec(accs[jm].load()))

            local_max = c_neg_inf
            for jm in fx.range_constexpr(WARP_ATOMS_M):
                for v in fx.range_constexpr(FRAG_ELEMS):
                    scaled = all_vals[jm][v] * c_scale_log2e
                    local_max = fx.Float32(local_max).maxnumf(
                        fx.Float32(scaled))

            for mask in [16, 32]:
                peer = fx.Float32(local_max).shuffle_xor(
                    fx.Int32(mask), fx.Int32(WARP_SIZE))
                local_max = fx.Float32(local_max).maxnumf(
                    fx.Float32(peer))

            m_old = m_running
            m_new = fx.Float32(m_old).maxnumf(fx.Float32(local_max))
            corr = fx.Float32(m_old - m_new).exp2()

            for nt in fx.range_constexpr(d_atoms):
                pv_v = Vec(pv_accs[nt].load())
                elems = []
                for i in fx.range_constexpr(FRAG_ELEMS):
                    elems.append(pv_v[i] * corr)
                pv_accs[nt].store(
                    vector.from_elements(
                        T.vec(FRAG_ELEMS, T.f32), elems))

            l_running_old = l_running

            local_sum = arith.constant(0.0, type=T.f32)
            neg_m_new = arith.negf(_to_raw(m_new))
            for jm in fx.range_constexpr(WARP_ATOMS_M):
                p_elems = []
                for v in fx.range_constexpr(FRAG_ELEMS):
                    fma_val = _math_dialect.fma(
                        _to_raw(all_vals[jm][v]),
                        _to_raw(c_scale_log2e),
                        neg_m_new)
                    p = fx.Float32(fma_val).exp2()
                    p_elems.append(p)
                    local_sum = local_sum + p
                accs[jm].store(
                    vector.from_elements(
                        T.vec(FRAG_ELEMS, T.f32), p_elems))

            for mask in [16, 32]:
                ps = fx.Float32(local_sum).shuffle_xor(
                    fx.Int32(mask), fx.Int32(WARP_SIZE))
                local_sum = local_sum + ps

            l_running = fx.Float32(
                _math_dialect.fma(
                    _to_raw(l_running_old), _to_raw(corr),
                    _to_raw(local_sum)))
            m_running = m_new

            fx.ixdl.sched_barrier()

            # Drain V async + cross-warp coherence.
            _sync_arrive()
            _sync_wait()

            # Prefetch next tile's K DURING PV MMAD.
            not_last = CmpIOp(
                CmpIPredicate.ne,
                _to_raw(fx.Int32(kv_tile)),
                _to_raw(fx.Int32(num_kv_tiles - 1))).result
            kv_next = fx.Int32(kv_tile) + fx.Int32(1)
            _if_prefetch = scf.IfOp(not_last)
            with ir.InsertionPoint(_if_prefetch.then_block):
                for kt in fx.range_constexpr(k_tiles):
                    issue_k_stage(kv_next, kt, kt * k_stage_elems)
                scf.YieldOp([])

            # ============================================================
            #  PV MMAD: V as A-operand, P as B-operand -> O^T[d, m]
            # ============================================================
            d_atoms_per_chunk = BK // ATOM_N

            for ki in fx.range_constexpr(k_steps_pv):
                p_vals = Vec(accs[ki].load())
                bf16_list = []
                for v in fx.range_constexpr(FRAG_ELEMS):
                    bf16_list.append(p_vals[v].to(BFloat16))
                p_vec = vector.from_elements(
                    T.vec(FRAG_ELEMS, T.bf16), bf16_list)

                frag_P = thr_mma.make_fragment_B(dummy_b_tile)
                frag_P.store(p_vec)

                for chunk_id in fx.range_constexpr(k_tiles):
                    chunk_sme_base = v_chunk_bases[chunk_id]

                    for nt_loc in fx.range_constexpr(
                            d_atoms_per_chunk):
                        nt = chunk_id * d_atoms_per_chunk + nt_loc
                        jn = ki
                        d_brick = nt_loc // 2
                        d_sub = nt_loc % 2

                        brick_off = fx.Int32(
                            chunk_sme_base
                            + (jn * cta_atoms_k + d_brick)
                            * BRICK_ELEMS)
                        v_view = _sme_view_dyn(
                            smem_ptr, fx.BFloat16, brick_off, True)
                        v_atoms = fx.zipped_divide(
                            v_view, tile_atom_B)
                        v_tile_s = fx.slice(
                            v_atoms, (None, d_sub))

                        frag_V = thr_mma.make_fragment_B(v_tile_s)
                        fx.copy(copy_atom_s2r,
                                thr_copy_B.partition_S(v_tile_s),
                                thr_copy_B.retile(frag_V),
                                pred=None)

                        fx.gemm(mma_atom, pv_accs[nt],
                                frag_V, frag_P, pv_accs[nt])

                fx.ixdl.sched_barrier()

            # Drain K_next async (if any) + cross-warp coherence.
            _sync_arrive()
            _sync_wait()

            yield [m_running, l_running]

        m_running = _loop_results[0]
        l_running = _loop_results[1]

        # ============================================================
        #  Epilogue: TransposeCToB16 (MR-style, HEAD_DIM=128 path)
        #
        #  O^T accumulator: M=head_dim (v*4+laneRow), N=seq_q (laneCol).
        #  Process 4 head_dim atoms at a time:
        #    - Pack bf16x2: vr0 = {atom0, atom2}, vr1 = {atom1, atom3}
        #    - TransposeCToB16: 512-word warp-private SMEM, __byte_perm
        #    - Direct i32 global stores, no shuffle needed.
        # ============================================================

        inv_l = arith.constant(1.0, type=T.f32) / l_running

        i32_smem_ptr = fx.recast_iter(
            fx.PointerType.get(T.i32, fx.AddressSpace.Shared),
            smem_ptr)
        tc_warp_base = warp_id * fx.Int32(TC_WORDS_PER_WARP)
        tc_smem = fx.Tensor(fx.make_view(
            fx.add_offset(i32_smem_ptr,
                          fx.make_int_tuple(tc_warp_base)),
            fx.make_layout((TC_WORDS_PER_WARP,), (1,))))

        lane5 = lane_id >> fx.Int32(5)
        lane04 = lane_id & fx.Int32(31)
        lane24 = lane04 >> fx.Int32(2)

        laneCol = lane_id % fx.Int32(16)
        laneRow = lane_id // fx.Int32(16)

        c_16_raw = _to_raw(fx.Int32(16))
        lo_mask_raw = _to_raw(fx.Int32(0xFFFF))
        hi_mask_raw = _to_raw(fx.Int32(-65536))

        seq_q_g_base = warp_id * fx.Int32(WARP_M)

        for ni_quad in fx.range_constexpr(d_atoms // 4):
            nt0 = ni_quad * 4
            nt1 = ni_quad * 4 + 1
            nt2 = ni_quad * 4 + 2
            nt3 = ni_quad * 4 + 3

            pv_v0 = Vec(pv_accs[nt0].load())
            pv_v1 = Vec(pv_accs[nt1].load())
            pv_v2 = Vec(pv_accs[nt2].load())
            pv_v3 = Vec(pv_accs[nt3].load())

            def _norm_trunc(val):
                normed = val * inv_l
                raw = normed._ir_value if hasattr(normed, '_ir_value') else _to_raw(normed)
                return TruncFOp(T.bf16, raw).result

            def _pack_bf16x2(bf16_lo, bf16_hi):
                i16_lo = BitcastOp(T.i16, bf16_lo).result
                i16_hi = BitcastOp(T.i16, bf16_hi).result
                i32_lo = ExtUIOp(T.i32, i16_lo).result
                i32_hi = ShLIOp(ExtUIOp(T.i32, i16_hi).result, c_16_raw).result
                return OrIOp(i32_lo, i32_hi).result

            vr0_vals = []
            vr1_vals = []
            for v in fx.range_constexpr(FRAG_ELEMS):
                h0 = _norm_trunc(pv_v0[v])
                h1 = _norm_trunc(pv_v1[v])
                h2 = _norm_trunc(pv_v2[v])
                h3 = _norm_trunc(pv_v3[v])
                vr0_vals.append(_pack_bf16x2(h0, h2))
                vr1_vals.append(_pack_bf16x2(h1, h3))

            # TransposeCToB16 store (8 stores per thread)
            for i in fx.range_constexpr(FRAG_ELEMS):
                idx0 = (lane04 * fx.Int32(16)
                        + ((fx.Int32(i) ^ lane24) * fx.Int32(2))
                        + lane5)
                idx1 = (lane04 * fx.Int32(16)
                        + ((fx.Int32(i + 4) ^ lane24) * fx.Int32(2))
                        + lane5)
                tc_smem[idx0] = vr0_vals[i]
                tc_smem[idx1] = vr1_vals[i]

            # TransposeCToB16 load + byte_perm (8 loads → 8 outputs)
            out0_vals = []
            out1_vals = []
            for i in fx.range_constexpr(FRAG_ELEMS):
                load_idx0 = (fx.Int32(i) * fx.Int32(64)
                             + (lane_id ^ (fx.Int32(i) * fx.Int32(2))))
                load_idx1 = (fx.Int32(i + 4) * fx.Int32(64)
                             + (lane_id ^ (fx.Int32(i + 4) * fx.Int32(2))))
                val0_raw = _to_raw(tc_smem[load_idx0])
                val1_raw = _to_raw(tc_smem[load_idx1])

                bp0 = OrIOp(
                    AndIOp(val0_raw, lo_mask_raw).result,
                    ShLIOp(AndIOp(val1_raw, lo_mask_raw).result, c_16_raw).result
                ).result
                bp1 = OrIOp(
                    ShRUIOp(val0_raw, c_16_raw).result,
                    AndIOp(val1_raw, hi_mask_raw).result
                ).result
                out0_vals.append(bp0)
                out1_vals.append(bp1)

            # Global i32 stores: 2 column groups per quad
            for ei in fx.range_constexpr(FRAG_ELEMS):
                row_g = seq_q_g_base + fx.Int32(ei * 4) + laneRow

                col_g0 = fx.Int32(ni_quad * 4 * ATOM_N // 2) + laneCol
                out_i32_tile[row_g, col_g0] = out0_vals[ei]

                col_g1 = fx.Int32((ni_quad * 4 + 2) * ATOM_N // 2) + laneCol
                out_i32_tile[row_g, col_g1] = out1_vals[ei]

    return flash_attn_opt_kernel, threads, smem_bytes, (BM, BN, BK)


def _build_launcher(batch, head_q, head_kv, seq_q, seq_k, k_rep,
                    num_warps=None):
    kernel, threads, smem_bytes, tile = _build_kernel(
        batch, head_q, head_kv, seq_q, seq_k, k_rep, num_warps)
    grid = (seq_q // tile[0], batch, head_q)
    block = (threads, 1, 1)

    @flyc.jit
    def flash_attn_opt(Q, K, V, O, stream=fx.Stream(None)):
        kernel(Q, K, V, O).launch(
            grid=grid, block=block, smem=smem_bytes, stream=stream)

    return flash_attn_opt, (grid, block, smem_bytes, tile)


def _make_tensors(batch, head_q, head_kv, seq_q, seq_k):
    torch.manual_seed(0)
    q = (0.1 * torch.randn(batch, head_q, seq_q, HEAD_DIM)).to(
        torch.bfloat16).cuda()
    k = (0.1 * torch.randn(batch, head_kv, seq_k, HEAD_DIM)).to(
        torch.bfloat16).cuda()
    v = torch.randn(batch, head_kv, seq_k, HEAD_DIM,
                    dtype=torch.bfloat16).cuda()
    o = torch.empty_like(q)
    return q, k, v, o


def _check(batch, head_q, head_kv, seq_q, seq_k, k_rep, num_warps=None):
    import time as _t
    _t0 = _t.perf_counter()
    q, k, v, o = _make_tensors(batch, head_q, head_kv, seq_q, seq_k)
    print(f"[timing] tensors: {_t.perf_counter()-_t0:.2f}s", flush=True)

    _t0 = _t.perf_counter()
    launcher, cfg = _build_launcher(
        batch, head_q, head_kv, seq_q, seq_k, k_rep, num_warps)
    print(f"[timing] build_launcher: {_t.perf_counter()-_t0:.2f}s",
          flush=True)

    stream = torch.cuda.Stream()
    q_flat, k_flat, v_flat, o_flat = (
        q.reshape(-1), k.reshape(-1), v.reshape(-1), o.reshape(-1))

    _t0 = _t.perf_counter()
    compiled = flyc.compile(
        launcher, q_flat, k_flat, v_flat, o_flat, fx.Stream(stream))
    print(f"[timing] compile: {_t.perf_counter()-_t0:.2f}s", flush=True)

    if os.environ.get("COMPILE_ONLY", "0") == "1":
        print("[timing] COMPILE_ONLY=1, skipping launch", flush=True)
        return True

    _t0 = _t.perf_counter()
    compiled(q_flat, k_flat, v_flat, o_flat, fx.Stream(stream))
    print(f"[timing] launch: {_t.perf_counter()-_t0:.2f}s", flush=True)
    torch.cuda.synchronize()
    print(f"[timing] sync: {_t.perf_counter()-_t0:.2f}s", flush=True)

    expected = _reference(q, k, v)
    diff = (o.float() - expected.float()).abs()
    max_abs = diff.max().item()
    finite = torch.isfinite(o.float()).all().item()
    ok = torch.allclose(
        o.float(), expected.float(), atol=6e-2, rtol=6e-2)
    print(
        f"[check] opt B={batch} Hq={head_q} Hkv={head_kv} "
        f"Sq={seq_q} Sk={seq_k} cfg={cfg} finite={finite} "
        f"max_abs={max_abs:.3e} ok={ok}")
    if not ok:
        fi = diff.reshape(-1).argmax().item()
        mp = tuple(int(x) for x in torch.unravel_index(
            torch.tensor(fi), diff.shape))
        print("[check] max_pos", mp,
              "o", o[mp].float().item(),
              "expected", expected[mp].float().item())
        print("[check] o[0,0,0,:8]       ",
              o[0, 0, 0, :8].float().cpu())
        print("[check] expected[0,0,0,:8]",
              expected[0, 0, 0, :8].float().cpu())
    return bool(ok and finite)


def _bench(batch, head_q, head_kv, seq_q, seq_k, k_rep, iters, warmup,
           num_warps=None):
    q, k, v, o = _make_tensors(batch, head_q, head_kv, seq_q, seq_k)
    launcher, cfg = _build_launcher(
        batch, head_q, head_kv, seq_q, seq_k, k_rep, num_warps)
    stream = torch.cuda.Stream()
    q_flat, k_flat, v_flat, o_flat = (
        q.reshape(-1), k.reshape(-1), v.reshape(-1), o.reshape(-1))

    t0 = time.perf_counter()
    compiled = flyc.compile(
        launcher, q_flat, k_flat, v_flat, o_flat, fx.Stream(stream))
    torch.cuda.synchronize()
    print(f"[compile] opt {(time.perf_counter()-t0)*1e3:.1f} ms "
          f"cfg={cfg}")

    for _ in range(warmup):
        compiled(q_flat, k_flat, v_flat, o_flat, fx.Stream(stream))
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record()
        for _ in range(iters):
            compiled(
                q_flat, k_flat, v_flat, o_flat, fx.Stream(stream))
        end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    per_iter_us = total_ms * 1e3 / iters
    flops = (2.0 * batch * head_q * seq_q * seq_k
             * (HEAD_DIM + HEAD_DIM))
    tflops = flops / (per_iter_us * 1e-6) / 1e12
    print(
        f"[bench] opt B={batch} Hq={head_q} Hkv={head_kv} "
        f"Sq={seq_q} Sk={seq_k} {per_iter_us:.1f} us/iter "
        f"{tflops:.2f} TFLOPS")
    return per_iter_us, tflops


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--head-q", type=int, default=6)
    p.add_argument("--head-kv", type=int, default=3)
    p.add_argument("--seq-q", type=int, default=256)
    p.add_argument("--seq-k", type=int, default=256)
    p.add_argument("--k-rep", type=int, default=8, choices=[2, 4, 8])
    p.add_argument("--num-warps", type=int, default=None)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--skip-check", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if not args.skip_check:
        if not _check(args.batch, args.head_q, args.head_kv,
                       args.seq_q, args.seq_k, args.k_rep,
                       args.num_warps):
            sys.exit(1)
    if not args.check_only:
        _bench(args.batch, args.head_q, args.head_kv,
               args.seq_q, args.seq_k, args.k_rep,
               args.iters, args.warmup, args.num_warps)
