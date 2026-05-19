"""FlashAttention on IXDL – Q/K swapped roles for register-level P conversion.

Key insight from ixattention-backend: by computing QK as K(A) × Q(B) instead of
Q(A) × K(B), the C-accumulator layout (m=(lane/16)*4+v → seq_k, n=lane%16 → seq_q)
directly matches the B-operand layout for PV MMAD. This allows zero-shuffle,
thread-local f32→bf16 conversion (CastMMP) — eliminating the P SMEM round-trip.

Benefits over example 29:
  1. No P write-to-SMEM + barrier + read-from-SMEM
  2. No corr_vals SMEM redistribution (same seq_q mapping in QK/PV accumulators)
  3. Reduced SMEM usage (~64 KB vs ~65 KB)
  4. Fewer barriers overall

SMEM layout: [0,32KB)=K staging (row_xfb16, A-copy; reused from Q col_xfb8 load)
             [32KB,64KB)=V storage (col_xfb8)
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

ATOM_M = 16
ATOM_N = 16
ATOM_K = 16
SME_ROWS = 16
SME_BF16_PER_ROW = 32
WARP_SIZE = 64
STAGES = 2
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


def _build_kernel(batch, head_q, head_kv, seq_q, seq_k, k_rep):
    WARP_M = ATOM_M                        # 16
    NUM_WARPS = 8
    BM = WARP_M * NUM_WARPS                # 128
    BN = 128
    BK = ATOM_K * k_rep                    # 64 with k_rep=4
    threads = NUM_WARPS * WARP_SIZE         # 512

    WARP_ATOMS_M = BN // ATOM_M             # 8 (seq_k atoms per warp in swapped QK)
    K_STEPS_QK = HEAD_DIM // ATOM_K         # 8
    k_steps_pv = BN // ATOM_K              # 8
    d_atoms = HEAD_DIM // ATOM_N            # 8

    assert HEAD_DIM == 128
    assert seq_q % BM == 0 and seq_k % BN == 0
    assert head_q % head_kv == 0
    assert HEAD_DIM % BK == 0
    assert BK % SME_BF16_PER_ROW == 0

    repeat = head_q // head_kv
    scale_log2e = (1.0 / math.sqrt(HEAD_DIM)) * _LOG2E

    cta_atoms_k = BK // SME_BF16_PER_ROW           # 2
    cta_atoms_k_q = HEAD_DIM // SME_BF16_PER_ROW   # 4

    k_atoms_m = BN // SME_ROWS                      # 8 (seq_k bricks)
    k_atoms_total = k_atoms_m * cta_atoms_k          # 16
    q_atoms_m = BM // SME_ROWS                       # 8 (seq_q bricks)
    q_atoms_total = q_atoms_m * cta_atoms_k_q        # 32
    assert k_atoms_total % NUM_WARPS == 0
    assert q_atoms_total % NUM_WARPS == 0
    k_per_warp = k_atoms_total // NUM_WARPS          # 2
    q_per_warp = q_atoms_total // NUM_WARPS          # 4

    k_stage_elems = BN * BK                          # 8192 bf16 = 16 KB
    v_smem_offset = STAGES * k_stage_elems           # 16384 bf16 = 32 KB
    smem_bytes = (v_smem_offset + HEAD_DIM * BN) * 2 # 64 KB

    k_tiles = HEAD_DIM // BK                         # 2

    @flyc.kernel
    def flash_attn_qk_swap_kernel(
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

        # ---- SME copy atoms (SWAPPED: K→A with row_xfb16, Q→B with col_xfb8) ----
        sme_atom_K = fx.make_copy_atom(fx.ixdl.SMECopy(
            fx.BFloat16, (SME_ROWS, SME_BF16_PER_ROW),
            stride_byte=HEAD_DIM * 2, major="k",
            cache_op="cache_all", swizzle="row_xfb16",
        ), fx.BFloat16)
        sme_atom_Q = fx.make_copy_atom(fx.ixdl.SMECopy(
            fx.BFloat16, (SME_ROWS, SME_BF16_PER_ROW),
            stride_byte=HEAD_DIM * 2, major="mn",
            cache_op="cache_all", swizzle="col_xfb8",
        ), fx.BFloat16)

        # ---- S2R / R2S copy atoms ----
        copy_atom_s2r = fx.make_copy_atom(
            fx.UniversalCopy32b(), fx.BFloat16)
        copy_atom_r2s_c = fx.make_copy_atom(
            fx.UniversalCopy32b(), fx.Float32)
        tiled_copy_A = fx.make_tiled_copy_A(copy_atom_s2r, tiled_mma)
        tiled_copy_B = fx.make_tiled_copy_B(copy_atom_s2r, tiled_mma)
        tiled_copy_C = fx.make_tiled_copy_C(copy_atom_r2s_c, tiled_mma)
        thr_copy_A = tiled_copy_A.get_slice(lane_id)
        thr_copy_B = tiled_copy_B.get_slice(lane_id)
        thr_copy_C = tiled_copy_C.get_slice(lane_id)

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

        # ---- softmax state (single scalar per thread — seq_q = lane%16) ----
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

        # ---- K smem views: row_xfb16 for A-operand (transpose=False) ----
        def _make_k_atoms(stage_base):
            k_atoms_s = []
            for jm in fx.range_constexpr(WARP_ATOMS_M):
                row = []
                for ki in fx.range_constexpr(cta_atoms_k):
                    off = fx.Int32(
                        stage_base
                        + (jm * cta_atoms_k + ki) * BRICK_ELEMS)
                    row.append(fx.zipped_divide(
                        _sme_view_dyn(
                            smem_ptr, fx.BFloat16, off, False),
                        tile_atom_A))
                k_atoms_s.append(row)
            return k_atoms_s

        k0_atoms = _make_k_atoms(0)
        k1_atoms = _make_k_atoms(k_stage_elems)

        # ============================================================
        #  Phase 0: Load Q to SMEM with col_xfb8 (B-operand), copy to
        #           register B-fragments, then free SMEM for K
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
                sme_atom_Q,
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

        # ---- output global view ----
        out_2d = fx.Tensor(fx.make_view(
            fx.get_iter(O),
            fx.make_layout(
                (batch * head_q * seq_q, HEAD_DIM), (HEAD_DIM, 1))))
        out_tile = fx.slice(
            fx.flat_divide(out_2d, (BM, HEAD_DIM)),
            (None, None, q_tile_id, 0))

        # ---- V smem chunks ----
        v_chunk0_base = v_smem_offset
        v_chunk1_base = v_smem_offset + k_stage_elems

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
                k_off = (fx.Int32(stage_base)
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
                    sme_atom_Q,
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
            for kk in fx.range_constexpr(k_rep):
                fQ, kFs = load_k_frags(sK, kk, k_tile_idx)
                if kk < k_rep - 1 or last:
                    mma_frags(fQ, kFs)
                else:
                    pending_q_frag = fQ
                    pending_k_frags = kFs
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
        #  Main KV-tile loop
        # ============================================================
        for kv_tile in fx.range_constexpr(seq_k // BN):
            for jm in fx.range_constexpr(WARP_ATOMS_M):
                accs[jm].fill(0)

            # ---- K-only pipeline with deferred MMAD ----
            pending_valid = False
            issue_k_stage(kv_tile, 0, 0)
            _sync_arrive()

            for k_tile_idx in fx.range_constexpr(k_tiles - 1):
                stage_now = k_tile_idx % STAGES
                stage_next = (k_tile_idx + 1) % STAGES
                sK_now = k0_atoms if stage_now == 0 else k1_atoms
                sb = 0 if stage_next == 0 else k_stage_elems
                _sync_wait()
                issue_k_stage(kv_tile, k_tile_idx + 1, sb)
                compute_qk_stage(sK_now, k_tile_idx,
                                 first=(k_tile_idx > 0), last=False)
                _sync_arrive()

            _sync_wait()
            sK_last = (k0_atoms if (k_tiles - 1) % STAGES == 0
                       else k1_atoms)
            compute_qk_stage(sK_last, k_tiles - 1,
                             first=True, last=True)

            # ============================================================
            #  Online softmax (swapped layout: single m/l per thread)
            #  C-acc: seq_k = (lane/16)*4+v, seq_q = lane%16
            #  Reduction across seq_k for fixed seq_q:
            #    - local: across 8 m-atoms × 4 frag elems = 32 values
            #    - cross-group: shuffle_xor {16, 32}
            # ============================================================
            all_vals = []
            for jm in fx.range_constexpr(WARP_ATOMS_M):
                all_vals.append(Vec(accs[jm].load()))

            local_max = c_neg_inf
            for jm in fx.range_constexpr(WARP_ATOMS_M):
                for v in fx.range_constexpr(FRAG_ELEMS):
                    scaled = all_vals[jm][v] * c_scale_log2e
                    local_max = fx.Float32(local_max).maximumf(
                        fx.Float32(scaled))

            for mask in [16, 32]:
                peer = fx.Float32(local_max).shuffle_xor(
                    fx.Int32(mask), fx.Int32(WARP_SIZE))
                local_max = fx.Float32(local_max).maximumf(
                    fx.Float32(peer))

            m_old = m_running
            m_new = fx.Float32(m_old).maximumf(fx.Float32(local_max))
            corr = fx.Float32(m_old - m_new).exp2()

            # Apply correction directly to PV accumulators (no SMEM needed!)
            for nt in fx.range_constexpr(d_atoms):
                pv_v = Vec(pv_accs[nt].load())
                elems = []
                for i in fx.range_constexpr(FRAG_ELEMS):
                    elems.append(pv_v[i] * corr)
                pv_accs[nt].store(
                    vector.from_elements(
                        T.vec(FRAG_ELEMS, T.f32), elems))

            l_running = l_running * corr

            # Compute exp2 probabilities and accumulate sum
            local_sum = arith.constant(0.0, type=T.f32)
            for jm in fx.range_constexpr(WARP_ATOMS_M):
                p_elems = []
                for v in fx.range_constexpr(FRAG_ELEMS):
                    p = fx.Float32(
                        all_vals[jm][v] * c_scale_log2e - m_new
                    ).exp2()
                    p_elems.append(p)
                    local_sum = local_sum + p
                accs[jm].store(
                    vector.from_elements(
                        T.vec(FRAG_ELEMS, T.f32), p_elems))

            for mask in [16, 32]:
                ps = fx.Float32(local_sum).shuffle_xor(
                    fx.Int32(mask), fx.Int32(WARP_SIZE))
                local_sum = local_sum + ps

            l_running = l_running + local_sum
            m_running = m_new

            # ---- V SMECopy: load V to [32KB, 64KB) ----
            issue_v_sme(kv_tile, 0, v_chunk0_base)
            issue_v_sme(kv_tile, 1, v_chunk1_base)
            _sync_arrive()
            _sync_wait()
            gpu.barrier()

            # ============================================================
            #  PV MMAD: register-level P conversion, V from col_xfb8 as A
            #  O^T[d, m] = sum_sk V^T[d, sk] * P[m, sk]
            # ============================================================
            d_atoms_per_chunk = BK // ATOM_N  # 4

            for ki in fx.range_constexpr(k_steps_pv):
                # Thread-local P conversion: f32 → bf16
                p_vals = Vec(accs[ki].load())
                bf16_list = []
                for v in fx.range_constexpr(FRAG_ELEMS):
                    bf16_list.append(p_vals[v].to(BFloat16))
                p_vec = vector.from_elements(
                    T.vec(FRAG_ELEMS, T.bf16), bf16_list)

                frag_P = thr_mma.make_fragment_B(dummy_b_tile)
                frag_P.store(p_vec)

                for chunk_id in fx.range_constexpr(k_tiles):
                    chunk_sme_base = (v_chunk0_base if chunk_id == 0
                                      else v_chunk1_base)

                    for nt_loc in fx.range_constexpr(
                            d_atoms_per_chunk):
                        nt = chunk_id * d_atoms_per_chunk + nt_loc
                        # V brick addressing
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

                        # V as A-operand, P as B-operand
                        fx.gemm(mma_atom, pv_accs[nt],
                                frag_V, frag_P, pv_accs[nt])

            gpu.barrier()

        # ============================================================
        #  Epilogue: normalize O^T → O, write to global
        # ============================================================
        l_ptr = fx.recast_iter(
            fx.PointerType.get(
                fx.Float32.ir_type, fx.AddressSpace.Shared),
            smem_ptr)
        # Write l_running to SMEM for normalization
        # In swapped layout: seq_q = lane%16, each lane within a group
        # has a unique seq_q. Lanes 0..15, 16..31, 32..47, 48..63 in
        # the same warp share the same seq_q values.
        # Only one lane per seq_q row needs to write l_running.
        lane_group = lane_id // fx.Int32(16)
        if lane_group == fx.Int32(0):
            seq_q_local = lane_id % fx.Int32(16)
            row_g = warp_id * fx.Int32(WARP_M) + seq_q_local
            l_off = fx.add_offset(l_ptr, fx.make_int_tuple(row_g))
            fx.Tensor(fx.make_view(
                l_off, fx.make_layout((1,), (1,))))[0] = l_running

        gpu.barrier()
        l_tile = fx.Tensor(fx.make_view(
            l_ptr, fx.make_layout((BM,), (1,))))

        out_f32_ptr = fx.recast_iter(
            fx.PointerType.get(
                fx.Float32.ir_type, fx.AddressSpace.Shared),
            smem_ptr)
        out_f32_ptr = fx.add_offset(
            out_f32_ptr, fx.make_int_tuple(BM))

        for nt in fx.range_constexpr(d_atoms):
            warp_out_ptr = fx.add_offset(
                out_f32_ptr,
                fx.make_int_tuple(warp_id * WARP_M * ATOM_N))
            warp_out_tile = fx.Tensor(fx.make_view(
                warp_out_ptr,
                fx.make_layout((WARP_M, ATOM_N), (ATOM_N, 1))))
            warp_out_atoms = fx.flat_divide(
                warp_out_tile, (ATOM_M, ATOM_N))
            out_atom = fx.slice(
                warp_out_atoms, (None, None, 0, 0))
            fx.copy(copy_atom_r2s_c,
                    thr_copy_C.retile(pv_accs[nt]),
                    thr_copy_C.partition_S(out_atom), pred=None)

            gpu.barrier()

            full_out = fx.Tensor(fx.make_view(
                out_f32_ptr,
                fx.make_layout((BM, ATOM_N), (ATOM_N, 1))))

            for linear in fx.range_constexpr(
                    0, BM * ATOM_N, threads):
                elem = fx.Int32(linear) + tid
                row = elem // fx.Int32(ATOM_N)
                col = elem - row * fx.Int32(ATOM_N)
                warp_src = row // fx.Int32(WARP_M)
                d_local = row - warp_src * fx.Int32(WARP_M)
                m_local = col
                m_global = warp_src * fx.Int32(WARP_M) + m_local
                d_col = fx.Int32(nt * ATOM_N) + d_local
                val = full_out[row, col]
                out_tile[m_global, d_col] = (
                    val / l_tile[m_global]).truncf(T.bf16)

            gpu.barrier()

    return flash_attn_qk_swap_kernel, threads, smem_bytes, (BM, BN, BK)


def _build_launcher(batch, head_q, head_kv, seq_q, seq_k, k_rep):
    kernel, threads, smem_bytes, tile = _build_kernel(
        batch, head_q, head_kv, seq_q, seq_k, k_rep)
    grid = (seq_q // tile[0], batch, head_q)
    block = (threads, 1, 1)

    @flyc.jit
    def flash_attn_qk_swap(Q, K, V, O, stream=fx.Stream(None)):
        kernel(Q, K, V, O).launch(
            grid=grid, block=block, smem=smem_bytes, stream=stream)

    return flash_attn_qk_swap, (grid, block, smem_bytes, tile)


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


def _check(batch, head_q, head_kv, seq_q, seq_k, k_rep):
    import time as _t
    _t0 = _t.perf_counter()
    q, k, v, o = _make_tensors(batch, head_q, head_kv, seq_q, seq_k)
    print(f"[timing] tensors: {_t.perf_counter()-_t0:.2f}s", flush=True)

    _t0 = _t.perf_counter()
    launcher, cfg = _build_launcher(
        batch, head_q, head_kv, seq_q, seq_k, k_rep)
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
        f"[check] qk-swap B={batch} Hq={head_q} Hkv={head_kv} "
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


def _bench(batch, head_q, head_kv, seq_q, seq_k, k_rep, iters, warmup):
    q, k, v, o = _make_tensors(batch, head_q, head_kv, seq_q, seq_k)
    launcher, cfg = _build_launcher(
        batch, head_q, head_kv, seq_q, seq_k, k_rep)
    stream = torch.cuda.Stream()
    q_flat, k_flat, v_flat, o_flat = (
        q.reshape(-1), k.reshape(-1), v.reshape(-1), o.reshape(-1))

    t0 = time.perf_counter()
    compiled = flyc.compile(
        launcher, q_flat, k_flat, v_flat, o_flat, fx.Stream(stream))
    torch.cuda.synchronize()
    print(f"[compile] qk-swap {(time.perf_counter()-t0)*1e3:.1f} ms "
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
        f"[bench] qk-swap B={batch} Hq={head_q} Hkv={head_kv} "
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
    p.add_argument("--k-rep", type=int, default=4, choices=[2, 4, 8])
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--skip-check", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if not args.skip_check:
        if not _check(args.batch, args.head_q, args.head_kv,
                       args.seq_q, args.seq_k, args.k_rep):
            sys.exit(1)
    if not args.check_only:
        _bench(args.batch, args.head_q, args.head_kv,
               args.seq_q, args.seq_k, args.k_rep,
               args.iters, args.warmup)
