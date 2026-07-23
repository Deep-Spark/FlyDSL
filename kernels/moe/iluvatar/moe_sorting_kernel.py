# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar MoE token sorting V1 (fp32 weights, i32 ids/counts).

V1 scope is intentionally narrow:

- Independent entry ``compile_iluvatar_moe_sorting`` (no fused softmax).
- No ``expert_mask`` (EP mode) and no ``moe_buf`` zero-init side product.
- Single kernel, single block (``grid=(1,1,1)``, ``block=(256,1,1)``), three
  phases (histogram → prefix / expert_ids / sentinel fill → scatter)
  separated by ``__syncthreads()``.
- Two-pass per-expert scan, completely atomic-free — output is deterministic
  and bit-exact reproducible.
- Compile-time ``num_experts / topk / unit_size``, runtime ``M``.

Output layout (CK-compatible packed format):

- ``sorted_ids[max_padded]``: packed ``(topk_pos << 24) | token_id``;
  padding sentinel ``(topk << 24) | M``.
- ``sorted_weights[max_padded]``: fp32; padding slots filled with ``0.0``.
- ``sorted_expert_ids[max_blocks]``: expert id per ``unit_size`` block.
- ``num_valid_ids[2] = [total_padded, M]``.

Where ``max_padded = M * topk + E * unit_size - topk`` and
``max_blocks = ceil(max_padded / unit_size)``.
"""

import math
from collections.abc import Callable

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from kernels.gemm.iluvatar.common import WARP_SIZE

BLOCK_THREADS = 256
NUM_WARPS = BLOCK_THREADS // WARP_SIZE
DEFAULT_UNIT_SIZE = 32

TORCH_I32_NAME = "torch.int32"
TORCH_F32_NAME = "torch.float32"


def _dtype_name(tensor) -> str:
    return str(tensor.dtype)


def _byte_range(tensor) -> tuple[int, int]:
    start = int(tensor.data_ptr())
    end = start + int(tensor.numel()) * int(tensor.element_size())
    return start, end


def _tensors_overlap(a, b) -> bool:
    a0, a1 = _byte_range(a)
    b0, b1 = _byte_range(b)
    return max(a0, b0) < min(a1, b1)


def _validate_compile_args(num_experts: int, topk: int, unit_size: int) -> None:
    if not isinstance(num_experts, int):
        raise ValueError(f"num_experts must be int, got {type(num_experts).__name__}")
    if not isinstance(topk, int):
        raise ValueError(f"topk must be int, got {type(topk).__name__}")
    if not isinstance(unit_size, int):
        raise ValueError(f"unit_size must be int, got {type(unit_size).__name__}")
    if num_experts <= 0:
        raise ValueError(f"num_experts must be > 0, got {num_experts}")
    if topk <= 0:
        raise ValueError(f"topk must be > 0, got {topk}")
    if unit_size <= 0:
        raise ValueError(f"unit_size must be > 0, got {unit_size}")
    if topk > num_experts:
        raise ValueError(f"topk must be <= num_experts, got topk={topk} num_experts={num_experts}")
    # Sentinel packing requires (topk << 24) + M to stay in i32; guard the
    # topk-side (M-side is checked at runtime).
    if topk >= 128:
        raise ValueError(f"topk must be < 128 to fit the packed-id sentinel, got {topk}")


def _build_moe_sorting_kernel(*, num_experts: int, topk: int, unit_size: int):
    E = int(num_experts)
    K = int(topk)
    U = int(unit_size)
    log2_warp = int(math.log2(WARP_SIZE))
    assert (1 << log2_warp) == WARP_SIZE, "WARP_SIZE must be a power of two"

    # Shared-memory layout — depends on compile-time E, so the struct is defined
    # inside the factory rather than at module scope.
    @fx.struct
    class _SmemLayout:
        s_count: fx.Array[fx.Int32, E]
        s_prefix: fx.Array[fx.Int32, E]
        # NUM_WARPS slots for per-warp scan bases + 1 extra slot for block total.
        s_warp_sums: fx.Array[fx.Int32, NUM_WARPS + 1]

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def _moe_sorting_kernel(
        topk_ids: fx.Tensor,
        topk_weights: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        M: fx.Int32,
    ):
        tid = fx.thread_idx.x
        lane_id = fx.Int32(fx.lane_id)
        warp_id = tid // WARP_SIZE

        c_zero_i = fx.Int32(0)
        c_one_i = fx.Int32(1)

        smem = fx.SharedAllocator().allocate(_SmemLayout).peek()
        s_count = smem.s_count.view(fx.make_layout(E, 1))
        s_prefix = smem.s_prefix.view(fx.make_layout(E, 1))
        s_warp_sums = smem.s_warp_sums.view(fx.make_layout(NUM_WARPS + 1, 1))

        # ---- Nested helpers ------------------------------------------------
        def _warp_reduce_add(x):
            """Butterfly sum across the warp via shuffle_xor (all lanes get total)."""
            v = x
            for sh_exp in range_constexpr(log2_warp):
                offset = WARP_SIZE // (2 << sh_exp)
                peer = v.shuffle_xor(offset, WARP_SIZE)
                v = v + peer
            return v

        def _warp_scan_inclusive_add(x):
            """Kogge-Stone inclusive prefix sum within the warp using shuffle_idx."""
            v = x
            for sh_exp in range_constexpr(log2_warp):
                shift = 1 << sh_exp
                src_lane_safe = (lane_id >= fx.Int32(shift)).select(lane_id - fx.Int32(shift), lane_id)
                peer = v.shuffle_idx(src_lane_safe, WARP_SIZE)
                v = (lane_id >= fx.Int32(shift)).select(v + peer, v)
            return v

        def _block_reduce_add(x):
            """Block-wide sum; broadcast to all threads."""
            if const_expr(NUM_WARPS == 1):
                return _warp_reduce_add(x)

            warp_sum = _warp_reduce_add(x)
            if lane_id == 0:
                fx.memref_store(warp_sum, s_warp_sums, warp_id)
            gpu.barrier()

            if warp_id == 0:
                in_range = lane_id < fx.Int32(NUM_WARPS)
                lane_safe = in_range.select(lane_id, c_zero_i)
                v = fx.memref_load(s_warp_sums, lane_safe)
                vv = in_range.select(v, c_zero_i)
                total = _warp_reduce_add(vv)
                if lane_id == 0:
                    fx.memref_store(total, s_warp_sums, fx.Int32(NUM_WARPS))
            gpu.barrier()

            return fx.memref_load(s_warp_sums, fx.Int32(NUM_WARPS))

        def _block_scan_exclusive_add(x):
            """Block-wide exclusive scan. Returns (exclusive_rank, block_total)."""
            warp_incl = _warp_scan_inclusive_add(x)
            warp_excl = warp_incl - x
            # Broadcast this warp's total from its last lane.
            warp_sum = warp_incl.shuffle_idx(fx.Int32(WARP_SIZE - 1), WARP_SIZE)

            if lane_id == 0:
                fx.memref_store(warp_sum, s_warp_sums, warp_id)
            gpu.barrier()

            # Sequentially derive cross-warp exclusive prefix and block total in tid==0.
            if tid == 0:
                acc = c_zero_i
                for w in range_constexpr(NUM_WARPS):
                    v = fx.memref_load(s_warp_sums, fx.Int32(w))
                    fx.memref_store(acc, s_warp_sums, fx.Int32(w))
                    acc = acc + v
                fx.memref_store(acc, s_warp_sums, fx.Int32(NUM_WARPS))
            gpu.barrier()

            warp_base = fx.memref_load(s_warp_sums, warp_id)
            block_total = fx.memref_load(s_warp_sums, fx.Int32(NUM_WARPS))
            return warp_base + warp_excl, block_total

        # -- Precomputed constants -------------------------------------------
        pair_total = M * fx.Int32(K)
        n_iters = (pair_total + fx.Int32(BLOCK_THREADS - 1)) // fx.Int32(BLOCK_THREADS)
        sentinel = (fx.Int32(K) << fx.Int32(24)) | M

        # ---------------------------------------------------------------
        # Phase 1: histogram — count[e] = #{(t,k) : topk_ids[t,k] == e}
        # ---------------------------------------------------------------
        for e in range(E):
            my_count = c_zero_i
            for i in range(n_iters):
                pair_idx = fx.Int32(i) * fx.Int32(BLOCK_THREADS) + tid
                valid = pair_idx < pair_total
                t = pair_idx // fx.Int32(K)
                k = pair_idx - t * fx.Int32(K)
                t_safe = valid.select(t, c_zero_i)
                k_safe = valid.select(k, c_zero_i)
                eid = fx.Int32(topk_ids[t_safe, k_safe])
                is_match = valid & (eid == fx.Int32(e))
                my_count = my_count + is_match.select(c_one_i, c_zero_i)
            total = _block_reduce_add(my_count)
            if tid == 0:
                fx.memref_store(total, s_count, fx.Int32(e))
            gpu.barrier()

        # ---------------------------------------------------------------
        # Phase 2: single-thread sequential — prefix, expert_ids, sentinel, nv
        # ---------------------------------------------------------------
        if tid == 0:
            ids_cursor = c_zero_i
            exp_ids_cursor = c_zero_i
            for e in range(E):
                c = fx.memref_load(s_count, fx.Int32(e))
                nb = (c + fx.Int32(U - 1)) // fx.Int32(U)
                padded = nb * fx.Int32(U)
                fx.memref_store(ids_cursor, s_prefix, fx.Int32(e))
                # Write expert id into sorted_expert_ids for each of this expert's blocks.
                for i in range(nb):
                    sorted_expert_ids[exp_ids_cursor + fx.Int32(i)] = fx.Int32(e)
                exp_ids_cursor = exp_ids_cursor + nb
                # Fill tail padding of this expert's slots with sentinel / 0.
                for j in range(c, padded):
                    sorted_ids[ids_cursor + fx.Int32(j)] = sentinel
                    sorted_weights[ids_cursor + fx.Int32(j)] = fx.Float32(0.0)
                ids_cursor = ids_cursor + padded
            num_valid_ids[fx.Int32(0)] = ids_cursor
            num_valid_ids[fx.Int32(1)] = M
        gpu.barrier()

        # ---------------------------------------------------------------
        # Phase 3: scatter — write packed_id + weight into per-expert range
        # ---------------------------------------------------------------
        for e in range(E):
            prefix_e = fx.memref_load(s_prefix, fx.Int32(e))
            wc = c_zero_i
            for i in range(n_iters):
                pair_idx = fx.Int32(i) * fx.Int32(BLOCK_THREADS) + tid
                valid = pair_idx < pair_total
                t = pair_idx // fx.Int32(K)
                k = pair_idx - t * fx.Int32(K)
                t_safe = valid.select(t, c_zero_i)
                k_safe = valid.select(k, c_zero_i)
                eid = fx.Int32(topk_ids[t_safe, k_safe])
                is_match = valid & (eid == fx.Int32(e))
                delta = is_match.select(c_one_i, c_zero_i)
                local_rank, block_total = _block_scan_exclusive_add(delta)
                if is_match:
                    slot = prefix_e + wc + local_rank
                    packed = (k_safe << fx.Int32(24)) | t_safe
                    sorted_ids[slot] = packed
                    sorted_weights[slot] = fx.Float32(topk_weights[t_safe, k_safe])
                wc = wc + block_total

    return _moe_sorting_kernel


def compile_iluvatar_moe_sorting(
    *,
    num_experts: int,
    topk: int,
    unit_size: int = DEFAULT_UNIT_SIZE,
) -> Callable:
    """Build an Iluvatar MoE token sorting launcher.

    Args:
        num_experts: Number of experts ``E``. Compile-time constant, must be ``> 0``.
        topk: Router top-k per token. Compile-time constant, must satisfy
            ``0 < topk <= num_experts`` and ``topk < 128``.
        unit_size: Padding granularity per expert block (GEMM tile-M).
            Compile-time constant, must be ``> 0``.

    Returns:
        ``launch(topk_ids, topk_weights, sorted_ids, sorted_weights,
        sorted_expert_ids, num_valid_ids, stream=None)`` which fills the caller-
        allocated output tensors. Caller sizes them via::

            max_padded = M * topk + num_experts * unit_size - topk
            max_blocks = (max_padded + unit_size - 1) // unit_size
    """
    _validate_compile_args(num_experts, topk, unit_size)
    kernel = _build_moe_sorting_kernel(num_experts=num_experts, topk=topk, unit_size=unit_size)

    @flyc.jit
    def _launch_kernel(
        topk_ids: fx.Tensor,
        topk_weights: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        kernel(
            topk_ids,
            topk_weights,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            m_in,
        ).launch(
            grid=(1, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    def launch_moe_sorting(
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        stream=None,
    ):
        # ---- Dim / shape validation ------------------------------------
        if topk_ids.dim() != 2:
            raise ValueError(
                f"expected topk_ids shape (M,topk), got dim={topk_ids.dim()} shape={tuple(topk_ids.shape)}"
            )
        if topk_weights.dim() != 2:
            raise ValueError(
                f"expected topk_weights shape (M,topk), got dim={topk_weights.dim()} "
                f"shape={tuple(topk_weights.shape)}"
            )
        if sorted_ids.dim() != 1:
            raise ValueError(f"expected sorted_ids 1D, got dim={sorted_ids.dim()} shape={tuple(sorted_ids.shape)}")
        if sorted_weights.dim() != 1:
            raise ValueError(
                f"expected sorted_weights 1D, got dim={sorted_weights.dim()} shape={tuple(sorted_weights.shape)}"
            )
        if sorted_expert_ids.dim() != 1:
            raise ValueError(
                f"expected sorted_expert_ids 1D, got dim={sorted_expert_ids.dim()} "
                f"shape={tuple(sorted_expert_ids.shape)}"
            )
        if num_valid_ids.dim() != 1:
            raise ValueError(
                f"expected num_valid_ids 1D shape (2,), got dim={num_valid_ids.dim()} "
                f"shape={tuple(num_valid_ids.shape)}"
            )

        M = int(topk_ids.shape[0])
        if int(topk_ids.shape[1]) != topk:
            raise ValueError(f"expected topk_ids.shape[1] == topk={topk}, got {int(topk_ids.shape[1])}")
        if tuple(topk_weights.shape) != (M, topk):
            raise ValueError(f"expected topk_weights shape (M,topk)=({M},{topk}), got {tuple(topk_weights.shape)}")

        max_padded_needed = M * topk + num_experts * unit_size - topk
        max_blocks_needed = (max_padded_needed + unit_size - 1) // unit_size
        if int(sorted_ids.shape[0]) < max_padded_needed:
            raise ValueError(
                f"sorted_ids too small: need >= {max_padded_needed} elements, got {int(sorted_ids.shape[0])}"
            )
        if int(sorted_weights.shape[0]) < max_padded_needed:
            raise ValueError(
                f"sorted_weights too small: need >= {max_padded_needed} elements, "
                f"got {int(sorted_weights.shape[0])}"
            )
        if int(sorted_expert_ids.shape[0]) < max_blocks_needed:
            raise ValueError(
                f"sorted_expert_ids too small: need >= {max_blocks_needed} elements, "
                f"got {int(sorted_expert_ids.shape[0])}"
            )
        if int(num_valid_ids.shape[0]) != 2:
            raise ValueError(f"expected num_valid_ids shape (2,), got {tuple(num_valid_ids.shape)}")

        # Sentinel packing requires (topk << 24) + M to fit i32.
        if M >= (1 << 24):
            raise ValueError(f"M must be < 2^24 (16777216), got {M}")

        # ---- Contiguity -------------------------------------------------
        if not topk_ids.is_contiguous():
            raise ValueError("topk_ids must be contiguous")
        if not topk_weights.is_contiguous():
            raise ValueError("topk_weights must be contiguous")
        if not sorted_ids.is_contiguous():
            raise ValueError("sorted_ids must be contiguous")
        if not sorted_weights.is_contiguous():
            raise ValueError("sorted_weights must be contiguous")
        if not sorted_expert_ids.is_contiguous():
            raise ValueError("sorted_expert_ids must be contiguous")
        if not num_valid_ids.is_contiguous():
            raise ValueError("num_valid_ids must be contiguous")

        # ---- Dtype ------------------------------------------------------
        for name, tensor, expected in (
            ("topk_ids", topk_ids, TORCH_I32_NAME),
            ("topk_weights", topk_weights, TORCH_F32_NAME),
            ("sorted_ids", sorted_ids, TORCH_I32_NAME),
            ("sorted_weights", sorted_weights, TORCH_F32_NAME),
            ("sorted_expert_ids", sorted_expert_ids, TORCH_I32_NAME),
            ("num_valid_ids", num_valid_ids, TORCH_I32_NAME),
        ):
            actual = _dtype_name(tensor)
            if actual != expected:
                raise ValueError(f"{name} dtype must be {expected}, got {actual}")

        # ---- Device -----------------------------------------------------
        dev = topk_ids.device
        for name, tensor in (
            ("topk_weights", topk_weights),
            ("sorted_ids", sorted_ids),
            ("sorted_weights", sorted_weights),
            ("sorted_expert_ids", sorted_expert_ids),
            ("num_valid_ids", num_valid_ids),
        ):
            if tensor.device != dev:
                raise ValueError(f"all tensors must be on same device; topk_ids on {dev}, {name} on {tensor.device}")

        # ---- Overlap ----------------------------------------------------
        inputs = (("topk_ids", topk_ids), ("topk_weights", topk_weights))
        outputs = (
            ("sorted_ids", sorted_ids),
            ("sorted_weights", sorted_weights),
            ("sorted_expert_ids", sorted_expert_ids),
            ("num_valid_ids", num_valid_ids),
        )
        for in_name, in_t in inputs:
            for out_name, out_t in outputs:
                if _tensors_overlap(in_t, out_t):
                    raise ValueError(f"{out_name} must not overlap with {in_name}")

        # M == 0: nothing to scatter, but Phase 2 still needs to run to write
        # num_valid_ids = [0, 0]. However Phase 1's outer loop over E ends up
        # writing count[e] = 0 for all e, and Phase 2 writes num_valid_ids
        # correctly. Skipping the kernel entirely would leave num_valid_ids
        # uninitialised, so still launch.

        if stream is None:
            _launch_kernel(
                topk_ids,
                topk_weights,
                sorted_ids,
                sorted_weights,
                sorted_expert_ids,
                num_valid_ids,
                M,
            )
        else:
            _launch_kernel(
                topk_ids,
                topk_weights,
                sorted_ids,
                sorted_weights,
                sorted_expert_ids,
                num_valid_ids,
                M,
                stream=stream,
            )
        return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids

    return launch_moe_sorting


__all__ = [
    "BLOCK_THREADS",
    "DEFAULT_UNIT_SIZE",
    "compile_iluvatar_moe_sorting",
]
