# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""FA fast path for ``flydsl_flex_attn_func``.

Vanilla flex-attention (no score/mask mods, no explicit tile) routes to
``flash_attn_varlen_func``. Eligibility is decided up front; FA errors are
not caught as a fallback onto the generic kernel.
"""

from typing import Optional

_FA_DTYPE = "bf16"
_FA_HEAD_DIMS = (128, 256)
_PAGED_PAGE = 64


def flex_fa_fastpath_enabled() -> bool:
    import os

    try:
        from flydsl.utils.env import runtime

        if hasattr(runtime, "flex_fa_fastpath"):
            return bool(runtime.flex_fa_fastpath)
    except (AttributeError, ImportError):
        pass
    raw = os.environ.get("FLYDSL_FLEX_FA_FASTPATH")
    if raw is None:
        return True
    return raw.lower() in ("1", "true", "yes", "on")


def is_flex_fa_fastpath_eligible(
    *,
    mode: str,
    dtype: str,
    head_dim: int,
    causal: bool,
    sq: int,
    skv: int,
    window_size: Optional[int],
    softcap: Optional[float],
    alibi_slopes,
    score_bias,
    score_mod,
    mask_mod,
    block_mask,
    block_masks,
    tile_config_explicit: bool,
    varlen_tight: bool = True,
    env_enabled: Optional[bool] = None,
) -> bool:
    """Return True iff this call is a strict FA subset (no try/except routing)."""
    enabled = flex_fa_fastpath_enabled() if env_enabled is None else bool(env_enabled)
    if not enabled:
        return False
    if tile_config_explicit:
        return False
    if score_mod is not None or mask_mod is not None:
        return False
    if block_mask is not None or block_masks is not None:
        return False
    if alibi_slopes is not None or score_bias is not None:
        return False
    if window_size is not None or softcap is not None:
        return False
    if dtype != _FA_DTYPE:
        return False
    if int(head_dim) not in _FA_HEAD_DIMS:
        return False
    if mode == "dense" and causal and int(sq) != int(skv):
        return False
    if mode == "varlen" and not varlen_tight:
        return False
    return True


def varlen_pack_is_tight(cu_seqlens, seq_lens) -> bool:
    """True when each packed span equals the logical length (no per-seq pad)."""
    import torch

    spans = cu_seqlens[1:] - cu_seqlens[:-1]
    sl = seq_lens.to(device=spans.device, dtype=spans.dtype)
    if tuple(spans.shape) != tuple(sl.shape):
        return False
    return bool(torch.equal(spans, sl))


def _i32_on(t, device):
    import torch

    out = t.contiguous()
    if out.dtype != torch.int32:
        out = out.to(torch.int32)
    if out.device != device:
        out = out.to(device)
    return out


def _pack_bshd_as_varlen(x):
    """[B, S, H, D] -> packed [B * S, H, D]."""
    b, s, h, d = x.shape
    return x.contiguous().reshape(b * s, h, d)


def _dense_cu_seqlens(*, batch: int, seqlen: int, device):
    import torch

    return torch.arange(0, batch + 1, device=device, dtype=torch.int32) * int(seqlen)


def run_flex_attn_fa_fastpath(
    mode: str,
    q,
    k,
    v,
    *,
    causal: bool,
    sm_scale: Optional[float],
    out,
    cu_seqlens,
    seq_lens,
    block_table,
    seq_lens_kv,
    max_seqlen_q: Optional[int],
    max_seqlen_kv: Optional[int],
    stream,
):
    """Invoke ``flash_attn_varlen_func`` after packing dense/paged Q to varlen.

    ``stream`` is forwarded unchanged. ``None`` lets FA bind
    ``torch.cuda.current_stream`` (do not wrap as ``fx.Stream(None)``).
    """
    from kernels.attention.iluvatar.flash_attn_varlen import flash_attn_varlen_func

    if mode == "dense":
        return _run_dense(
            q,
            k,
            v,
            causal=causal,
            sm_scale=sm_scale,
            out=out,
            stream=stream,
            flash_attn_varlen_func=flash_attn_varlen_func,
        )
    if mode == "varlen":
        return _run_varlen(
            q,
            k,
            v,
            causal=causal,
            sm_scale=sm_scale,
            out=out,
            cu_seqlens=cu_seqlens,
            seq_lens=seq_lens,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            stream=stream,
            flash_attn_varlen_func=flash_attn_varlen_func,
        )
    return _run_paged(
        q,
        k,
        v,
        causal=causal,
        sm_scale=sm_scale,
        out=out,
        block_table=block_table,
        seq_lens_kv=seq_lens_kv,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        stream=stream,
        flash_attn_varlen_func=flash_attn_varlen_func,
    )


def _run_dense(q, k, v, *, causal, sm_scale, out, stream, flash_attn_varlen_func):
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            f"dense path expects BSHD q/k/v with ndim=4; got q{tuple(q.shape)} " f"k{tuple(k.shape)} v{tuple(v.shape)}"
        )
    b, sq, h, d = q.shape
    b2, skv, hkv, d2 = k.shape
    if (b2, d2) != (b, d) or v.shape != k.shape:
        raise ValueError(f"dense k/v shape mismatch: k{tuple(k.shape)} v{tuple(v.shape)} vs q{tuple(q.shape)}")
    if h % hkv != 0:
        raise ValueError(f"H ({h}) must be divisible by Hkv ({hkv})")
    if causal and sq != skv:
        raise ValueError(f"is_causal=True requires Sq == Skv (self-attention); got Sq={sq}, Skv={skv}")

    q_pack = _pack_bshd_as_varlen(q)
    k_pack = _pack_bshd_as_varlen(k)
    v_pack = _pack_bshd_as_varlen(v)
    cu_q = _dense_cu_seqlens(batch=b, seqlen=sq, device=q.device)
    cu_k = _dense_cu_seqlens(batch=b, seqlen=skv, device=q.device)
    o_pack = flash_attn_varlen_func(
        q_pack,
        k_pack,
        v_pack,
        cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=sq,
        max_seqlen_k=skv,
        softmax_scale=sm_scale,
        causal=causal,
        stream=stream,
    )
    o_logic = o_pack.reshape(b, sq, h, d)
    if out is not None:
        out.copy_(o_logic)
        return out
    return o_logic


def _run_varlen(
    q,
    k,
    v,
    *,
    causal,
    sm_scale,
    out,
    cu_seqlens,
    seq_lens,
    max_seqlen_q,
    max_seqlen_kv,
    stream,
    flash_attn_varlen_func,
):
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError(
            f"varlen path expects packed q/k/v with ndim=3; got q{tuple(q.shape)} "
            f"k{tuple(k.shape)} v{tuple(v.shape)}"
        )
    total, h, d = q.shape
    total_k, hkv, d_k = k.shape
    if v.shape != k.shape:
        raise ValueError(f"varlen v must match k shape {tuple(k.shape)}, got {tuple(v.shape)}")
    if (total_k, d_k) != (total, d):
        raise ValueError(f"varlen k shape {tuple(k.shape)} incompatible with q {tuple(q.shape)}")
    if h % hkv != 0:
        raise ValueError(f"H ({h}) must be divisible by Hkv ({hkv})")

    import torch

    sl = seq_lens
    if sl.dtype != torch.int32:
        raise ValueError(f"seq_lens must be int32, got {sl.dtype}")
    num_seqs = int(sl.numel())
    if cu_seqlens.numel() != num_seqs + 1:
        raise ValueError(f"cu_seqlens length must be B+1={num_seqs + 1}, got {cu_seqlens.numel()}")

    max_s = int(sl.max().item()) if max_seqlen_q is None else int(max_seqlen_q)
    if max_seqlen_kv is not None and int(max_seqlen_kv) != max_s:
        raise ValueError(f"varlen self-attn requires max_seqlen_q == max_seqlen_kv; got {max_s} vs {max_seqlen_kv}")
    if max_seqlen_kv is None:
        max_seqlen_kv = max_s

    cu = _i32_on(cu_seqlens, q.device)
    seqused = _i32_on(sl, q.device)
    # FA checks total_q <= batch * max_seqlen_q. Flex packs may keep unused
    # tail after cu_seqlens[-1]; slice to the addressed prefix and keep the
    # caller's full [total, H, D] output buffer.
    end = int(cu[-1].item())
    if end < 0 or end > q.shape[0]:
        raise ValueError(f"cu_seqlens[-1]={end} exceeds packed q length {q.shape[0]}")
    q_run = q[:end].contiguous()
    k_run = k[:end].contiguous()
    v_run = v[:end].contiguous()
    o_full = torch.zeros_like(q) if out is None else out
    if o_full.shape != q.shape:
        raise ValueError(f"out shape must match q {tuple(q.shape)}, got {tuple(o_full.shape)}")
    o_run = o_full[:end]
    flash_attn_varlen_func(
        q_run,
        k_run,
        v_run,
        cu,
        cu_seqlens_k=cu,
        max_seqlen_q=max_s,
        max_seqlen_k=int(max_seqlen_kv),
        softmax_scale=sm_scale,
        causal=causal,
        seqused_k=seqused,
        out=o_run,
        stream=stream,
    )
    return o_full


def _run_paged(
    q,
    k,
    v,
    *,
    causal,
    sm_scale,
    out,
    block_table,
    seq_lens_kv,
    max_seqlen_q,
    max_seqlen_kv,
    stream,
    flash_attn_varlen_func,
):
    import torch

    if q.dim() != 4:
        raise ValueError(f"paged path expects BSHD q with ndim=4, got {tuple(q.shape)}")
    if k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            f"paged path expects K/V pages [NumBlocks, {_PAGED_PAGE}, Hkv, D], got k{tuple(k.shape)} v{tuple(v.shape)}"
        )
    b, sq, h, d = q.shape
    _, page, hkv, d_k = k.shape
    if page != _PAGED_PAGE or d_k != d or v.shape != k.shape:
        raise ValueError(
            f"paged K/V must be same-shaped [NumBlocks, {_PAGED_PAGE}, Hkv, D]; "
            f"got k{tuple(k.shape)} v{tuple(v.shape)}"
        )
    if h % hkv != 0:
        raise ValueError(f"H ({h}) must be divisible by Hkv ({hkv})")
    if block_table.size(0) != b or seq_lens_kv.numel() != b:
        raise ValueError(
            f"block_table/seq_lens_kv batch must match q batch {b}; "
            f"got bt={tuple(block_table.shape)} sl={tuple(seq_lens_kv.shape)}"
        )
    if seq_lens_kv.dtype != torch.int32:
        raise ValueError(f"seq_lens_kv must be int32, got {seq_lens_kv.dtype}")

    sq_compile = int(max_seqlen_q) if max_seqlen_q is not None else sq
    if sq > sq_compile:
        raise ValueError(f"q seqlen {sq} exceeds max_seqlen_q={sq_compile}")

    q_pack = _pack_bshd_as_varlen(q)
    cu_q = _dense_cu_seqlens(batch=b, seqlen=sq, device=q.device)
    seqused = _i32_on(seq_lens_kv, q.device)
    bt = _i32_on(block_table, q.device)
    max_k = int(seqused.max().item()) if max_seqlen_kv is None else int(max_seqlen_kv)

    o_pack = flash_attn_varlen_func(
        q_pack,
        k.contiguous(),
        v.contiguous(),
        cu_q,
        max_seqlen_q=sq,
        max_seqlen_k=max_k,
        softmax_scale=sm_scale,
        causal=causal,
        block_table=bt,
        seqused_k=seqused,
        kv_cache_layout="NHD",
        stream=stream,
    )
    o_logic = o_pack.reshape(b, sq, h, d)
    if out is not None:
        out.copy_(o_logic)
        return out
    return o_logic
