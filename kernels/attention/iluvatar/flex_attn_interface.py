# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""High-level Iluvatar flex-attention API.

Wraps ``compile_iluvatar_flex_attention`` behind a single Torch entry:

    ``flydsl_flex_attn_func(q, k, v, ...)``

Selects dense / varlen / paged from optional metadata, caches compiled
launchers (fingerprint-keyed LRU), and owns dense/paged phys-pad + V
transpose so callers can pass logical BSHD (or packed varlen / natural
paged pages). Dense and varlen accept ``score_mod``; dense uses
``mask_mod`` / ``block_mask``, varlen uses ``mask_mod`` /
``block_masks`` (V3-6).

Also exposes ``autotune_iluvatar_flex_attention_tile`` for opt-in dense-only
tile search (returns a ``tile_config`` dict; does not alter the default path).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping, Optional, Sequence

import flydsl.expr as fx
from flydsl.expr.trace_mod import TracedMaskMod, TracedScoreMod
from kernels.attention.iluvatar.block_mask import FlexBlockMask, pack_block_masks_varlen
from kernels.attention.iluvatar.flex_attention import (
    _SUPPORTED_BLOCK,
    _normalize_tile_config,
    compile_iluvatar_flex_attention,
)

__all__ = ["flydsl_flex_attn_func", "autotune_iluvatar_flex_attention_tile"]

# Manual LRU: Traced* objects are not safely hashable for lru_cache
# (callable + dict closure); keys use stable fingerprints instead.
_LAUNCHER_CACHE: OrderedDict = OrderedDict()
_LAUNCHER_CACHE_MAX = 256


def _torch():
    import torch

    return torch


def _dtype_str(t) -> str:
    torch = _torch()
    mapping = {torch.bfloat16: "bf16", torch.float16: "f16"}
    s = mapping.get(t.dtype)
    if s is None:
        raise ValueError(f"flydsl_flex_attn_func only supports bf16/f16, got {t.dtype!r}")
    return s


def _phys_seq(seq: int, block: int) -> int:
    return ((int(seq) + int(block) - 1) // int(block)) * int(block)


def _as_fx_stream(stream):
    if stream is None:
        return fx.Stream(None)
    if isinstance(stream, fx.Stream):
        return stream
    return fx.Stream(stream)


def _resolve_mode(*, cu_seqlens, seq_lens, block_table, seq_lens_kv) -> str:
    has_varlen = cu_seqlens is not None or seq_lens is not None
    has_paged = block_table is not None or seq_lens_kv is not None
    if has_varlen and has_paged:
        raise ValueError(
            "flydsl_flex_attn_func: varlen (cu_seqlens/seq_lens) and paged "
            "(block_table/seq_lens_kv) are mutually exclusive"
        )
    if has_paged:
        if block_table is None or seq_lens_kv is None:
            raise ValueError("paged path requires both block_table [B, max_num_pages] and seq_lens_kv [B]")
        return "paged"
    if has_varlen:
        if cu_seqlens is None or seq_lens is None:
            raise ValueError("varlen path requires both cu_seqlens [B+1] and seq_lens [B]")
        return "varlen"
    return "dense"


def _default_tile_configs() -> list[dict[str, int]]:
    return [{"block_m": bm, "block_n": bn} for bm in _SUPPORTED_BLOCK for bn in _SUPPORTED_BLOCK]


def _build_launcher(
    B: int,
    H: int,
    Sq: int,
    Skv: int,
    D: int,
    Hkv: int,
    dtype: str,
    is_causal: bool,
    window_size: Optional[int],
    softcap: Optional[float],
    sm_scale: Optional[float],
    varlen: bool,
    paged: bool,
    has_alibi: bool,
    has_score_bias: bool,
    block_m: int,
    block_n: int,
    score_mod: Optional[TracedScoreMod] = None,
    mask_mod: Optional[TracedMaskMod] = None,
    has_block_mask: bool = False,
):
    sm_fp = score_mod.fingerprint if score_mod is not None else ""
    mm_fp = mask_mod.fingerprint if mask_mod is not None else ""
    key = (
        B,
        H,
        Sq,
        Skv,
        D,
        Hkv,
        dtype,
        is_causal,
        window_size,
        softcap,
        sm_scale,
        varlen,
        paged,
        has_alibi,
        has_score_bias,
        block_m,
        block_n,
        sm_fp,
        mm_fp,
        bool(has_block_mask),
    )
    hit = _LAUNCHER_CACHE.get(key)
    if hit is not None:
        _LAUNCHER_CACHE.move_to_end(key)
        return hit
    launch = compile_iluvatar_flex_attention(
        B,
        H,
        Sq,
        Skv,
        D,
        Hkv=Hkv,
        dtype=dtype,
        is_causal=is_causal,
        window_size=window_size,
        softcap=softcap,
        sm_scale=sm_scale,
        varlen=varlen,
        paged=paged,
        has_alibi=has_alibi,
        has_score_bias=has_score_bias,
        tile_config={"block_m": block_m, "block_n": block_n},
        score_mod=score_mod,
        mask_mod=mask_mod,
        has_block_mask=bool(has_block_mask),
    )
    _LAUNCHER_CACHE[key] = launch
    while len(_LAUNCHER_CACHE) > _LAUNCHER_CACHE_MAX:
        _LAUNCHER_CACHE.popitem(last=False)
    return launch


def _bshd_to_bhsd(x):
    """[B, S, H, D] -> [B, H, S, D]."""
    return x.permute(0, 2, 1, 3).contiguous()


def _bhsd_to_bshd(x):
    """[B, H, S, D] -> [B, S, H, D]."""
    return x.permute(0, 2, 1, 3).contiguous()


def _pad_bhsd(x_bhsd, logical_s: int, block: int):
    """Pad sequence dim of BHSD tensor to a multiple of ``block``."""
    torch = _torch()
    b, h, s, d = x_bhsd.shape
    if s != logical_s:
        raise ValueError(f"expected logical seqlen {logical_s} on dim 2, got {s}")
    phys = _phys_seq(logical_s, block)
    if s == phys:
        return x_bhsd.contiguous()
    out = torch.zeros(b, h, phys, d, device=x_bhsd.device, dtype=x_bhsd.dtype)
    out[:, :, :logical_s].copy_(x_bhsd)
    return out


def _transpose_v_bhsd(v_bhsd):
    """[B, Hkv, Skv, D] -> [B, Hkv, D, Skv] for MMA2."""
    return v_bhsd.transpose(-1, -2).contiguous()


def _transpose_v_pages(v_pages):
    """[NumBlocks, page, Hkv, D] -> [NumBlocks, Hkv, D, page]."""
    return v_pages.permute(0, 2, 3, 1).contiguous()


def _check_varlen_alignment(cu_seqlens, total: int) -> None:
    torch = _torch()
    cu = cu_seqlens
    if cu.dtype != torch.int32:
        raise ValueError(f"cu_seqlens must be int32, got {cu.dtype}")
    if cu.device.type != "cpu":
        cu_host = cu.detach().cpu()
    else:
        cu_host = cu
    for i, v in enumerate(cu_host.tolist()):
        if int(v) % 32 != 0:
            raise ValueError(f"varlen cu_seqlens[{i}]={v} must be a multiple of 32 for SME G2S")
    if total % 32 != 0:
        raise ValueError(f"varlen total_tokens={total} must be a multiple of 32 for SME G2S")


def flydsl_flex_attn_func(
    q,
    k,
    v,
    *,
    causal: bool = False,
    window_size: Optional[int] = None,
    softcap: Optional[float] = None,
    alibi_slopes=None,
    score_bias=None,
    score_mod=None,
    mask_mod=None,
    block_mask=None,
    block_masks=None,
    sm_scale: Optional[float] = None,
    out=None,
    # Varlen (packed): both required to select the varlen path.
    cu_seqlens=None,
    seq_lens=None,
    # Paged KV: both required to select the paged path.
    block_table=None,
    seq_lens_kv=None,
    # Optional compile-time upper bounds for varlen/paged (defaults: current max).
    max_seqlen_q: Optional[int] = None,
    max_seqlen_kv: Optional[int] = None,
    tile_config: Mapping[str, int] | None = None,
    stream=None,
):
    """Run Iluvatar flex-attention (dense / varlen / paged).

    Mode selection (mutually exclusive):
      * ``block_table`` + ``seq_lens_kv`` -> paged
      * ``cu_seqlens`` + ``seq_lens`` -> varlen
      * otherwise -> dense

    Layouts:
      * Dense / paged Q: BSHD ``q: [B, Sq, H, D]``, ``k/v: [B, Skv, Hkv, D]``
        (paged K/V are linear pages ``[NumBlocks, 64, Hkv, D]`` instead).
      * Varlen: packed ``q/o: [total, H, D]``, ``k: [total, Hkv, D]``,
        ``v: [total, Hkv, D]`` (natural; transposed inside). Physical
        ``cu_seqlens`` must already be 32-aligned (no re-pack).

    ``tile_config`` is optional ``{"block_m", "block_n"}`` with values in
    ``{32, 64}`` (default 64x64). Paged requires ``block_n=64``.

    Dense and varlen may pass ``score_mod`` (``TracedScoreMod``). Dense:
    ``mask_mod`` and/or ``block_mask`` (``FlexBlockMask``). Varlen:
    ``mask_mod`` and/or ``block_masks`` (``Sequence[FlexBlockMask]``, one per
    seq). Paged rejects all mods. If a BlockMask is set and ``tile_config`` is
    omitted, the mask tile is used.

    Returns:
        Output in the same layout as ``q`` (logical lengths; no phys pad).
    """
    torch = _torch()
    mode = _resolve_mode(
        cu_seqlens=cu_seqlens,
        seq_lens=seq_lens,
        block_table=block_table,
        seq_lens_kv=seq_lens_kv,
    )

    if score_mod is not None and not isinstance(score_mod, TracedScoreMod):
        raise ValueError(f"score_mod must be TracedScoreMod or None, got {type(score_mod).__name__}")
    if mask_mod is not None and not isinstance(mask_mod, TracedMaskMod):
        raise ValueError(f"mask_mod must be TracedMaskMod or None, got {type(mask_mod).__name__}")
    if block_mask is not None and block_masks is not None:
        raise ValueError("pass either block_mask (dense) or block_masks (varlen), not both")
    if block_mask is not None and not isinstance(block_mask, FlexBlockMask):
        raise TypeError(f"block_mask must be FlexBlockMask or None, got {type(block_mask).__name__}")
    if block_masks is not None:
        if not isinstance(block_masks, (list, tuple)) or not block_masks:
            raise TypeError("block_masks must be a non-empty Sequence[FlexBlockMask]")
        if not all(isinstance(m, FlexBlockMask) for m in block_masks):
            raise TypeError("block_masks entries must be FlexBlockMask")
    if mode == "paged" and (
        score_mod is not None or mask_mod is not None or block_mask is not None or block_masks is not None
    ):
        raise ValueError("score_mod/mask_mod/block_mask(s) are not supported with paged")
    if mode == "dense" and block_masks is not None:
        raise ValueError("block_masks is varlen-only; use block_mask for dense")
    if mode == "varlen" and block_mask is not None:
        raise ValueError("varlen uses block_masks=Sequence[FlexBlockMask], not block_mask=")

    tile_src = block_mask if block_mask is not None else (block_masks[0] if block_masks else None)
    if tile_src is not None and tile_config is None:
        tile_config = {
            "block_m": int(tile_src.block_m),
            "block_n": int(tile_src.block_n),
        }

    dtype = _dtype_str(q)
    fxs = _as_fx_stream(stream)
    block_m, block_n = _normalize_tile_config(tile_config, paged=(mode == "paged"))
    if tile_src is not None and (int(tile_src.block_m) != int(block_m) or int(tile_src.block_n) != int(block_n)):
        raise ValueError(
            f"block_mask tile ({tile_src.block_m}x{tile_src.block_n}) must match " f"tile_config ({block_m}x{block_n})"
        )

    if mode == "dense":
        if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
            raise ValueError(
                f"dense path expects BSHD q/k/v with ndim=4; got q{tuple(q.shape)} "
                f"k{tuple(k.shape)} v{tuple(v.shape)}"
            )
        b, sq, h, d = q.shape
        b2, skv, hkv, d2 = k.shape
        if (b2, d2) != (b, d) or v.shape != k.shape:
            raise ValueError(f"dense k/v shape mismatch: k{tuple(k.shape)} v{tuple(v.shape)} vs q{tuple(q.shape)}")
        if h % hkv != 0:
            raise ValueError(f"H ({h}) must be divisible by Hkv ({hkv})")

        if alibi_slopes is not None and score_bias is not None:
            raise ValueError("alibi_slopes and score_bias are mutually exclusive")
        has_alibi = alibi_slopes is not None
        has_score_bias = score_bias is not None
        has_bm = block_mask is not None
        launch = _build_launcher(
            b,
            h,
            sq,
            skv,
            d,
            hkv,
            dtype,
            bool(causal),
            window_size,
            softcap,
            sm_scale,
            False,
            False,
            has_alibi,
            has_score_bias,
            block_m,
            block_n,
            score_mod=score_mod,
            mask_mod=mask_mod,
            has_block_mask=has_bm,
        )
        q_bhsd = _pad_bhsd(_bshd_to_bhsd(q), sq, block_m)
        k_bhsd = _pad_bhsd(_bshd_to_bhsd(k), skv, block_n)
        v_tn = _transpose_v_bhsd(_pad_bhsd(_bshd_to_bhsd(v), skv, block_n))
        o_bhsd = torch.zeros_like(q_bhsd)
        sb_phys = None
        if has_score_bias:
            # Accept BSHD logical [B,Sq,H,Skv] or BHSD [B,H,Sq,Skv]; pad to phys.
            sb = score_bias
            if sb.dim() != 4:
                raise ValueError(f"score_bias must be 4D, got {tuple(sb.shape)}")
            if sb.shape[1] == h and sb.shape[2] == sq:
                sb_bhsd = sb
            elif sb.shape[1] == sq and sb.shape[2] == h:
                sb_bhsd = sb.permute(0, 2, 1, 3).contiguous()
            else:
                raise ValueError(f"score_bias shape {tuple(sb.shape)} incompatible with B={b} H={h} Sq={sq} Skv={skv}")
            if sb_bhsd.shape[0] not in (1, b) or sb_bhsd.shape[3] != skv:
                raise ValueError(f"score_bias shape {tuple(sb.shape)} incompatible with batch/Skv")
            if sb_bhsd.shape[0] == 1 and b > 1:
                sb_bhsd = sb_bhsd.expand(b, -1, -1, -1)
            sq_phys = _phys_seq(sq, block_m)
            skv_phys = _phys_seq(skv, block_n)
            sb_phys = torch.zeros(b, h, sq_phys, skv_phys, device=sb.device, dtype=torch.float32)
            sb_phys[:, :, :sq, :skv].copy_(sb_bhsd.float())
        ali = alibi_slopes.float().contiguous() if has_alibi else None
        launch(
            q_bhsd,
            k_bhsd,
            v_tn,
            o_bhsd,
            alibi_slopes=ali,
            score_bias=sb_phys,
            block_mask=block_mask,
            stream=fxs,
        )
        o_logic = _bhsd_to_bshd(o_bhsd[:, :, :sq, :])
        if out is not None:
            out.copy_(o_logic)
            return out
        return o_logic

    if mode == "varlen":
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
        if alibi_slopes is not None or score_bias is not None:
            raise ValueError("alibi/score_bias are dense-only (not supported with varlen)")

        sl = seq_lens
        if sl.dtype != torch.int32:
            raise ValueError(f"seq_lens must be int32, got {sl.dtype}")
        num_seqs = int(sl.numel())
        if cu_seqlens.numel() != num_seqs + 1:
            raise ValueError(f"cu_seqlens length must be B+1={num_seqs + 1}, got {cu_seqlens.numel()}")
        _check_varlen_alignment(cu_seqlens, total)

        max_s = int(sl.max().item()) if max_seqlen_q is None else int(max_seqlen_q)
        if max_seqlen_kv is not None and int(max_seqlen_kv) != max_s:
            raise ValueError(
                f"varlen self-attn requires max_seqlen_q == max_seqlen_kv; " f"got {max_s} vs {max_seqlen_kv}"
            )
        if max_seqlen_kv is None:
            max_seqlen_kv = max_s

        launch = _build_launcher(
            num_seqs,
            h,
            max_s,
            int(max_seqlen_kv),
            d,
            hkv,
            dtype,
            bool(causal),
            window_size,
            softcap,
            sm_scale,
            True,
            False,
            False,
            False,
            block_m,
            block_n,
            score_mod=score_mod,
            mask_mod=mask_mod,
            has_block_mask=block_masks is not None,
        )
        v_tn = v.permute(1, 2, 0).contiguous()  # [Hkv, D, total]
        o = torch.zeros_like(q) if out is None else out
        if o.shape != q.shape:
            raise ValueError(f"out shape must match q {tuple(q.shape)}, got {tuple(o.shape)}")
        packed_bm = None
        if block_masks is not None:
            if len(block_masks) != num_seqs:
                raise ValueError(f"block_masks length must equal num_seqs={num_seqs}, got {len(block_masks)}")
            packed_bm = pack_block_masks_varlen(block_masks, max_seqlen_q=max_s, max_seqlen_kv=int(max_seqlen_kv))
        launch(
            q,
            k,
            v_tn,
            o,
            cu_seqlens=cu_seqlens,
            seq_lens=seq_lens,
            block_mask=packed_bm,
            stream=fxs,
        )
        return o

    # paged
    if q.dim() != 4:
        raise ValueError(f"paged path expects BSHD q with ndim=4, got {tuple(q.shape)}")
    if k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            f"paged path expects K/V pages [NumBlocks, 64, Hkv, D], got k{tuple(k.shape)} v{tuple(v.shape)}"
        )
    b, sq, h, d = q.shape
    num_blocks, page, hkv, d_k = k.shape
    if page != block_n or d_k != d or v.shape != k.shape:
        raise ValueError(
            f"paged K/V must be same-shaped [NumBlocks, {block_n}, Hkv, D]; " f"got k{tuple(k.shape)} v{tuple(v.shape)}"
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
    if alibi_slopes is not None or score_bias is not None:
        raise ValueError("alibi/score_bias are dense-only (not supported with paged)")

    skv = int(seq_lens_kv.max().item()) if max_seqlen_kv is None else int(max_seqlen_kv)
    sq_compile = int(max_seqlen_q) if max_seqlen_q is not None else sq
    if sq > sq_compile:
        raise ValueError(f"q seqlen {sq} exceeds max_seqlen_q={sq_compile}")

    launch = _build_launcher(
        b,
        h,
        sq_compile,
        skv,
        d,
        hkv,
        dtype,
        bool(causal),
        window_size,
        softcap,
        sm_scale,
        False,
        True,
        False,
        False,
        block_m,
        block_n,
    )
    q_bhsd = _pad_bhsd(_bshd_to_bhsd(q), sq, block_m)
    # If compile Sq > runtime Sq, pad Q rows to compile Sq_phys already handled
    # by pad to block_m from logical sq; kernel masks via causal delta / seq_lens_kv.
    # When max_seqlen_q > sq, need pad to compile Sq_phys for launch shape check.
    sq_phys_compile = _phys_seq(sq_compile, block_m)
    if q_bhsd.size(2) != sq_phys_compile:
        q_pad = torch.zeros(b, h, sq_phys_compile, d, device=q.device, dtype=q.dtype)
        q_pad[:, :, : q_bhsd.size(2)].copy_(q_bhsd)
        q_bhsd = q_pad
    v_tn = _transpose_v_pages(v)
    o_bhsd = torch.zeros_like(q_bhsd)
    launch(
        q_bhsd,
        k.contiguous(),
        v_tn,
        o_bhsd,
        block_table=block_table,
        seq_lens_kv=seq_lens_kv,
        stream=fxs,
    )
    o_logic = _bhsd_to_bshd(o_bhsd[:, :, :sq, :])
    if out is not None:
        out.copy_(o_logic)
        return out
    return o_logic


def autotune_iluvatar_flex_attention_tile(
    q,
    k,
    v,
    *,
    causal: bool = False,
    window_size: Optional[int] = None,
    softcap: Optional[float] = None,
    sm_scale: Optional[float] = None,
    configs: Sequence[Mapping[str, int]] | None = None,
    warmup: int = 5,
    rep: int = 25,
    stream=None,
) -> dict[str, int]:
    """Opt-in dense-only tile search; returns the fastest ``tile_config``.

    Sweeps ``configs`` (default: all ``{32,64} x {32,64}``) via ``do_bench``,
    then returns ``{"block_m", "block_n"}`` for the median-fastest config.
    Does not write a disk artifact. Caller should pass the result back into
    ``flydsl_flex_attn_func(..., tile_config=...)``.
    """
    torch = _torch()
    from flydsl.autotune import do_bench

    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            "autotune_iluvatar_flex_attention_tile is dense-only; "
            f"expected BSHD q/k/v ndim=4, got q{tuple(q.shape)} k{tuple(k.shape)} v{tuple(v.shape)}"
        )

    tile_list = list(configs) if configs is not None else _default_tile_configs()
    if not tile_list:
        raise ValueError("configs must be a non-empty sequence of tile_config dicts")

    best_cfg: dict[str, int] | None = None
    best_ms = float("inf")
    fxs = _as_fx_stream(stream)

    for raw in tile_list:
        block_m, block_n = _normalize_tile_config(raw, paged=False)
        cfg = {"block_m": block_m, "block_n": block_n}

        # Allocate pads / out once per config (phys shape depends on tile).
        def _run(cfg=cfg):
            flydsl_flex_attn_func(
                q,
                k,
                v,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                sm_scale=sm_scale,
                tile_config=cfg,
                stream=fxs,
            )

        # Force one compile/warmup launch before timing so JIT cost is excluded.
        _run()
        torch.cuda.synchronize()
        ms = float(do_bench(_run, warmup=max(0, int(warmup) - 1), rep=int(rep)))
        if ms < best_ms:
            best_ms = ms
            best_cfg = cfg

    assert best_cfg is not None
    return dict(best_cfg)
