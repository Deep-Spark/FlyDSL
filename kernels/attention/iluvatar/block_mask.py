# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Host-side BlockMask for Iluvatar flex-attention (V3-3 / V3-6).

Builds a compact ``kv_num_blocks`` / ``kv_indices`` / ``kv_is_full`` table from
preset masks (causal / SWA) and an optional ``TracedMaskMod``. The attention
kernel iterates indices to skip EMPTY tiles; FULL tiles may skip element masks;
PARTIAL tiles apply preset ∧ ``mask_mod`` element holes.

V3-6 adds packed-varlen helpers: one ``FlexBlockMask`` per sequence plus a
batched pack for launch (``[num_seqs, max_q_tiles, ...]``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

from flydsl.expr.trace_mod import TracedMaskMod

__all__ = [
    "FlexBlockMask",
    "PackedVarlenBlockMask",
    "create_block_mask",
    "create_block_masks_varlen",
    "pack_block_masks_varlen",
]


def _phys_seq(seq: int, block: int) -> int:
    return ((int(seq) + int(block) - 1) // int(block)) * int(block)


def _element_visible(
    q: int,
    kv: int,
    *,
    Sq: int,
    Skv: int,
    is_causal: bool,
    window_size: int | None,
    mask_mod: Optional[TracedMaskMod],
) -> bool:
    if q < 0 or kv < 0 or q >= Sq or kv >= Skv:
        return False
    if is_causal and kv > q:
        return False
    if window_size is not None and (q - kv) > int(window_size):
        return False
    if mask_mod is not None:
        # V3-3: mask is shared across batch/head; evaluate at (0, 0).
        if not bool(mask_mod.eval_host(0, 0, q, kv)):
            return False
    return True


@dataclass(frozen=True)
class FlexBlockMask:
    """Dense-forward BlockMask (b/h-shared; not bit-identical to PyTorch)."""

    block_m: int
    block_n: int
    Sq: int
    Skv: int
    num_q_tiles: int
    num_kv_tiles: int
    kv_num_blocks: Any  # int32 [num_q_tiles]
    kv_indices: Any  # int32 [num_q_tiles, num_kv_tiles]
    kv_is_full: Any  # int32 [num_q_tiles, num_kv_tiles] (1=FULL, 0=PARTIAL)

    def sparsity(self) -> float:
        """Fraction of KV tiles that are EMPTY (skipped), in ``[0, 1]``."""
        total = int(self.num_q_tiles) * int(self.num_kv_tiles)
        if total <= 0:
            return 0.0
        kept = int(self.kv_num_blocks.sum().item())
        return float(total - kept) / float(total)


def create_block_mask(
    mask_mod: Optional[TracedMaskMod],
    B: int,
    H: int,
    Sq: int,
    Skv: int,
    *,
    block_m: int,
    block_n: int,
    is_causal: bool = False,
    window_size: int | None = None,
    device=None,
):
    """Host-build a ``FlexBlockMask`` from presets ∧ optional ``mask_mod``.

    Args:
        mask_mod: ``TracedMaskMod`` or ``None`` (presets only). Must not depend
            on batch/head for this knife (evaluated at ``(0, 0)``).
        B, H: Accepted for API symmetry; table is shared across batch/head.
            ``H>1`` triggers a head-independence spot-check when ``mask_mod`` set.
        Sq, Skv: Logical sequence lengths.
        block_m, block_n: Must match the attention compile tile.
        is_causal, window_size: Preset masks combined with ``mask_mod``.
        device: Torch device for the output tensors (default CUDA if available).

    Returns:
        ``FlexBlockMask`` with phys-tiled FULL / PARTIAL indices; pad is invisible.
    """
    if mask_mod is not None and not isinstance(mask_mod, TracedMaskMod):
        raise TypeError(f"mask_mod must be TracedMaskMod or None, got {type(mask_mod).__name__}")
    if window_size is not None and int(window_size) <= 0:
        raise ValueError(f"window_size must be > 0 when set, got {window_size}")
    if int(block_m) <= 0 or int(block_n) <= 0:
        raise ValueError(f"block_m/block_n must be positive, got {block_m}x{block_n}")

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("create_block_mask requires torch") from exc

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _ = int(B)  # API symmetry; shared table ignores batch
    H_i = int(H)
    Sq = int(Sq)
    Skv = int(Skv)
    block_m = int(block_m)
    block_n = int(block_n)
    Sq_phys = _phys_seq(Sq, block_m)
    Skv_phys = _phys_seq(Skv, block_n)
    num_q_tiles = Sq_phys // block_m
    num_kv_tiles = Skv_phys // block_n

    kv_num = torch.zeros(num_q_tiles, dtype=torch.int32, device=device)
    kv_indices = torch.full((num_q_tiles, num_kv_tiles), -1, dtype=torch.int32, device=device)
    kv_is_full = torch.zeros((num_q_tiles, num_kv_tiles), dtype=torch.int32, device=device)

    if mask_mod is not None and H_i > 1:
        for q, kv in ((0, 0), (max(Sq - 1, 0), 0), (0, max(Skv - 1, 0))):
            if q >= Sq or kv >= Skv:
                continue
            v00 = bool(mask_mod.eval_host(0, 0, q, kv))
            v01 = bool(mask_mod.eval_host(0, min(1, H_i - 1), q, kv))
            if v00 != v01:
                raise ValueError(
                    "V3-3 BlockMask requires mask_mod independent of head; "
                    f"got different results at head 0 vs 1 for (q,kv)=({q},{kv})"
                )

    for qi in range(num_q_tiles):
        q0 = qi * block_m
        q1 = min(q0 + block_m, Sq_phys)
        out_n = 0
        for kj in range(num_kv_tiles):
            k0 = kj * block_n
            k1 = min(k0 + block_n, Skv_phys)
            any_vis = False
            all_vis = True
            for q in range(q0, q1):
                for kv in range(k0, k1):
                    vis = _element_visible(
                        q,
                        kv,
                        Sq=Sq,
                        Skv=Skv,
                        is_causal=bool(is_causal),
                        window_size=window_size,
                        mask_mod=mask_mod,
                    )
                    any_vis = any_vis or vis
                    all_vis = all_vis and vis
            if not any_vis:
                continue
            is_full = bool(all_vis) and (q1 <= Sq) and (k1 <= Skv)
            kv_indices[qi, out_n] = kj
            kv_is_full[qi, out_n] = 1 if is_full else 0
            out_n += 1
        kv_num[qi] = out_n

    return FlexBlockMask(
        block_m=block_m,
        block_n=block_n,
        Sq=Sq,
        Skv=Skv,
        num_q_tiles=num_q_tiles,
        num_kv_tiles=num_kv_tiles,
        kv_num_blocks=kv_num,
        kv_indices=kv_indices,
        kv_is_full=kv_is_full,
    )


@dataclass(frozen=True)
class PackedVarlenBlockMask:
    """Batched BlockMask tables for packed-varlen launch (V3-6).

    Short sequences are right-padded to ``max_q_tiles`` / ``max_kv_tiles``; pad
    q-tiles keep ``kv_num_blocks == 0`` so the kernel skips them.
    """

    block_m: int
    block_n: int
    max_q_tiles: int
    max_kv_tiles: int
    num_seqs: int
    kv_num_blocks: Any  # int32 [num_seqs, max_q_tiles]
    kv_indices: Any  # int32 [num_seqs, max_q_tiles, max_kv_tiles]
    kv_is_full: Any  # int32 [num_seqs, max_q_tiles, max_kv_tiles]


def create_block_masks_varlen(
    mask_mod: Optional[TracedMaskMod],
    seq_lens: Sequence[int],
    *,
    block_m: int,
    block_n: int,
    H: int = 1,
    is_causal: bool = False,
    window_size: int | None = None,
    device=None,
) -> list[FlexBlockMask]:
    """Build one logical ``FlexBlockMask`` per sequence (self-attn).

    Each entry uses logical ``Sq=Skv=seq_lens[i]`` (pad tokens invisible), matching
    dense ``create_block_mask`` semantics. Call ``pack_block_masks_varlen`` before
    a varlen launch that needs a single batched table.
    """
    if not seq_lens:
        raise ValueError("seq_lens must be non-empty")
    masks: list[FlexBlockMask] = []
    for s in seq_lens:
        s_i = int(s)
        if s_i < 0:
            raise ValueError(f"seq_lens entries must be non-negative, got {s}")
        masks.append(
            create_block_mask(
                mask_mod,
                B=1,
                H=H,
                Sq=s_i,
                Skv=s_i,
                block_m=block_m,
                block_n=block_n,
                is_causal=is_causal,
                window_size=window_size,
                device=device,
            )
        )
    return masks


def pack_block_masks_varlen(
    masks: Sequence[FlexBlockMask],
    *,
    max_seqlen_q: int | None = None,
    max_seqlen_kv: int | None = None,
) -> PackedVarlenBlockMask:
    """Pack per-seq masks into ``[num_seqs, max_q_tiles, ...]`` launch tensors.

    ``max_seqlen_*`` default to the max logical length across ``masks``. Tile
    counts follow the same phys-round as ``create_block_mask`` / compile.
    """
    if not masks:
        raise ValueError("masks must be non-empty")
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("pack_block_masks_varlen requires torch") from exc

    block_m = int(masks[0].block_m)
    block_n = int(masks[0].block_n)
    for i, m in enumerate(masks):
        if int(m.block_m) != block_m or int(m.block_n) != block_n:
            raise ValueError(f"masks[{i}] tile {m.block_m}x{m.block_n} != masks[0] {block_m}x{block_n}")

    max_sq = max(int(m.Sq) for m in masks)
    max_skv = max(int(m.Skv) for m in masks)
    max_seqlen_q = int(max_seqlen_q) if max_seqlen_q is not None else max_sq
    max_seqlen_kv = int(max_seqlen_kv) if max_seqlen_kv is not None else max_skv
    if max_seqlen_q < max_sq or max_seqlen_kv < max_skv:
        raise ValueError(
            f"max_seqlen_q/kv ({max_seqlen_q},{max_seqlen_kv}) must cover " f"mask logical max ({max_sq},{max_skv})"
        )

    max_q_tiles = _phys_seq(max_seqlen_q, block_m) // block_m
    max_kv_tiles = _phys_seq(max_seqlen_kv, block_n) // block_n
    num_seqs = len(masks)
    device = masks[0].kv_num_blocks.device

    kv_num = torch.zeros((num_seqs, max_q_tiles), dtype=torch.int32, device=device)
    kv_indices = torch.full((num_seqs, max_q_tiles, max_kv_tiles), -1, dtype=torch.int32, device=device)
    kv_is_full = torch.zeros((num_seqs, max_q_tiles, max_kv_tiles), dtype=torch.int32, device=device)

    for seq_id, m in enumerate(masks):
        nq = int(m.num_q_tiles)
        nk = int(m.num_kv_tiles)
        if nq > max_q_tiles or nk > max_kv_tiles:
            raise ValueError(f"masks[{seq_id}] tiles ({nq},{nk}) exceed pack " f"({max_q_tiles},{max_kv_tiles})")
        kv_num[seq_id, :nq].copy_(m.kv_num_blocks)
        kv_indices[seq_id, :nq, :nk].copy_(m.kv_indices)
        kv_is_full[seq_id, :nq, :nk].copy_(m.kv_is_full)
        # Rows [nq, max_q_tiles) stay kv_num_blocks=0 (inactive q-tiles).

    return PackedVarlenBlockMask(
        block_m=block_m,
        block_n=block_n,
        max_q_tiles=max_q_tiles,
        max_kv_tiles=max_kv_tiles,
        num_seqs=num_seqs,
        kv_num_blocks=kv_num,
        kv_indices=kv_indices,
        kv_is_full=kv_is_full,
    )
