"""IXDL (Iluvatar) intrinsic wrappers for ivcore11.

Uses native IXDL dialect ops (auto-generated from IXDLOps.td) where possible,
falls back to ``llvm.call_intrinsic`` for ops not in the IXDL dialect
(e.g. descriptor-based store/load).

Usage::

    from flydsl.expr import ixdl

    ixdl.sme_load(slb_addr, desc, g_offset, kop, shape=[16, 1], element_size=16)
    ixdl.barrier()
    acc = ixdl.mmad_f16(a_v4f16, b_v4f16, c_v4f32)
    ixdl.cp_async_commit_group()
    ixdl.cp_async_wait_group(0)
"""

from .._mlir import ir
from .._mlir.dialects import llvm, arith as _arith
from .._mlir.dialects import ixdl as _ixdl_dialect


# ═══════════════════════════════════════════════════════════════════
# Type helpers
# ═══════════════════════════════════════════════════════════════════

def _i32():
    return ir.IntegerType.get_signless(32)


def _i64():
    return ir.IntegerType.get_signless(64)


def _v4i32():
    return ir.VectorType.get([4], _i32())


def _v4f16():
    return ir.VectorType.get([4], ir.F16Type.get())


def _v4f32():
    return ir.VectorType.get([4], ir.F32Type.get())


def _i32_attr(val):
    return ir.IntegerAttr.get(_i32(), val)


def _i32_const(val):
    return _arith.ConstantOp(_i32(), _i32_attr(val)).result


def _call_void_intrinsic(name, args, *, loc=None, ip=None):
    llvm.call_intrinsic(None, name, args, [], [], loc=loc, ip=ip)


def _call_intrinsic(result_type, name, args, *, loc=None, ip=None):
    return llvm.call_intrinsic(result_type, name, args, [], [], loc=loc, ip=ip)


# ═══════════════════════════════════════════════════════════════════
# G2S: SME async copy  (ixdl.cp.async)
#
# The native op automatically selects the correct intrinsic
# based on (shape, elementSize, transpose) attributes.
# ═══════════════════════════════════════════════════════════════════

def sme_load(slb_addr, desc, g_offset, kop,
             shape, element_size, transpose=False,
             *, loc=None, ip=None):
    """Generic SME load using native ``ixdl.cp.async`` op.

    Args:
        slb_addr:      i32 — SLB (shared memory) byte offset
        desc:          v4i32 — SME 2D descriptor
        g_offset:      i32 — global memory offset in bytes
        kop:           i32 — cache operation hint
        shape:         list[int] — e.g. [16, 1] for 16x1b64
        element_size:  int — 16 for FP16, 8 for INT8, 32 for FP32
        transpose:     bool — True for col-major (colxf), False for row-major (rowxf)
    """
    _ixdl_dialect.cp_async(
        slb_addr, desc, g_offset, kop,
        shape=shape,
        element_size=element_size,
        transpose=transpose,
        loc=loc, ip=ip,
    )


def sme_load_16x1b64_rowxfb16(slb_addr, desc, g_offset, kop,
                               *, loc=None, ip=None):
    """16 rows × 64 bytes, FP16, row-major. Convenience wrapper.

    shape[1] = 32 (FP16 elements per row) because the IXDL dialect
    transforms: shape[1] = shape[1] * elementSize / 512.
    For 1 b64 segment: 32 * 16 / 512 = 1.
    """
    sme_load(slb_addr, desc, g_offset, kop,
             shape=[16, 32], element_size=16, transpose=False,
             loc=loc, ip=ip)


def sme_load_16x1b64_colxfb16(slb_addr, desc, g_offset, kop,
                               *, loc=None, ip=None):
    """16 rows × 64 bytes, FP16, col-major transpose.

    For transpose, shape[1] is overwritten to shape[0]*elementSize/512,
    so shape[0]=32 gives: 32*16/512 = 1 b64 segment.
    """
    sme_load(slb_addr, desc, g_offset, kop,
             shape=[32, 32], element_size=16, transpose=True,
             loc=loc, ip=ip)


# ═══════════════════════════════════════════════════════════════════
# Async copy synchronization (native IXDL ops)
# ═══════════════════════════════════════════════════════════════════

def cp_async_commit_group(*, loc=None, ip=None):
    _ixdl_dialect.cp_async_commit_group(loc=loc, ip=ip)


def cp_async_wait_group(n, *, loc=None, ip=None):
    """Wait until at most *n* async copy groups remain.

    Args:
        n: int — number of groups (compile-time constant).
    """
    _ixdl_dialect.cp_async_wait_group(n=n, loc=loc, ip=ip)


# ═══════════════════════════════════════════════════════════════════
# Barrier (native IXDL op)
# ═══════════════════════════════════════════════════════════════════

def barrier(*, loc=None, ip=None):
    """Block-level barrier (__syncthreads on Iluvatar)."""
    _ixdl_dialect.barrier(loc=loc, ip=ip)


# ═══════════════════════════════════════════════════════════════════
# TCU MMA — native ixdl.mmad
# ═══════════════════════════════════════════════════════════════════

def mmad_f16(a_v4f16, b_v4f16, c_v4f32, *, loc=None, ip=None):
    """FP16 TCU 16×16×16: D[v4f32] = A[v4f16] × B[v4f16] + C[v4f32].

    Uses native ``ixdl.mmad`` with shape/layout/type attributes.
    The IXDL→LLVM translation automatically maps to
    ``@llvm.bi.matrix.mad.f32x4.f16x4``.
    """
    return _ixdl_dialect.mmad(
        res=_v4f32(),
        shape=ir.Attribute.parse('#ixdl.shape<m = 16, n = 16, k = 16>'),
        layout_a=_ixdl_dialect.MMADLayout.row,
        layout_b=_ixdl_dialect.MMADLayout.col,
        operand_a=[a_v4f16],
        operand_b=[b_v4f16],
        operand_c=[c_v4f32],
        multiplicand_a_type=_ixdl_dialect.MMADTypes.f16,
        multiplicand_b_type=_ixdl_dialect.MMADTypes.f16,
        loc=loc, ip=ip,
    )


def mmad_bf16(a, b, c, *, loc=None, ip=None):
    """BF16 TCU 16×16×16."""
    return _ixdl_dialect.mmad(
        res=_v4f32(),
        shape=ir.Attribute.parse('#ixdl.shape<m = 16, n = 16, k = 16>'),
        layout_a=_ixdl_dialect.MMADLayout.row,
        layout_b=_ixdl_dialect.MMADLayout.col,
        operand_a=[a],
        operand_b=[b],
        operand_c=[c],
        multiplicand_a_type=_ixdl_dialect.MMADTypes.bf16,
        multiplicand_b_type=_ixdl_dialect.MMADTypes.bf16,
        loc=loc, ip=ip,
    )


def mmad_i8(a, b, c, *, loc=None, ip=None):
    """INT8 TCU 16×16×16."""
    return _ixdl_dialect.mmad(
        res=_v4i32(),
        shape=ir.Attribute.parse('#ixdl.shape<m = 16, n = 16, k = 16>'),
        layout_a=_ixdl_dialect.MMADLayout.row,
        layout_b=_ixdl_dialect.MMADLayout.col,
        operand_a=[a],
        operand_b=[b],
        operand_c=[c],
        multiplicand_a_type=_ixdl_dialect.MMADTypes.s8,
        multiplicand_b_type=_ixdl_dialect.MMADTypes.s8,
        loc=loc, ip=ip,
    )


def mmad(res_type, shape_m, shape_n, shape_k,
         a, b, c,
         layout_a=None, layout_b=None,
         type_a=None, type_b=None,
         *, loc=None, ip=None):
    """Fully parameterized TCU mmad. For advanced use."""
    if layout_a is None:
        layout_a = _ixdl_dialect.MMADLayout.row
    if layout_b is None:
        layout_b = _ixdl_dialect.MMADLayout.col
    shape_str = f'#ixdl.shape<m = {shape_m}, n = {shape_n}, k = {shape_k}>'
    return _ixdl_dialect.mmad(
        res=res_type,
        shape=ir.Attribute.parse(shape_str),
        layout_a=layout_a,
        layout_b=layout_b,
        operand_a=[a],
        operand_b=[b],
        operand_c=[c],
        multiplicand_a_type=type_a,
        multiplicand_b_type=type_b,
        loc=loc, ip=ip,
    )


# ═══════════════════════════════════════════════════════════════════
# Lane / shuffle (native IXDL ops)
# ═══════════════════════════════════════════════════════════════════

def lane_id(*, loc=None, ip=None):
    """Lane ID (0~63) within the warp."""
    return _ixdl_dialect.lane_id(res=_i32(), loc=loc, ip=ip)


def readlane(val, lane, *, loc=None, ip=None):
    """Read scalar from a specific lane."""
    return _ixdl_dialect.readlane(res=_i32(), src=val, lane=lane, loc=loc, ip=ip)


def shfl_idx(src, index, *, loc=None, ip=None):
    """Shuffle index within warp."""
    return _ixdl_dialect.shfl_idx(res=_i32(), src=src, index=index, loc=loc, ip=ip)


# ═══════════════════════════════════════════════════════════════════
# Thread / block IDs (native IXDL ops)
# ═══════════════════════════════════════════════════════════════════

def workitem_id_x(*, loc=None, ip=None):
    return _ixdl_dialect.workitem_id_x(res=_i32(), loc=loc, ip=ip)


def workitem_id_y(*, loc=None, ip=None):
    return _ixdl_dialect.workitem_id_y(res=_i32(), loc=loc, ip=ip)


def workitem_id_z(*, loc=None, ip=None):
    return _ixdl_dialect.workitem_id_z(res=_i32(), loc=loc, ip=ip)


def workgroup_id_x(*, loc=None, ip=None):
    return _ixdl_dialect.workgroup_id_x(res=_i32(), loc=loc, ip=ip)


def workgroup_id_y(*, loc=None, ip=None):
    return _ixdl_dialect.workgroup_id_y(res=_i32(), loc=loc, ip=ip)


def workgroup_id_z(*, loc=None, ip=None):
    return _ixdl_dialect.workgroup_id_z(res=_i32(), loc=loc, ip=ip)


# ═══════════════════════════════════════════════════════════════════
# Enum re-exports for convenience
# ═══════════════════════════════════════════════════════════════════

MMADLayout = _ixdl_dialect.MMADLayout
MMADTypes = _ixdl_dialect.MMADTypes
ShflKind = _ixdl_dialect.ShflKind


# ═══════════════════════════════════════════════════════════════════
# Below: ops NOT in the IXDL dialect → llvm.call_intrinsic fallback
# ═══════════════════════════════════════════════════════════════════


# ─── Memory store/load via descriptor ─────────────────────────────

def ml_mem_store_i32(val, desc, offset_a, offset_b, kop,
                     *, loc=None, ip=None):
    """Store i32 via 1D descriptor.

    addr = desc_base + offset_a + offset_b.
    Maps to ``@llvm.bi.stp.vs``.
    """
    ptr = _desc_to_global_ptr(desc)
    _call_void_intrinsic(
        "llvm.bi.stp.vs",
        [val, ptr, offset_a, offset_b, kop],
        loc=loc, ip=ip,
    )


def ml_mem_store_i16(val, desc, offset_a, offset_b, kop,
                     *, loc=None, ip=None):
    """Store i16 via 1D descriptor (for FP16 epilogue)."""
    ptr = _desc_to_global_ptr(desc)
    _call_void_intrinsic(
        "llvm.bi.stp.vs",
        [val, ptr, offset_a, offset_b, kop],
        loc=loc, ip=ip,
    )


def ml_mem_load_i32(desc, voffset, soffset, kop,
                    *, loc=None, ip=None):
    """Load i32 via 1D descriptor.  Maps to ``@llvm.bi.ldp.vs``."""
    ptr = _desc_to_global_ptr(desc)
    return _call_intrinsic(
        _i32(),
        "llvm.bi.ldp.vs",
        [ptr, voffset, soffset, kop],
        loc=loc, ip=ip,
    )


def _desc_to_global_ptr(desc):
    """Extract a global-address-space pointer from a v4i32 descriptor.

    Mirrors ixcc Clang codegen: bitcast v4i32 → v2i64, extract elem 0, inttoptr.
    """
    i64 = _i64()
    v2i64 = ir.VectorType.get([2], i64)
    llvm_ptr = ir.Type.parse("!llvm.ptr<1>")

    as_v2i64 = llvm.bitcast(v2i64, desc)
    c0 = _i32_const(0)
    base_i64 = llvm.ExtractElementOp(as_v2i64, c0).result
    return llvm.inttoptr(llvm_ptr, base_i64)


# ─── Descriptor builders ─────────────────────────────────────────

def build_desc_sme(base_ptr, stride_bytes, *, loc=None, ip=None):
    """Build a v4i32 SME 2D descriptor.

    Mirrors CUDA ``__build_desc_sme(ptr, stride)``.
    word0-1: base address, word2: 0 (flags), word3: stride in bytes.
    """
    i32 = _i32()
    i64 = _i64()

    ptr_i64 = llvm.ptrtoint(i64, base_ptr)
    lo = _arith.TruncIOp(i32, ptr_i64).result
    hi = _arith.TruncIOp(
        i32, _arith.ShRUIOp(
            ptr_i64,
            _arith.ConstantOp(i64, ir.IntegerAttr.get(i64, 32)).result,
        ).result
    ).result
    flags = _i32_const(0)

    undef = llvm.mlir_undef(_v4i32())
    v = llvm.InsertElementOp(undef, lo,           _i32_const(0)).result
    v = llvm.InsertElementOp(v,     hi,           _i32_const(1)).result
    v = llvm.InsertElementOp(v,     flags,        _i32_const(2)).result
    v = llvm.InsertElementOp(v,     stride_bytes, _i32_const(3)).result
    return v


def build_desc(base_ptr, *, loc=None, ip=None):
    """Build a v4i32 1D descriptor (for store).

    Mirrors CUDA ``__build_desc(ptr)``.
    word0-1: base address, word2: 0xFFFFFFFF, word3: 0.
    """
    i32 = _i32()
    i64 = _i64()

    ptr_i64 = llvm.ptrtoint(i64, base_ptr)
    lo = _arith.TruncIOp(i32, ptr_i64).result
    hi = _arith.TruncIOp(
        i32, _arith.ShRUIOp(
            ptr_i64,
            _arith.ConstantOp(i64, ir.IntegerAttr.get(i64, 32)).result,
        ).result
    ).result
    num_rec = _i32_const(0xFFFFFFFF)
    flags = _i32_const(0)

    undef = llvm.mlir_undef(_v4i32())
    v = llvm.InsertElementOp(undef, lo,      _i32_const(0)).result
    v = llvm.InsertElementOp(v,     hi,      _i32_const(1)).result
    v = llvm.InsertElementOp(v,     num_rec, _i32_const(2)).result
    v = llvm.InsertElementOp(v,     flags,   _i32_const(3)).result
    return v


def advance_desc_base(desc, byte_delta, *, loc=None, ip=None):
    """Advance a descriptor's base address by *byte_delta* bytes.

    Equivalent to CUDA ``reinterpret_cast<v2i64&>(desc).x += byte_delta;``
    """
    i32 = _i32()
    i64 = _i64()

    c0 = _i32_const(0)
    c1 = _i32_const(1)

    lo = llvm.ExtractElementOp(desc, c0).result
    hi = llvm.ExtractElementOp(desc, c1).result

    lo_64 = _arith.ExtUIOp(i64, lo).result
    hi_64 = _arith.ExtUIOp(i64, hi).result
    base = _arith.OrIOp(
        _arith.ShLIOp(hi_64, _arith.ConstantOp(i64, ir.IntegerAttr.get(i64, 32)).result).result,
        lo_64,
    ).result

    if isinstance(byte_delta, int):
        byte_delta = _i32_const(byte_delta)
    delta_64 = _arith.ExtSIOp(i64, byte_delta).result
    new_base = _arith.AddIOp(base, delta_64).result

    new_lo = _arith.TruncIOp(i32, new_base).result
    new_hi = _arith.TruncIOp(
        i32, _arith.ShRUIOp(
            new_base,
            _arith.ConstantOp(i64, ir.IntegerAttr.get(i64, 32)).result,
        ).result
    ).result

    v = llvm.InsertElementOp(desc,  new_lo, c0).result
    v = llvm.InsertElementOp(v,     new_hi, c1).result
    return v


# ─── Scheduling hints ────────────────────────────────────────────

def sch_barrier(*, loc=None, ip=None):
    """Scheduling barrier hint. Maps to ``@llvm.bi.sch.barrier``."""
    _call_void_intrinsic("llvm.bi.sch.barrier", [], loc=loc, ip=ip)


# ─── Pipeline barrier (split arrive/wait) ─────────────────────────

def pipebar_req(n=0, *, loc=None, ip=None):
    """Pipeline barrier arrive (producer signals completion).

    Maps to ``@llvm.bi.pipebar.req``.
    Used in multi-stage GEMM after ``sl_waitcnt`` to signal that
    this warp's outstanding G2S loads have completed.
    """
    _call_void_intrinsic(
        "llvm.bi.pipebar.req", [_i32_const(n)], loc=loc, ip=ip)


def pipebar_wait(n=0, *, loc=None, ip=None):
    """Pipeline barrier wait (consumer waits for all producers).

    Maps to ``@llvm.bi.pipebar.wait``.
    Blocks until all warps in the block have called ``pipebar_req``.
    """
    _call_void_intrinsic(
        "llvm.bi.pipebar.wait", [_i32_const(n)], loc=loc, ip=ip)


# ─── Wait count ──────────────────────────────────────────────────

def sl_waitcnt(cnt, *, loc=None, ip=None):
    """Wait for outstanding memory operations.

    *cnt* is a packed 64-bit WaitCount value. On ivcore11 the bit layout is::

        bit  2 : LM   (shared-memory counter enable)
        bit  3 : G2S  (SME async-copy counter enable)
        bits 19-22 : LM_CNT   (max outstanding shared-memory ops)
        bits 23-28 : G2S_CNT  (max outstanding G2S ops)

    For multi-stage GEMM Stage 2: ``cnt = 0x0C`` (G2S=1, LM=1, all CNT=0).
    """
    i64 = _i64()
    if isinstance(cnt, int):
        cnt = _arith.ConstantOp(i64, ir.IntegerAttr.get(i64, cnt)).result
    _call_void_intrinsic("llvm.bi.sl.waitcnt", [cnt], loc=loc, ip=ip)


# ═══════════════════════════════════════════════════════════════════
# High-level G2S helper for CuTe-style GEMM
# ═══════════════════════════════════════════════════════════════════

def sme_g2s_tile(global_tensor, smem_tensor, stride_bytes, *,
                 loc=None, ip=None):
    """Async copy a single 16×32 f16 tile from global to shared memory via SME.

    Block-collective operation. Use for single-sub-tile G2S (BM=16).
    For multi-sub-tile tiles (BM>16), use ``sme_g2s_warp`` instead.

    Args:
        global_tensor: CuTe ``fly.memref`` pointing to global memory.
        smem_tensor:   CuTe ``fly.memref`` pointing to shared memory
                       (created by ``make_smem_tile``).
        stride_bytes:  i32 value — row stride of the global tensor in bytes.
    """
    from .._mlir.dialects import fly as _fly

    llvm_ptr_ty = ir.Type.parse("!llvm.ptr")

    global_ptr = _fly.extract_aligned_pointer_as_index(
        llvm_ptr_ty, global_tensor, loc=loc, ip=ip)

    smem_ptr = _fly.extract_aligned_pointer_as_index(
        llvm_ptr_ty, smem_tensor, loc=loc, ip=ip)
    smem_i64 = llvm.ptrtoint(_i64(), smem_ptr)
    slb_addr = _arith.TruncIOp(_i32(), smem_i64).result

    desc = build_desc_sme(global_ptr, stride_bytes, loc=loc, ip=ip)

    g_offset = _i32_const(0)
    kop = _i32_const(1)

    sme_load_16x1b64_rowxfb16(slb_addr, desc, g_offset, kop,
                               loc=loc, ip=ip)
    cp_async_commit_group(loc=loc, ip=ip)
    cp_async_wait_group(0, loc=loc, ip=ip)


def sme_g2s_warp(global_tensor, smem_tensor, stride_bytes,
                 M, K, warp_id, *,
                 loc=None, ip=None):
    """Warp-level G2S: each warp loads its assigned 16×K sub-tile via SME.

    For an M×K shared-memory tile (M must be a multiple of 16), the tile is
    decomposed into M//16 independent 16×K sub-tiles.  Each warp (identified
    by ``warp_id``) loads the sub-tile at rows ``[warp_id*16 .. (warp_id+1)*16)``.

    Within each sub-tile, K is further decomposed into K//32 strips of 32
    columns, each loaded by one ``sme_load_16x1b64_rowxfb16`` call.

    Commit and wait are issued per-warp so that all sub-tile data is available
    after this call + a ``barrier()``.

    Args:
        global_tensor: CuTe ``fly.memref`` (global memory) — base of the
                       current block's tile (already sliced by bid_m/ki).
        smem_tensor:   CuTe ``fly.memref`` (shared memory) — created by
                       ``make_smem_tile(M, K, ...)``.
        stride_bytes:  i32 constant — row stride of the **original global
                       matrix** in bytes (not the tile's BK*2).
        M:             Python int — tile rows (must be multiple of 16).
        K:             Python int — tile columns (must be multiple of 32).
        warp_id:       ir.Value (i32) — ``tid // 64``, range ``[0, M//16)``.
    """
    assert M % 16 == 0, f"sme_g2s_warp: M={M} must be a multiple of 16"
    assert K % 32 == 0, f"sme_g2s_warp: K={K} must be a multiple of 32"
    from .._mlir.dialects import fly as _fly

    if hasattr(warp_id, 'value'):
        warp_id = warp_id.value
    if isinstance(warp_id.type, ir.IndexType):
        warp_id = _arith.IndexCastOp(_i32(), warp_id).result

    llvm_ptr_ty = ir.Type.parse("!llvm.ptr")
    i32 = _i32()
    i64 = _i64()

    global_ptr = _fly.extract_aligned_pointer_as_index(
        llvm_ptr_ty, global_tensor, loc=loc, ip=ip)
    smem_ptr = _fly.extract_aligned_pointer_as_index(
        llvm_ptr_ty, smem_tensor, loc=loc, ip=ip)
    smem_i64 = llvm.ptrtoint(i64, smem_ptr)
    slb_base = _arith.TruncIOp(i32, smem_i64).result

    # Global row offset: warp_id * 16 rows * stride_bytes
    row_byte_offset = _arith.MulIOp(
        _arith.MulIOp(warp_id, _i32_const(16)).result,
        stride_bytes,
    ).result
    row_byte_offset_i64 = _arith.ExtSIOp(i64, row_byte_offset).result
    global_base_i64 = llvm.ptrtoint(i64, global_ptr)
    global_sub_i64 = _arith.AddIOp(global_base_i64, row_byte_offset_i64).result
    global_sub_ptr = llvm.inttoptr(llvm_ptr_ty, global_sub_i64)

    # SMEM sub-tile offset: warp_id * 16 * K * sizeof(f16)
    sub_tile_bytes = 16 * K * 2
    smem_sub_offset = _arith.MulIOp(warp_id, _i32_const(sub_tile_bytes)).result
    slb_sub = _arith.AddIOp(slb_base, smem_sub_offset).result

    kop = _i32_const(1)
    k_strips = K // 32

    for ki in range(k_strips):
        if ki == 0:
            g_col_offset = _i32_const(0)
            slb_strip = slb_sub
        else:
            g_col_offset = _i32_const(ki * 32 * 2)
            slb_strip = _arith.AddIOp(
                slb_sub, _i32_const(ki * 16 * 32 * 2)
            ).result

        if ki == 0:
            desc = build_desc_sme(global_sub_ptr, stride_bytes, loc=loc, ip=ip)
        else:
            col_offset_i64 = _arith.ExtSIOp(i64, g_col_offset).result
            sub_base_i64 = llvm.ptrtoint(i64, global_sub_ptr)
            g_strip_i64 = _arith.AddIOp(sub_base_i64, col_offset_i64).result
            g_strip_ptr = llvm.inttoptr(llvm_ptr_ty, g_strip_i64)
            desc = build_desc_sme(g_strip_ptr, stride_bytes, loc=loc, ip=ip)

        sme_load_16x1b64_rowxfb16(slb_strip, desc, _i32_const(0), kop,
                                   loc=loc, ip=ip)

    cp_async_commit_group(loc=loc, ip=ip)
    cp_async_wait_group(0, loc=loc, ip=ip)
