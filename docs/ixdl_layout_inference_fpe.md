# IXDL CUTLASS Layout Inference FPE

## Summary

When `MmaAtomIXDLMMADType` was switched to the CUTLASS IX11 layouts for
`16x16x16 f16,f16->f32`, FlyDSL started crashing with a host-side
`Floating point exception` during compilation.

The crash was **not** caused by `ixdl.mmad` itself and **not** caused by the
backend LLVM/IXDL lowering. The failure happened earlier, inside FlyDSL's
fragment layout inference for `mma_atom.make_fragment`.

The root cause was a semantic mismatch:

- `MmaMakeFragmentOp::inferReturnTypes()` treated its input as if it were an
  **unpartitioned tensor view**
- but the lowering path already treats the input as an
  **already partitioned operand fragment**

This caused the type inference path to partition operand `B` a second time.
With the CUTLASS `B` layout, that extra partitioning produced an illegal
intermediate layout and eventually triggered a division-by-zero in FlyDSL's
layout algebra (`compositionImpl`), surfacing as `SIGFPE`.

The fix was to make type inference match lowering semantics: derive the
fragment layout directly from the input layout via `layoutMakeFragmentLayout()`,
instead of re-running operand partitioning.

## Affected Layouts

The triggering layouts were the CUTLASS IX11 canonical layouts:

```c++
A = ((16,4),(2,2)):((16,2),(1,8))
B = ((16,4),(2,2)):((1,32),(16,128))
C = ((16,4),4):((16,1),4)
```

In FlyDSL these correspond to:

- `IX11::Layout_16x16_16b_A`
- `IX11::Layout_16x16_16b_B`
- `IX11::Layout_16x16_32b_AC`

## Symptoms

Typical failure mode:

- `build.sh` succeeds
- `COMPILE_ONLY=1` crashes with `Floating point exception`
- isolated `make_fragment_B()` also crashes
- `make_fragment_A()` does **not** crash

This immediately narrows the problem to the `B` operand fragment inference path.

## Minimal Reproducer

With CUTLASS-style layouts enabled in `FlyTypeDefs.cpp`, the following path
fails during compilation:

1. build `tiled_mma_ixdl`
2. call `thr_mma.partition_B(B)`
3. call `thr_mma.make_fragment_B(part_B)`

At the type level, the key intermediate layout is:

```text
partition B = !fly.memref<f16, global, ((2,2),1,1):((1,8),0,0)>
```

`make_fragment_B()` should convert this already-partitioned view into a
register fragment layout. Instead, the old inference path re-applied
`layoutTiledMmaThrValOperandView(...)`, effectively treating the input as a
full logical operand again.

## Root Cause

### What lowering does

The lowering of `mma_atom.make_fragment` is semantically equivalent to:

```text
make_fragment_like(partitioned_tensor)
```

That is, it consumes an **already partitioned input** and materializes the
corresponding register fragment.

### What type inference used to do

Before the fix, `MmaMakeFragmentOp::inferReturnTypes()` did:

1. read the input layout
2. call `layoutTiledMmaThrValOperandView(...)`
3. slice out the first mode
4. expand that sliced result into a new "partitioned" layout
5. call `layoutMakeFragmentLayout(...)`

This is only valid if the input is an **unpartitioned logical operand**.
But the actual op input is already the result of `partition_A/B/C`.

### Why this caused FPE

For `B`, the extra partitioning step created an invalid intermediate layout
shape/stride combination. That bad layout flowed into:

- `layoutMakeFragmentLayout()`
- `layoutTiledProduct()`
- `compositionImpl()`

Inside `compositionImpl()`, one of the intermediate divisors became zero,
leading to a host-side integer divide-by-zero and a `Floating point exception`.

So the actual bug was:

> `MmaMakeFragmentOp` type inference and lowering implemented different
> semantics.

## Fix

### File

- `lib/Dialect/Fly/IR/FlyOps.cpp`

### Old behavior

Re-partition the input during `inferReturnTypes()`.

### New behavior

Treat the input layout as already partitioned and compute the fragment layout
directly:

```c++
LayoutAttr fragmentLayout = layoutMakeFragmentLayout(builder, inputLayout);
```

This makes `inferReturnTypes()` consistent with lowering and with CuTe's
`make_fragment_like(partitioned_tensor)` semantics.

### Code Summary

Before the fix, `inferReturnTypes()` effectively did:

```c++
LayoutAttr thrValView = layoutTiledMmaThrValOperandView(builder, mmaAtom, atomLayout,
                                                        permutationMNK, operandId, inputLayout);

IntTupleAttr resultShape = intTupleSlice(builder, thrValView.getShape(), sliceCoord);
IntTupleAttr resultStride = intTupleSlice(builder, thrValView.getStride(), sliceCoord);
LayoutAttr partitioned = LayoutAttr::get(intTupleExpand(builder, resultShape, {1}),
                                         intTupleExpand(builder, resultStride, {1}));

LayoutAttr fragmentLayout = layoutMakeFragmentLayout(builder, partitioned);
```

After the fix, it does:

```c++
LayoutAttr fragmentLayout = layoutMakeFragmentLayout(builder, inputLayout);
```

This is the critical change: remove the spurious second partitioning step and
compute the register fragment layout directly from the already-partitioned op
input.

## Why this fix is correct

After the fix:

- `make_fragment_A()` succeeds
- `make_fragment_B()` succeeds
- `COMPILE_ONLY=1` no longer crashes
- copy-only fragment roundtrip tests pass again

Most importantly, the fix does **not** weaken layout validation in general; it
simply removes an invalid second partitioning pass that should never have
happened for this op.

## Validation Performed

### 1. Compile-only

`examples/03-tiledMma-ixdl.py` with `COMPILE_ONLY=1` succeeds.

### 2. Fragment copy roundtrip

`examples/03-tiledMma-ixdl-copy-debug.py` passes:

- `A roundtrip: True`
- `B roundtrip: True`
- `C roundtrip: True`

### 3. Runtime note

This fix resolves the **layout inference / FPE** bug only.

There is still a separate runtime semantic issue on the IXDL MMA path: current
runtime behavior matches `A @ B^T` rather than `A @ B`. That remaining issue is
orthogonal to the FPE fix.

## Final Takeaway

The CUTLASS IX11 layouts were not inherently invalid.

The real problem was that FlyDSL's `MmaMakeFragmentOp::inferReturnTypes()`
implemented the wrong abstraction boundary: it assumed the op input was a
logical operand tensor, while lowering already treated it as a partitioned view.

Once type inference was changed to operate directly on the input partitioned
layout, the CUTLASS layouts became compilable again and the host-side FPE
disappeared.
