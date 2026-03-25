# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx


def _dynamic_bool(x):
    return x


@flyc.kernel
def vectorAddKernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    n: fx.Int32,
    block_dim: fx.Constexpr[int],
):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x
    gid = bid * block_dim + tid

    in_range = gid < n
    safe_gid = in_range.select(gid, fx.Int32(0))

    a = fx.memref_load(A, safe_gid)
    b = fx.memref_load(B, safe_gid)
    c = fx.arith.addf(a, b)

    # Use a call expression so the current AST rewriter lowers this to scf.if.
    if _dynamic_bool(in_range):
        fx.memref_store(c, C, gid)


@flyc.jit
def vectorAdd(
    A: fx.Tensor,
    B: fx.Tensor,
    C,
    n: fx.Int32,
    const_n: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    block_dim = 64
    grid_x = (n + block_dim - 1) // block_dim
    fx.printf("> vectorAdd(global): n={}, grid_x={}", n, grid_x)

    vectorAddKernel(A, B, C, n, block_dim).launch(
        grid=(grid_x, 1, 1), block=[block_dim, 1, 1], stream=stream
    )


def run_eager():
    n = 128
    A = torch.randint(0, 10, (n,), dtype=torch.float32).cuda()
    B = torch.randint(0, 10, (n,), dtype=torch.float32).cuda()
    C = torch.zeros(n, dtype=torch.float32).cuda()
    vectorAdd(A, B, C, n, n + 1, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    ok = torch.allclose(C, A + B)
    print(f"[Eager] Result correct: {ok}")
    if not ok:
        print("A:", A[:32])
        print("B:", B[:32])
        print("C:", C[:32])
    print("Hello, Fly!")
    return ok


def run_graph_capture():
    if flyc.compile_backend_name() == "ixdl":
        print("[Graph Capture] Skipped for ixdl backend")
        return True

    n = 128
    A = torch.randint(0, 10, (n,), dtype=torch.float32).cuda()
    B = torch.randint(0, 10, (n,), dtype=torch.float32).cuda()
    C = torch.zeros(n, dtype=torch.float32).cuda()

    vectorAdd(A, B, C, n, n + 1, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    C.zero_()

    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(capture_stream):
        with torch.cuda.graph(graph, stream=capture_stream):
            vectorAdd(A, B, C, n, n + 1, stream=capture_stream)

    C.zero_()
    graph.replay()
    torch.cuda.synchronize()

    ok = torch.allclose(C, A + B)
    print(f"[Graph Capture] Result correct: {ok}")
    if not ok:
        print(f"  Expected: {(A + B)[:16]}")
        print(f"  Got:      {C[:16]}")
    return ok


if __name__ == "__main__":
    print("=" * 50)
    print("Test 1: Eager execution")
    print("=" * 50)
    ok1 = run_eager()

    print()
    print("=" * 50)
    print("Test 2: CUDA Graph Capture")
    print("=" * 50)
    try:
        ok2 = run_graph_capture()
    except Exception as e:
        print(f"[Graph Capture] FAILED with exception: {e}")
        ok2 = False

    print()
    print(f"All passed: {ok1 and ok2}")
