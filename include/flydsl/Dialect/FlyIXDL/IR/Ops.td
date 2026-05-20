// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYIXDL_OPS
#define FLYIXDL_OPS

include "flydsl/Dialect/FlyIXDL/IR/Dialect.td"

def FlyIXDL_CpAsyncCommitGroupOp
    : FlyIXDL_Op<"cp_async_commit_group", []> {
  let summary = "Commit outstanding MR async copy group";
  let assemblyFormat = "attr-dict";
}

def FlyIXDL_CpAsyncWaitGroupOp : FlyIXDL_Op<"cp_async_wait_group", []> {
  let summary = "Wait for MR async copy group";
  let arguments = (ins I32Attr:$n);
  let assemblyFormat = "$n attr-dict";
}

#endif // FLYIXDL_OPS
