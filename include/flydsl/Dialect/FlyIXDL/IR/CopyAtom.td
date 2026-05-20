// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYIXDL_COPYATOM
#define FLYIXDL_COPYATOM

include "flydsl/Dialect/FlyIXDL/IR/Dialect.td"

class FlyIXDL_MRAsyncCopy<string name, string mnem, int bitSize>
    : FlyIXDL_StatefulCopyOp<name, mnem> {
  let extraClassDeclaration = [{
    static std::optional<unsigned> getFieldIndex(AtomStateField field);
    int32_t getBitSize() const { return }] # bitSize # [{; }
  }];
}

def FlyIXDL_CopyOpMR_AsyncCopy_4x64_b8_Row
    : FlyIXDL_MRAsyncCopy<"CopyOpMRAsyncCopy4x64B8Row", "mr.async_copy.4x64.b8.row", 2048>;
def FlyIXDL_CopyOpMR_AsyncCopy_4x64_b8_Col
    : FlyIXDL_MRAsyncCopy<"CopyOpMRAsyncCopy4x64B8Col", "mr.async_copy.4x64.b8.col", 2048>;
def FlyIXDL_CopyOpMR_AsyncCopy_16x64_b8_Row
    : FlyIXDL_MRAsyncCopy<"CopyOpMRAsyncCopy16x64B8Row", "mr.async_copy.16x64.b8.row", 8192>;
def FlyIXDL_CopyOpMR_AsyncCopy_16x64_b8_Col
    : FlyIXDL_MRAsyncCopy<"CopyOpMRAsyncCopy16x64B8Col", "mr.async_copy.16x64.b8.col", 8192>;

def FlyIXDL_CopyOpMR_AsyncCopy_4x32_b16_Row
    : FlyIXDL_MRAsyncCopy<"CopyOpMRAsyncCopy4x32B16Row", "mr.async_copy.4x32.b16.row", 2048>;
def FlyIXDL_CopyOpMR_AsyncCopy_4x32_b16_Col
    : FlyIXDL_MRAsyncCopy<"CopyOpMRAsyncCopy4x32B16Col", "mr.async_copy.4x32.b16.col", 2048>;
def FlyIXDL_CopyOpMR_AsyncCopy_16x32_b16_Row
    : FlyIXDL_MRAsyncCopy<"CopyOpMRAsyncCopy16x32B16Row", "mr.async_copy.16x32.b16.row", 8192>;
def FlyIXDL_CopyOpMR_AsyncCopy_16x32_b16_Col
    : FlyIXDL_MRAsyncCopy<"CopyOpMRAsyncCopy16x32B16Col", "mr.async_copy.16x32.b16.col", 8192>;

def FlyIXDL_CopyOpMR_AsyncCopy_1x1b64
    : FlyIXDL_MRAsyncCopy<"CopyOpMRAsyncCopy1x1B64", "mr.async_copy.1x1b64", 512>;
def FlyIXDL_CopyOpMR_AsyncCopy_1x4b64
    : FlyIXDL_MRAsyncCopy<"CopyOpMRAsyncCopy1x4B64", "mr.async_copy.1x4b64", 2048>;
def FlyIXDL_CopyOpMR_AsyncCopy_1x8b64
    : FlyIXDL_MRAsyncCopy<"CopyOpMRAsyncCopy1x8B64", "mr.async_copy.1x8b64", 4096>;
def FlyIXDL_CopyOpMR_AsyncCopy_1x16b64
    : FlyIXDL_MRAsyncCopy<"CopyOpMRAsyncCopy1x16B64", "mr.async_copy.1x16b64", 8192>;

def FlyIXDL_CopyOpMR_AsyncCopy_4x16_b32_Row
    : FlyIXDL_MRAsyncCopy<"CopyOpMRAsyncCopy4x16B32Row", "mr.async_copy.4x16.b32.row", 2048>;
def FlyIXDL_CopyOpMR_AsyncCopy_8x16_b32_Row
    : FlyIXDL_MRAsyncCopy<"CopyOpMRAsyncCopy8x16B32Row", "mr.async_copy.8x16.b32.row", 4096>;
def FlyIXDL_CopyOpMR_AsyncCopy_16x16_b32_Row
    : FlyIXDL_MRAsyncCopy<"CopyOpMRAsyncCopy16x16B32Row", "mr.async_copy.16x16.b32.row", 8192>;
def FlyIXDL_CopyOpMR_AsyncCopy_16x16_b32_Col
    : FlyIXDL_MRAsyncCopy<"CopyOpMRAsyncCopy16x16B32Col", "mr.async_copy.16x16.b32.col", 8192>;

#endif // FLYIXDL_COPYATOM
