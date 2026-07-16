// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"

#include <optional>

using namespace mlir;
using namespace mlir::fly;

namespace mlir {
#define GEN_PASS_DEF_IXDLADDRESSSIMPLIFYPASS
#include "flydsl/Conversion/FlyToIXDL/Passes.h.inc"
} // namespace mlir

namespace {

struct LaneAffineMatch {
  enum class Kind { Affine, Xor32Affine };

  Kind kind;
  int64_t scale;
  int64_t bias;
};

bool isLaneIdProducer(Value v) {
  if (v == nullptr)
    return false;
  return v.getDefiningOp<gpu::LaneIdOp>() != nullptr;
}

Value findI32LaneId(gpu::GPUFuncOp fn) {
  Value laneId;
  fn.walk([&](arith::IndexCastOp op) -> WalkResult {
    if (!op.getResult().getType().isInteger(32))
      return WalkResult::advance();
    if (op->getNumOperands() != 1 || !isLaneIdProducer(op->getOperand(0)))
      return WalkResult::advance();

    laneId = op.getResult();
    return WalkResult::interrupt();
  });
  return laneId;
}

std::optional<int64_t> evalLaneExprAt(Value v, Value laneId, int64_t lane, int depth = 0) {
  if (!v || depth > 32)
    return std::nullopt;
  if (v == laneId || isLaneIdProducer(v))
    return lane;
  if (auto c = getConstantIntValue(v))
    return *c;

  Operation *def = v.getDefiningOp();
  if (!def)
    return std::nullopt;

  if (auto indexCast = dyn_cast<arith::IndexCastOp>(def)) {
    if (indexCast->getNumOperands() != 1)
      return std::nullopt;
    return evalLaneExprAt(indexCast->getOperand(0), laneId, lane, depth + 1);
  }

  auto evalBinary = [&](auto op, auto fn) -> std::optional<int64_t> {
    auto lhs = evalLaneExprAt(op.getLhs(), laneId, lane, depth + 1);
    auto rhs = evalLaneExprAt(op.getRhs(), laneId, lane, depth + 1);
    if (!lhs || !rhs)
      return std::nullopt;
    return fn(*lhs, *rhs);
  };

  return llvm::TypeSwitch<Operation *, std::optional<int64_t>>(def)
      .Case<arith::AddIOp>(
          [&](auto op) { return evalBinary(op, [](int64_t a, int64_t b) { return a + b; }); })
      .Case<arith::SubIOp>(
          [&](auto op) { return evalBinary(op, [](int64_t a, int64_t b) { return a - b; }); })
      .Case<arith::MulIOp>(
          [&](auto op) { return evalBinary(op, [](int64_t a, int64_t b) { return a * b; }); })
      .Case<arith::DivSIOp>([&](auto op) -> std::optional<int64_t> {
        return evalBinary(op, [](int64_t a, int64_t b) -> std::optional<int64_t> {
          if (b == 0)
            return std::nullopt;
          return a / b;
        });
      })
      .Case<arith::FloorDivSIOp>([&](auto op) -> std::optional<int64_t> {
        return evalBinary(op, [](int64_t a, int64_t b) -> std::optional<int64_t> {
          if (b == 0)
            return std::nullopt;
          int64_t q = a / b;
          int64_t r = a % b;
          if (r != 0 && ((r > 0) != (b > 0)))
            --q;
          return q;
        });
      })
      .Case<arith::RemSIOp>([&](auto op) -> std::optional<int64_t> {
        return evalBinary(op, [](int64_t a, int64_t b) -> std::optional<int64_t> {
          if (b == 0)
            return std::nullopt;
          return a % b;
        });
      })
      .Case<arith::AndIOp>(
          [&](auto op) { return evalBinary(op, [](int64_t a, int64_t b) { return a & b; }); })
      .Case<arith::OrIOp>(
          [&](auto op) { return evalBinary(op, [](int64_t a, int64_t b) { return a | b; }); })
      .Case<arith::XOrIOp>(
          [&](auto op) { return evalBinary(op, [](int64_t a, int64_t b) { return a ^ b; }); })
      .Case<arith::ShLIOp>([&](auto op) -> std::optional<int64_t> {
        return evalBinary(op, [](int64_t a, int64_t b) -> std::optional<int64_t> {
          if (b < 0 || b >= 63)
            return std::nullopt;
          return a << b;
        });
      })
      .Case<arith::ShRUIOp>([&](auto op) -> std::optional<int64_t> {
        return evalBinary(op, [](int64_t a, int64_t b) -> std::optional<int64_t> {
          if (b < 0 || b >= 63)
            return std::nullopt;
          return static_cast<int64_t>(static_cast<uint64_t>(a) >> b);
        });
      })
      .Default(std::optional<int64_t>());
}

// Check if the lane offset is equal to "A * lane_id + B" or
// "A * (lane_id ^ 32) + B" over the target warp.
std::optional<LaneAffineMatch> matchLaneAffine(Value v, Value laneId, int warpSize) {
  if (warpSize <= 0)
    return std::nullopt;

  SmallVector<int64_t, 64> values;
  values.reserve(warpSize);
  for (int lane = 0; lane < warpSize; ++lane) {
    auto value = evalLaneExprAt(v, laneId, lane);
    if (!value)
      return std::nullopt;
    values.push_back(*value);
  }

  constexpr int64_t scales[] = {1, 2, 4};
  for (int64_t scale : scales) {
    int64_t bias = values.front();
    bool ok = true;
    for (int lane = 0; lane < warpSize; ++lane) {
      if (values[lane] != scale * lane + bias) {
        ok = false;
        break;
      }
    }
    if (ok)
      return LaneAffineMatch{LaneAffineMatch::Kind::Affine, scale, bias};
  }

  for (int64_t scale : scales) {
    int64_t bias = values.front() - scale * (0 ^ 32);
    bool ok = true;
    for (int lane = 0; lane < warpSize; ++lane) {
      if (values[lane] != scale * (lane ^ 32) + bias) {
        ok = false;
        break;
      }
    }
    if (ok)
      return LaneAffineMatch{LaneAffineMatch::Kind::Xor32Affine, scale, bias};
  }

  return std::nullopt;
}

Value materializeLaneAffine(OpBuilder &b, Location loc, Type ty, Value laneId,
                            LaneAffineMatch match) {
  Value cur = laneId;
  if (match.kind == LaneAffineMatch::Kind::Xor32Affine) {
    Value c32 = arith::ConstantIntOp::create(b, loc, ty, 32);
    cur = arith::XOrIOp::create(b, loc, cur, c32);
  }
  if (match.scale != 1) {
    Value scale = arith::ConstantIntOp::create(b, loc, ty, match.scale);
    cur = arith::MulIOp::create(b, loc, cur, scale);
  }
  if (match.bias != 0) {
    Value bias = arith::ConstantIntOp::create(b, loc, ty, match.bias);
    cur = arith::AddIOp::create(b, loc, cur, bias);
  }
  return cur;
}

bool isSupportedLaneExprOp(Operation *op) {
  return isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp, arith::FloorDivSIOp,
             arith::RemSIOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp, arith::ShLIOp,
             arith::ShRUIOp, arith::IndexCastOp>(op);
}

void rewriteLaneExprInAddr(Value v, Value laneId, int warpSize,
                           llvm::SmallPtrSetImpl<Value> &rewritten,
                           llvm::SmallPtrSetImpl<Value> &seen, int depth = 0) {
  if (!v || depth > 32 || !seen.insert(v).second)
    return;

  Operation *def = v.getDefiningOp();
  if (!def)
    return;

  if (v.getType().isInteger(32) && v != laneId && !isLaneIdProducer(v) &&
      !rewritten.contains(v)) {
    if (auto match = matchLaneAffine(v, laneId, warpSize)) {
      OpBuilder b(def);
      Value repl = materializeLaneAffine(b, def->getLoc(), v.getType(), laneId, *match);
      v.replaceAllUsesWith(repl);
      rewritten.insert(v);
      return;
    }
  }

  if (!isSupportedLaneExprOp(def))
    return;

  for (Value operand : def->getOperands())
    rewriteLaneExprInAddr(operand, laneId, warpSize, rewritten, seen, depth + 1);
}

// After swizzle, shared-memory loads usually carry a swizzled lane expression in
// add_offset. Collapse equivalent lane affine forms to expose simpler address
// calculation to the IXDL backend.
void optAddrInFunc(gpu::GPUFuncOp fn, int warpSize) {
  Value laneId = findI32LaneId(fn);
  if (!laneId)
    return;

  llvm::SmallPtrSet<Value, 32> rewritten;
  fn.walk([&](CopyAtomCall copy) {
    Value src = copy.getSrc();
    auto srcTy = dyn_cast<mlir::fly::MemRefType>(src.getType());
    if (!srcTy || !isGenericAddressSpace<AddressSpace::Shared>(srcTy.getAddressSpace()))
      return;

    auto makeView = src.getDefiningOp<MakeViewOp>();
    if (!makeView)
      return;
    auto addOffset = makeView.getIter().getDefiningOp<AddOffsetOp>();
    if (!addOffset)
      return;
    auto ptrTy = dyn_cast<mlir::fly::PointerType>(addOffset.getPtr().getType());
    if (!ptrTy || !isGenericAddressSpace<AddressSpace::Shared>(ptrTy.getAddressSpace()))
      return;

    auto makeOffset = addOffset.getOffset().getDefiningOp<MakeIntTupleOp>();
    if (!makeOffset || makeOffset->getNumOperands() != 1)
      return;

    llvm::SmallPtrSet<Value, 32> seen;
    rewriteLaneExprInAddr(makeOffset->getOperand(0), laneId, warpSize, rewritten, seen);
  });
}

class IXDLAddressSimplifyPass
    : public mlir::impl::IXDLAddressSimplifyPassBase<IXDLAddressSimplifyPass> {
public:
  using mlir::impl::IXDLAddressSimplifyPassBase<
      IXDLAddressSimplifyPass>::IXDLAddressSimplifyPassBase;

  void runOnOperation() override {
    auto module = getOperation();
    module->walk([&](gpu::GPUModuleOp gpuModule) {
      gpuModule.walk([&](gpu::GPUFuncOp fn) { optAddrInFunc(fn, /*warpSize=*/64); });
    });
  }
};

} // namespace
