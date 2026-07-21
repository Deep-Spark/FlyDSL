// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

// IXDL shared-memory address peeps (kept minimal for measured S2R benefit):
//   1) HGEMM: collapse equivalent lane-affine forms in shared copy offsets
//      (e.g. (lane^1)^33 → lane^32).
//   2) B8 / Row8b S2R: Euclidean thr→byte + ModSwizzle closed forms, then
//      readfirstlane on warp-uniform bases of `base + 4*lane`.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"

#include <algorithm>
#include <optional>
#include <utility>

using namespace mlir;
using namespace mlir::fly;

namespace mlir {
#define GEN_PASS_DEF_IXDLADDRESSSIMPLIFYPASS
#include "flydsl/Conversion/FlyToIXDL/Passes.h.inc"
} // namespace mlir

namespace {

// =============================================================================
// HGEMM: lane-affine collapse of swizzled shared copy offsets
// =============================================================================

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

// =============================================================================
// B8 / Row8b: Euclidean thr→byte fold + ModSwizzle closed forms
// =============================================================================
//
// Row8b S2R lowers CuTe-style ModSwizzle MS<B,M,S> into bit arithmetic. For the
// B8 tile path the concrete parameters are MS<2,6,2>:
//
//   yyyMask = 768 = 0b11_0000_0000   // bits [8,9]
//   zbMask  = 255 = 0b00_1111_1111   // low 8 bits (zb)
//   nzbMask = ~255 = -256            // bits above zb
//   shift   = 2
//
// Expanded IR (matched by simplifyModSwizzle on the final `high | low` ori):
//   yyy  = x & yyyMask
//   shr  = yyy >> shift
//   low  = (x + shr) & zbMask
//   high = x & nzbMask
//   MS(x) = high | low
//
// Folds (need UpperBound to prove ranges):
//   1) Identity: if x < yyyLo (= lowest set bit of yyyMask = 256), then
//      yyy==0 ⇒ MS(x) == x.
//   2) Second word: if x = base+256 with base < 256, then for MS<2,6,2>
//      yyy==256, shr==64, high==256, so
//        MS(base+256) → ((base + 320) & 255) | 256
//      (320 = 256+64 folds the constant into the low-byte add).
//
// These closed forms leave a natural `base + 4*lane` shape for the later
// readfirstlane peep.

// Exclusive upper bound for non-negative SSA values. Used to prove:
//   * A*(x%N)+E+(A*N)*(x/N) == A*x+E when x >= 0 (incl. A==1)
//   * ModSwizzle identity when x < yyyLo (so x & yyyMask == 0)
//   * Second-word closed form when base < 256 for x = base+256
class UpperBound {
public:
  std::optional<uint64_t> getExclusive(Value v);

private:
  static constexpr uint64_t kUnknown = uint64_t(-1);
  uint64_t fromDef(Value v);
  DenseMap<Value, uint64_t> cache;
  llvm::SmallPtrSet<Value, 16> visiting;
};

uint64_t UpperBound::fromDef(Value v) {
  if (auto c = getConstantIntValue(v)) {
    if (*c < 0)
      return kUnknown;
    return (uint64_t)*c + 1;
  }
  Operation *def = v.getDefiningOp();
  if (!def)
    return kUnknown;

  // ivcore11 warp is 64; lane_id ∈ [0, 64).
  if (isa<gpu::LaneIdOp>(def))
    return 64;
  if (isa<gpu::ThreadIdOp>(def))
    return kUnknown;

  return llvm::TypeSwitch<Operation *, uint64_t>(def)
      .Case<arith::IndexCastOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(
          [&](auto op) { return getExclusive(op.getIn()).value_or(kUnknown); })
      .Case<arith::AndIOp>([&](auto op) -> uint64_t {
        auto mask = getConstantIntValue(op.getRhs());
        if (!mask || *mask <= 0)
          mask = getConstantIntValue(op.getLhs());
        if (!mask || *mask <= 0)
          return kUnknown;
        uint64_t m = (uint64_t)*mask;
        if ((m & (m + 1)) != 0)
          return kUnknown;
        return m + 1;
      })
      .Case<arith::RemSIOp, arith::RemUIOp>([&](auto op) -> uint64_t {
        auto n = getConstantIntValue(op.getRhs());
        if (!n || *n <= 0)
          return kUnknown;
        return (uint64_t)*n;
      })
      .Case<arith::DivSIOp, arith::DivUIOp, arith::FloorDivSIOp>([&](auto op) -> uint64_t {
        auto n = getConstantIntValue(op.getRhs());
        auto hi = getExclusive(op.getLhs());
        if (!n || *n <= 0 || !hi || *hi == 0 || *hi == kUnknown)
          return kUnknown;
        return ((*hi - 1) / (uint64_t)*n) + 1;
      })
      .Case<arith::MulIOp>([&](auto op) -> uint64_t {
        auto cl = getConstantIntValue(op.getLhs());
        auto cr = getConstantIntValue(op.getRhs());
        if (cl && *cl >= 0) {
          auto hi = getExclusive(op.getRhs());
          if (!hi || *hi == kUnknown)
            return kUnknown;
          uint64_t maxR = *hi - 1;
          if (*cl != 0 && maxR > UINT64_MAX / (uint64_t)*cl)
            return kUnknown;
          return (uint64_t)*cl * maxR + 1;
        }
        if (cr && *cr >= 0) {
          auto hi = getExclusive(op.getLhs());
          if (!hi || *hi == kUnknown)
            return kUnknown;
          uint64_t maxL = *hi - 1;
          if (*cr != 0 && maxL > UINT64_MAX / (uint64_t)*cr)
            return kUnknown;
          return maxL * (uint64_t)*cr + 1;
        }
        return kUnknown;
      })
      .Case<arith::AddIOp>([&](auto op) -> uint64_t {
        auto a = getExclusive(op.getLhs());
        auto b = getExclusive(op.getRhs());
        if (!a || !b || *a == kUnknown || *b == kUnknown)
          return kUnknown;
        uint64_t maxA = *a - 1;
        uint64_t maxB = *b - 1;
        if (maxA > UINT64_MAX - maxB)
          return kUnknown;
        return maxA + maxB + 1;
      })
      .Case<arith::SelectOp>([&](auto op) -> uint64_t {
        auto a = getExclusive(op.getTrueValue());
        auto b = getExclusive(op.getFalseValue());
        if (!a || !b || *a == kUnknown || *b == kUnknown)
          return kUnknown;
        return std::max(*a, *b);
      })
      .Default(kUnknown);
}

std::optional<uint64_t> UpperBound::getExclusive(Value v) {
  if (auto it = cache.find(v); it != cache.end()) {
    if (it->second == kUnknown)
      return std::nullopt;
    return it->second;
  }
  if (!visiting.insert(v).second)
    return std::nullopt;
  uint64_t u = fromDef(v);
  visiting.erase(v);
  cache[v] = u;
  if (u == kUnknown)
    return std::nullopt;
  return u;
}

/// One term in a flattened add/sub tree: ``sign * v`` with ``sign ∈ {+1,-1}``.
struct SignedSummand {
  Value v;
  int sign;
};

/// Flatten a left/right-associative add/sub tree into signed summands.
/// ``a - b`` contributes ``+a`` and ``-b``. Depth is capped so matching stays local.
static void flattenAddSubTree(Value v, int sign, int depth, int maxDepth,
                              SmallVectorImpl<SignedSummand> &out) {
  if (depth < maxDepth) {
    if (auto add = v.getDefiningOp<arith::AddIOp>()) {
      flattenAddSubTree(add.getLhs(), sign, depth + 1, maxDepth, out);
      flattenAddSubTree(add.getRhs(), sign, depth + 1, maxDepth, out);
      return;
    }
    if (auto sub = v.getDefiningOp<arith::SubIOp>()) {
      flattenAddSubTree(sub.getLhs(), sign, depth + 1, maxDepth, out);
      flattenAddSubTree(sub.getRhs(), -sign, depth + 1, maxDepth, out);
      return;
    }
  }
  out.push_back({v, sign});
}

/// Match ``scale * (x rem/div N)`` or bare ``x rem N`` (scale == 1).
static bool matchScaledRemOrDiv(Value v, bool wantRem, Value &x, int64_t &scale,
                                int64_t &n) {
  auto matchRemDiv = [&](Value dyn, int64_t s) -> bool {
    if (wantRem) {
      if (auto r = dyn.getDefiningOp<arith::RemSIOp>()) {
        auto c = getConstantIntValue(r.getRhs());
        if (c && *c > 0) {
          x = r.getLhs();
          scale = s;
          n = *c;
          return true;
        }
      }
      if (auto r = dyn.getDefiningOp<arith::RemUIOp>()) {
        auto c = getConstantIntValue(r.getRhs());
        if (c && *c > 0) {
          x = r.getLhs();
          scale = s;
          n = *c;
          return true;
        }
      }
      return false;
    }
    if (auto d = dyn.getDefiningOp<arith::DivSIOp>()) {
      auto c = getConstantIntValue(d.getRhs());
      if (c && *c > 0) {
        x = d.getLhs();
        scale = s;
        n = *c;
        return true;
      }
    }
    if (auto d = dyn.getDefiningOp<arith::DivUIOp>()) {
      auto c = getConstantIntValue(d.getRhs());
      if (c && *c > 0) {
        x = d.getLhs();
        scale = s;
        n = *c;
        return true;
      }
    }
    if (auto d = dyn.getDefiningOp<arith::FloorDivSIOp>()) {
      auto c = getConstantIntValue(d.getRhs());
      if (c && *c > 0) {
        x = d.getLhs();
        scale = s;
        n = *c;
        return true;
      }
    }
    return false;
  };

  if (auto mul = v.getDefiningOp<arith::MulIOp>()) {
    if (auto c = getConstantIntValue(mul.getRhs()))
      return *c > 0 && matchRemDiv(mul.getLhs(), *c);
    if (auto c = getConstantIntValue(mul.getLhs()))
      return *c > 0 && matchRemDiv(mul.getRhs(), *c);
    return false;
  }
  // A == 1: bare rem/div (no explicit multiply).
  return matchRemDiv(v, /*s=*/1);
}

/// A*(x%N) + E + (A*N)*(x/N)  →  A*x + E   (also A==1: x%N + N*(x/N) → x)
/// when x is known non-negative. ``E`` is any remaining signed add/sub summands.
/// Rem/div must share the same sign (both + → +A*x; both − → −A*x).
/// Flatten depth is capped (kMaxAddDepth) to keep matching local.
static void strengthReduceEuclideanLaneAddr(gpu::GPUFuncOp fn, UpperBound &bounds) {
  constexpr int kMaxAddDepth = 4;

  SmallVector<Operation *> tips;
  fn.walk([&](Operation *op) {
    if (isa<arith::AddIOp, arith::SubIOp>(op))
      tips.push_back(op);
  });

  for (Operation *tip : tips) {
    if (!tip->getBlock())
      continue;

    SmallVector<SignedSummand, 8> summands;
    flattenAddSubTree(tip->getResult(0), /*sign=*/+1, /*depth=*/0, kMaxAddDepth,
                      summands);
    if (summands.size() < 2)
      continue;

    bool rewritten = false;
    for (size_t i = 0; i < summands.size() && !rewritten; ++i) {
      for (size_t j = 0; j < summands.size() && !rewritten; ++j) {
        if (i == j)
          continue;
        // Rem and div must enter with the same sign.
        if (summands[i].sign != summands[j].sign)
          continue;
        Value xRem, xDiv;
        int64_t aRem = 0, aDiv = 0, nRem = 0, nDiv = 0;
        if (!matchScaledRemOrDiv(summands[i].v, /*wantRem=*/true, xRem, aRem,
                                 nRem))
          continue;
        if (!matchScaledRemOrDiv(summands[j].v, /*wantRem=*/false, xDiv, aDiv,
                                 nDiv))
          continue;
        // A*(x%N) and (A*N)*(x/N) share the same x and N.
        if (xRem != xDiv || nRem != nDiv || aRem <= 0)
          continue;
        if (aDiv != aRem * nRem)
          continue;
        if (!bounds.getExclusive(xRem))
          continue;

        OpBuilder b(tip);
        Location loc = tip->getLoc();
        Type ty = tip->getResult(0).getType();
        Value scaledX = xRem;
        if (aRem != 1) {
          Value scale = arith::ConstantIntOp::create(b, loc, ty, aRem);
          scaledX = arith::MulIOp::create(b, loc, xRem, scale);
        }

        auto applySigned = [&](Value acc, Value v, int s) -> Value {
          if (s > 0)
            return arith::AddIOp::create(b, loc, acc, v);
          return arith::SubIOp::create(b, loc, acc, v);
        };

        Value result;
        if (summands[i].sign > 0) {
          // +A*x + E
          result = scaledX;
          for (size_t k = 0; k < summands.size(); ++k) {
            if (k == i || k == j)
              continue;
            result = applySigned(result, summands[k].v, summands[k].sign);
          }
        } else {
          // E - A*x  (build E first so a leading positive constant stays LHS)
          Value e;
          bool hasE = false;
          for (size_t k = 0; k < summands.size(); ++k) {
            if (k == i || k == j)
              continue;
            if (!hasE) {
              if (summands[k].sign > 0)
                e = summands[k].v;
              else {
                Value zero = arith::ConstantIntOp::create(b, loc, ty, 0);
                e = arith::SubIOp::create(b, loc, zero, summands[k].v);
              }
              hasE = true;
            } else {
              e = applySigned(e, summands[k].v, summands[k].sign);
            }
          }
          if (!hasE) {
            Value zero = arith::ConstantIntOp::create(b, loc, ty, 0);
            result = arith::SubIOp::create(b, loc, zero, scaledX);
          } else {
            result = arith::SubIOp::create(b, loc, e, scaledX);
          }
        }
        tip->getResult(0).replaceAllUsesWith(result);
        tip->erase();
        rewritten = true;
      }
    }
  }
}

/// Match expanded ModSwizzle
///   MS(x) = (x & ~zb) | (((x + ((x & yyy) >> s)) & zb)
/// and fold:
///   - Identity:  MS(x) → x  when exclusive upper bound of x ≤ yyyLo
///                (yyyLo = lowest set bit of yyy; for yyy=768, yyyLo=256).
///   - Second word (MS<2,6,2> only: yyy=768, zb=255, s=2):
///                MS(base+256) → ((base+320)&255)|256  when base < 256.
/// See section comment above for the algebra.
static void simplifyModSwizzle(gpu::GPUFuncOp fn, UpperBound &bounds) {
  SmallVector<arith::OrIOp> modCandidates;
  fn.walk([&](arith::OrIOp ori) { modCandidates.push_back(ori); });

  for (arith::OrIOp ori : modCandidates) {
    if (!ori->getBlock())
      continue;
    // Pattern tip: high | low  with high = x & nzb, low = (x + (yyy>>s)) & zb.
    auto highAnd = ori.getLhs().getDefiningOp<arith::AndIOp>();
    auto lowAnd = ori.getRhs().getDefiningOp<arith::AndIOp>();
    if (!highAnd || !lowAnd)
      continue;
    auto nzbC = getConstantIntValue(highAnd.getRhs());
    auto zbC = getConstantIntValue(lowAnd.getRhs());
    if (!nzbC || !zbC)
      continue;
    uint64_t zb = (uint64_t)*zbC;
    uint64_t nzb = (uint64_t)*nzbC;
    // zb must be a low-bit mask (2^k-1); nzb must be its bitwise complement.
    if (zb == 0 || (zb & (zb + 1)) != 0 || nzb != ~zb)
      continue;

    auto add = lowAnd.getLhs().getDefiningOp<arith::AddIOp>();
    if (!add)
      continue;
    Value x = highAnd.getLhs();
    Value shiftedTerm = add.getLhs() == x ? add.getRhs() : (add.getRhs() == x ? add.getLhs() : Value());
    if (!shiftedTerm)
      continue;
    auto shrui = shiftedTerm.getDefiningOp<arith::ShRUIOp>();
    if (!shrui)
      continue;
    auto yyyAnd = shrui.getLhs().getDefiningOp<arith::AndIOp>();
    if (!yyyAnd || yyyAnd.getLhs() != x)
      continue;
    auto yyyC = getConstantIntValue(yyyAnd.getRhs());
    auto shiftC = getConstantIntValue(shrui.getRhs());
    if (!yyyC || !shiftC || *yyyC == 0 || *shiftC < 0)
      continue;
    uint64_t yyy = (uint64_t)*yyyC;
    // Shifted yyy bits must land inside zb (otherwise not a valid MS shape).
    if (((yyy >> *shiftC) & zb) != (yyy >> *shiftC))
      continue;

    auto hi = bounds.getExclusive(x);
    if (!hi)
      continue;
    // Identity: x < yyyLo ⇒ (x & yyy) == 0 ⇒ MS(x) == x.
    uint64_t yyyLo = yyy & -yyy;
    if (yyyLo != 0 && *hi <= yyyLo) {
      ori.replaceAllUsesWith(x);
      ori.erase();
      continue;
    }

    // Second-word closed form for MS<2,6,2> on x = base+256, base < 256:
    //   (x&768)>>2 == 64, x&~255 == 256
    //   ⇒ MS = ((base+320)&255)|256.
    if (yyy == 768 && zb == 255 && *shiftC == 2) {
      if (auto add256 = x.getDefiningOp<arith::AddIOp>()) {
        auto imm = getConstantIntValue(add256.getRhs());
        Value base = add256.getLhs();
        if (!imm || *imm != 256) {
          imm = getConstantIntValue(add256.getLhs());
          base = add256.getRhs();
        }
        auto baseHi = bounds.getExclusive(base);
        if (imm && *imm == 256 && baseHi && *baseHi <= 256) {
          OpBuilder b(ori);
          Location loc = ori.getLoc();
          Type ty = ori.getType();
          Value c320 = arith::ConstantIntOp::create(b, loc, ty, 320);
          Value c255 = arith::ConstantIntOp::create(b, loc, ty, 255);
          Value c256 = arith::ConstantIntOp::create(b, loc, ty, 256);
          Value sum = arith::AddIOp::create(b, loc, base, c320);
          Value low = arith::AndIOp::create(b, loc, sum, c255);
          Value simplified = arith::OrIOp::create(b, loc, low, c256);
          ori.replaceAllUsesWith(simplified);
          ori.erase();
        }
      }
    }
  }
}

static bool isThreadIdXProducer(Value v) {
  if (auto tid = v.getDefiningOp<gpu::ThreadIdOp>())
    return tid.getDimension() == gpu::Dimension::x;
  return false;
}

static Value matchMulBy(Value v, int64_t scale) {
  auto mul = v.getDefiningOp<arith::MulIOp>();
  if (!mul)
    return Value();
  if (auto c = getConstantIntValue(mul.getRhs()); c && *c == scale)
    return mul.getLhs();
  if (auto c = getConstantIntValue(mul.getLhs()); c && *c == scale)
    return mul.getRhs();
  return Value();
}

/// True when ``v`` is constant across a warp (not ``lane`` / low tid bits).
static bool isWarpUniformAddr(Value v, Value lane, int depth = 0) {
  if (!v || depth > 24)
    return false;
  if (v == lane || isLaneIdProducer(v))
    return false;
  if (getConstantIntValue(v))
    return true;
  if (isa<BlockArgument>(v))
    return false;
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  if (isa<gpu::BlockIdOp, gpu::GridDimOp, gpu::BlockDimOp>(def))
    return true;
  if (auto call = dyn_cast<LLVM::CallIntrinsicOp>(def)) {
    if (call.getIntrin() == "llvm.bi.readfirstlane")
      return true;
  }
  if (isa<arith::IndexCastOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(
          def))
    return isWarpUniformAddr(def->getOperand(0), lane, depth + 1);
  if (isThreadIdXProducer(v))
    return false;

  // TODO: Replace the tidHighBits special-cases below with sampling, in the
  // same spirit as matchLaneAffine/evalLaneExprAt. Treat v as f(tid, lane):
  // for several warp bases W, bind tid = 64*W+lane over lane∈[0,63] and check
  // that f is constant within each W (eval failure → divergent). Do not sample
  // only tid∈[0,63] (misses false positives like tid/100 across warps).
  //
  // warp_id = tid >> 6  (or tid / 64) is warp-uniform.
  auto tidHighBits = [&](Value x) -> bool {
    if (auto c = x.getDefiningOp<arith::IndexCastOp>())
      x = c.getIn();
    return isThreadIdXProducer(x);
  };
  if (auto shr = dyn_cast<arith::ShRUIOp>(def)) {
    auto s = getConstantIntValue(shr.getRhs());
    if (s && *s >= 6 && tidHighBits(shr.getLhs()))
      return true;
  }
  if (auto shr = dyn_cast<arith::ShRSIOp>(def)) {
    auto s = getConstantIntValue(shr.getRhs());
    if (s && *s >= 6 && tidHighBits(shr.getLhs()))
      return true;
  }
  if (auto div = dyn_cast<arith::DivSIOp>(def)) {
    auto d = getConstantIntValue(div.getRhs());
    if (d && *d >= 64 && (*d & (*d - 1)) == 0 && tidHighBits(div.getLhs()))
      return true;
  }
  if (auto div = dyn_cast<arith::DivUIOp>(def)) {
    auto d = getConstantIntValue(div.getRhs());
    if (d && *d >= 64 && (*d & (*d - 1)) == 0 && tidHighBits(div.getLhs()))
      return true;
  }
  if (auto div = dyn_cast<arith::FloorDivSIOp>(def)) {
    auto d = getConstantIntValue(div.getRhs());
    if (d && *d >= 64 && (*d & (*d - 1)) == 0 && tidHighBits(div.getLhs()))
      return true;
  }
  if (auto andi = dyn_cast<arith::AndIOp>(def)) {
    // Masking off low 6 bits of tid → warp-uniform high bits.
    if (auto m = getConstantIntValue(andi.getRhs()); m && (*m & 63) == 0) {
      if (tidHighBits(andi.getLhs()))
        return true;
    }
    if (auto m = getConstantIntValue(andi.getLhs()); m && (*m & 63) == 0) {
      if (tidHighBits(andi.getRhs()))
        return true;
    }
    // tid & 63 is lane-dependent.
    if (auto m = getConstantIntValue(andi.getRhs()); m && *m == 63)
      return false;
    if (auto m = getConstantIntValue(andi.getLhs()); m && *m == 63)
      return false;
  }
  if (isa<arith::RemSIOp, arith::RemUIOp>(def)) {
    if (auto m = getConstantIntValue(def->getOperand(1)); m && *m == 64)
      return false;
  }
  if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::AndIOp,
          arith::OrIOp, arith::XOrIOp, arith::ShLIOp, arith::DivSIOp,
          arith::DivUIOp, arith::FloorDivSIOp, arith::SelectOp>(def)) {
    for (Value opr : def->getOperands()) {
      if (!isWarpUniformAddr(opr, lane, depth + 1))
        return false;
    }
    return true;
  }
  return false;
}

static bool isLaneTimes4(Value v, Value lane) {
  Value scaled = matchMulBy(v, 4);
  return scaled && scaled == lane;
}

/// Collapse trivial MS residue when ``k==0``: ``(4*lane)&255`` → ``4*lane``.
static void collapseTrivialLane4Masks(gpu::GPUFuncOp fn) {
  Value lane = findI32LaneId(fn);
  if (!lane)
    return;

  SmallVector<arith::AndIOp> andis;
  fn.walk([&](arith::AndIOp op) { andis.push_back(op); });
  for (arith::AndIOp op : andis) {
    if (!op->getBlock())
      continue;
    Value x = op.getLhs();
    auto m = getConstantIntValue(op.getRhs());
    if (!m || *m != 255) {
      x = op.getRhs();
      m = getConstantIntValue(op.getLhs());
      if (!m || *m != 255)
        continue;
    }
    if (!isLaneTimes4(x, lane))
      continue;
    op.replaceAllUsesWith(x);
    op.erase();
  }
}

/// Skip combines that still feed ModSwizzle bit math.
static bool isSharedAddrRootCombine(Operation *op) {
  for (OpOperand &use : op->getResult(0).getUses()) {
    Operation *user = use.getOwner();
    if (isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp, arith::ShRUIOp,
            arith::ShRSIOp, arith::ShLIOp>(user))
      return false;
  }
  return true;
}

/// ``base + 4*lane`` / ``base + (4*lane + U)`` → ``readfirstlane(base[+U]) + 4*lane``.
static void promoteLane4BaseWithReadFirstLane(gpu::GPUFuncOp fn) {
  Value lane = findI32LaneId(fn);
  if (!lane)
    return;

  fn->getContext()->loadDialect<LLVM::LLVMDialect>();

  auto isRFL = [](Value v) -> bool {
    if (auto call = v.getDefiningOp<LLVM::CallIntrinsicOp>())
      return call.getIntrin() == "llvm.bi.readfirstlane";
    return false;
  };

  DenseMap<Value, Value> rflOf;

  auto getRFL = [&](OpBuilder &builder, Location loc, Value v) -> Value {
    if (isRFL(v))
      return v;
    if (auto it = rflOf.find(v); it != rflOf.end())
      return it->second;
    auto rf = LLVM::CallIntrinsicOp::create(
        builder, loc, v.getType(),
        builder.getStringAttr("llvm.bi.readfirstlane"), ValueRange{v});
    Value uni = rf.getResult(0);
    rflOf[v] = uni;
    return uni;
  };

  auto matchLane4Term = [&](Value v) -> std::optional<std::pair<Value, Value>> {
    if (isLaneTimes4(v, lane))
      return std::make_pair(v, Value());
    auto add = v.getDefiningOp<arith::AddIOp>();
    if (!add)
      return std::nullopt;
    Value x = add.getLhs();
    Value y = add.getRhs();
    if (isLaneTimes4(x, lane) && isWarpUniformAddr(y, lane))
      return std::make_pair(x, y);
    if (isLaneTimes4(y, lane) && isWarpUniformAddr(x, lane))
      return std::make_pair(y, x);
    return std::nullopt;
  };

  SmallVector<Operation *> candidates;
  fn.walk([&](Operation *op) {
    if (isa<arith::AddIOp, arith::OrIOp>(op))
      candidates.push_back(op);
  });

  for (Operation *op : candidates) {
    if (!op->getBlock())
      continue;
    if (!isSharedAddrRootCombine(op))
      continue;
    Value a = op->getOperand(0);
    Value b = op->getOperand(1);
    Value base, lane4, extra;
    auto matchSide = [&](Value maybeBase, Value maybeTerm) -> bool {
      if (!isWarpUniformAddr(maybeBase, lane))
        return false;
      auto term = matchLane4Term(maybeTerm);
      if (!term)
        return false;
      if (isa<arith::OrIOp>(op) && term->second)
        return false;
      base = maybeBase;
      lane4 = term->first;
      extra = term->second;
      return true;
    };
    if (!matchSide(a, b) && !matchSide(b, a))
      continue;

    OpBuilder builder(op);
    Location loc = op->getLoc();
    Value uniSrc = base;
    if (extra)
      uniSrc = arith::AddIOp::create(builder, loc, base, extra).getResult();
    if (isRFL(uniSrc) && !extra)
      continue;
    // Constants are already warp-uniform; skip a useless readfirstlane.
    Value uni = getConstantIntValue(uniSrc) ? uniSrc : getRFL(builder, loc, uniSrc);

    if (!extra) {
      if (base == a)
        op->setOperand(0, uni);
      else
        op->setOperand(1, uni);
      continue;
    }
    Value sum = arith::AddIOp::create(builder, loc, uni, lane4).getResult();
    op->getResult(0).replaceAllUsesWith(sum);
    op->erase();
  }
}

class IXDLAddressSimplifyPass
    : public mlir::impl::IXDLAddressSimplifyPassBase<IXDLAddressSimplifyPass> {
public:
  using mlir::impl::IXDLAddressSimplifyPassBase<
      IXDLAddressSimplifyPass>::IXDLAddressSimplifyPassBase;

  void runOnOperation() override {
    auto module = getOperation();
    module->walk([&](gpu::GPUModuleOp gpuModule) {
      gpuModule.walk([&](gpu::GPUFuncOp fn) {
        optAddrInFunc(fn, /*warpSize=*/64);
        UpperBound bounds;
        // Euclidean first so ModSwizzle sees `4*lane`.
        strengthReduceEuclideanLaneAddr(fn, bounds);
        simplifyModSwizzle(fn, bounds);
        collapseTrivialLane4Masks(fn);
        promoteLane4BaseWithReadFirstLane(fn);
      });
    });
  }
};

} // namespace
