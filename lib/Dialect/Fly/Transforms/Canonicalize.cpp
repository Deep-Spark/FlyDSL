// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::fly;

namespace mlir {
namespace fly {
#define GEN_PASS_DEF_FLYCANONICALIZEPASS
#include "flydsl/Dialect/Fly/Transforms/Passes.h.inc"
} // namespace fly
} // namespace mlir

namespace {

template <typename IntTupleLikeOp>
class RewriteToMakeIntTuple final : public OpRewritePattern<IntTupleLikeOp> {
  using OpRewritePattern<IntTupleLikeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IntTupleLikeOp op, PatternRewriter &rewriter) const override {
    auto newOp = MakeIntTupleOp::create(rewriter, op.getLoc(), op.getResult().getType(),
                                        op->getOperands(), op->getAttrs());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

class FlyCanonicalizePass : public mlir::fly::impl::FlyCanonicalizePassBase<FlyCanonicalizePass> {
public:
  using mlir::fly::impl::FlyCanonicalizePassBase<FlyCanonicalizePass>::FlyCanonicalizePassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<RewriteToMakeIntTuple<MakeShapeOp>, RewriteToMakeIntTuple<MakeStrideOp>,
                 RewriteToMakeIntTuple<MakeCoordOp>>(context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
