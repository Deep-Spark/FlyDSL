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
#define GEN_PASS_DEF_FLYFOLDSTATICPASS
#include "flydsl/Dialect/Fly/Transforms/Passes.h.inc"
} // namespace fly
} // namespace mlir

namespace {

class RebuildStaticValue : public RewritePattern {
public:
  RebuildStaticValue(MLIRContext *context, PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1)
      return failure();
    Type resultType = op->getResult(0).getType();

    auto mayStatic = dyn_cast<MayStaticTypeInterface>(resultType);
    if (!mayStatic || !mayStatic.isStatic())
      return failure();

    Value rebuild = mayStatic.rebuildStaticValue(rewriter, op->getLoc(), op->getResult(0));
    if (!rebuild)
      return failure();

    rewriter.replaceOp(op, rebuild);
    return success();
  }
};

class FlyFoldStaticPass : public mlir::fly::impl::FlyFoldStaticPassBase<FlyFoldStaticPass> {
public:
  using mlir::fly::impl::FlyFoldStaticPassBase<FlyFoldStaticPass>::FlyFoldStaticPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<RebuildStaticValue>(context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
