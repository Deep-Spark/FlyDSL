#include "flydsl/Conversion/FlyToIXDL/FlyToIXDL.h"
#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/IntTupleUtils.h"
#include "flydsl/Dialect/Fly/Utils/LayoutUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_FLYTOIXDLCONVERSIONPASS
#include "flydsl/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::fly;

namespace {

static unsigned mapAddressSpace(AddressSpace space) {
  switch (space) {
  case AddressSpace::Global:
    return 1;
  case AddressSpace::Shared:
    return 3;
  case AddressSpace::Register:
    return 5;
  case AddressSpace::BufferDesc:
    // Fallback to global for IXDL if no special buffer descriptor exists
    return 1;
  }
  return 0;
}

static FailureOr<Value> toI64(Value v, Location loc, ConversionPatternRewriter &rewriter) {
  Type i64Ty = rewriter.getI64Type();
  if (v.getType() == i64Ty)
    return v;
  if (v.getType().isIndex())
    return arith::IndexCastOp::create(rewriter, loc, i64Ty, v).getResult();
  if (auto intTy = dyn_cast<IntegerType>(v.getType())) {
    if (intTy.getWidth() < 64)
      return arith::ExtSIOp::create(rewriter, loc, i64Ty, v).getResult();
    if (intTy.getWidth() > 64)
      return arith::TruncIOp::create(rewriter, loc, i64Ty, v).getResult();
  }
  return failure();
}

static FailureOr<Value> materializeScalarIndex(Value intTuple, Location loc,
                                               ConversionPatternRewriter &rewriter) {
  auto tupleTy = dyn_cast<fly::IntTupleType>(intTuple.getType());
  if (!tupleTy)
    return failure();

  IntTupleAttr profile = tupleTy.getAttr();
  if (!profile.isLeaf())
    return failure();

  if (auto intAttr = dyn_cast<IntAttr>(profile.getValue())) {
    if (intAttr.isStatic()) {
      return (Value)arith::ConstantIndexOp::create(rewriter, loc, intAttr.getValue());
    }
  }
  if (profile.getLeafAsInt().isNone()) {
    return (Value)arith::ConstantIndexOp::create(rewriter, loc, 0);
  }

  if (Operation *defOp = intTuple.getDefiningOp()) {
    if (defOp->getName().getStringRef() == "fly.make_int_tuple" && defOp->getNumOperands() == 1) {
      Value v = defOp->getOperand(0);
      if (v.getType().isIndex())
        return v;
      if (v.getType().isSignlessInteger())
        return arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), v).getResult();
    }
  }

  return failure();
}

class FlyTypeConverter : public TypeConverter {
public:
  FlyTypeConverter() {
    addConversion([](Type type) { return type; });

    addConversion([&](fly::MemRefType flyMemRefTy) -> Type {
      unsigned as = mapAddressSpace(flyMemRefTy.getAddressSpace().getValue());
      return LLVM::LLVMPointerType::get(flyMemRefTy.getContext(), as);
    });
    addConversion([&](fly::PointerType flyPtrTy) -> Type {
      unsigned as = mapAddressSpace(flyPtrTy.getAddressSpace().getValue());
      return LLVM::LLVMPointerType::get(flyPtrTy.getContext(), as);
    });
  }
};

class MakePtrOpLowering : public OpConversionPattern<MakePtrOp> {
public:
  using OpConversionPattern<MakePtrOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(MakePtrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto flyPtrTy = dyn_cast<fly::PointerType>(op.getResult().getType());
    if (!flyPtrTy)
      return failure();

    auto resultTy = dyn_cast<LLVM::LLVMPointerType>(getTypeConverter()->convertType(flyPtrTy));
    if (!resultTy)
      return failure();

    auto args = adaptor.getArgs();
    if (args.size() == 1) {
      Value src = args[0];
      if (src.getType() == resultTy) {
        rewriter.replaceOp(op, src);
        return success();
      }
      rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, resultTy, src);
      return success();
    }

    return failure();
  }
};

class MemRefAllocOpLowering : public OpConversionPattern<MemRefAllocaOp> {
public:
  using OpConversionPattern<MemRefAllocaOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(MemRefAllocaOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto flyMemRefTy = dyn_cast<fly::MemRefType>(op.getResult().getType());
    if (!flyMemRefTy)
      return failure();

    LayoutAttr layoutAttr = cast<LayoutAttr>(flyMemRefTy.getLayout());
    auto elemTy = flyMemRefTy.getElemTy();

    LayoutBuilder<LayoutAttr> builder(rewriter.getContext());
    IntTupleAttr totalSize = layoutCosize(builder, layoutAttr);

    assert(totalSize.isStatic() && totalSize.isLeaf());

    auto convertedPtrTy =
        dyn_cast<LLVM::LLVMPointerType>(getTypeConverter()->convertType(flyMemRefTy));
    if (!convertedPtrTy)
      return failure();

    auto loc = op.getLoc();
    Value nElems = arith::ConstantIntOp::create(rewriter, loc, totalSize.getLeafAsInt().getValue(), 64).getResult();
    Value ptr = LLVM::AllocaOp::create(rewriter, loc, convertedPtrTy, elemTy, nElems, 0);
    rewriter.replaceOp(op, ptr);
    return success();
  }
};

class MakeViewOpLowering : public OpConversionPattern<MakeViewOp> {
public:
  MakeViewOpLowering(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<MakeViewOp>(typeConverter, context) {}

  LogicalResult matchAndRewrite(MakeViewOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value base = adaptor.getIter();
    Type resultTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultTy)
      return failure();
    if (base.getType() == resultTy) {
      rewriter.replaceOp(op, base);
      return success();
    }
    if (isa<LLVM::LLVMPointerType>(base.getType()) && isa<LLVM::LLVMPointerType>(resultTy)) {
      rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, resultTy, base);
      return success();
    }
    return failure();
  }
};

class AddOffsetOpLowering : public OpConversionPattern<AddOffsetOp> {
public:
  using OpConversionPattern<AddOffsetOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(AddOffsetOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value base = adaptor.getPtr();
    auto flyPtrTy = dyn_cast<fly::PointerType>(op.getPtr().getType());
    if (!flyPtrTy)
      return failure();

    auto offsetIdx = materializeScalarIndex(op.getOffset(), loc, rewriter);
    if (failed(offsetIdx))
      return failure();

    auto resultTy = dyn_cast<LLVM::LLVMPointerType>(getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultTy)
      return failure();

    FailureOr<Value> offsetI64 = toI64(*offsetIdx, loc, rewriter);
    if (failed(offsetI64))
      return failure();

    Type elemTy = flyPtrTy.getElemTy();
    Value gep = LLVM::GEPOp::create(rewriter, loc, resultTy, elemTy, base, ValueRange{*offsetI64});
    rewriter.replaceOp(op, gep);
    return success();
  }
};

class MemRefLoadVecOpLowering : public OpConversionPattern<MemRefLoadVecOp> {
public:
  using OpConversionPattern<MemRefLoadVecOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(MemRefLoadVecOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = adaptor.getMemref();
    auto resVecTy = dyn_cast<VectorType>(op.getResult().getType());
    if (!resVecTy)
      return failure();

    Value loaded = LLVM::LoadOp::create(rewriter, loc, resVecTy, input);
    rewriter.replaceOp(op, loaded);
    return success();
  }
};

class MemRefStoreVecOpLowering : public OpConversionPattern<MemRefStoreVecOp> {
public:
  using OpConversionPattern<MemRefStoreVecOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(MemRefStoreVecOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value dest = adaptor.getMemref();
    Value valueToStore = adaptor.getVector();

    LLVM::StoreOp::create(rewriter, loc, valueToStore, dest);
    rewriter.eraseOp(op);
    return success();
  }
};

class CopyAtomCallLowering : public OpConversionPattern<CopyAtomCall> {
public:
  using OpConversionPattern<CopyAtomCall>::OpConversionPattern;

  LogicalResult matchAndRewrite(CopyAtomCall op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto copyAtom = dyn_cast<CopyAtomType>(op.getCopyAtom().getType());
    if (!copyAtom)
      return failure();

    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();
    Value pred = adaptor.getPred();

    auto srcFlyTy = dyn_cast<fly::MemRefType>(op.getSrc().getType());
    if (!srcFlyTy)
      return failure();

    Location loc = op.getLoc();

    auto emitCopyBody = [&](ConversionPatternRewriter &rewriter) -> LogicalResult {
      LayoutBuilder<LayoutAttr> attrBuilder(rewriter.getContext());
      auto thrValLayoutSrc = dyn_cast<LayoutAttr>(copyAtom.getThrValLayoutSrc());
      if (!thrValLayoutSrc)
        return failure();
      
      int64_t numValSrc = intTupleProduct(attrBuilder, thrValLayoutSrc.getShape().at(1)).getLeafAsInt().getValue();
      Type elemTy = srcFlyTy.getElemTy();
      
      int64_t elemBits = 0;
      if (auto ft = dyn_cast<FloatType>(elemTy)) elemBits = ft.getWidth();
      else if (auto it = dyn_cast<IntegerType>(elemTy)) elemBits = it.getWidth();
      
      int64_t copyBytes = numValSrc * elemBits / 8;
      Value len = arith::ConstantIntOp::create(rewriter, loc, copyBytes, 64).getResult();
      LLVM::MemcpyOp::create(rewriter, loc, dst, src, len, false);
      rewriter.eraseOp(op);
      return success();
    };

    if (pred) {
      auto predFlyTy = dyn_cast<fly::MemRefType>(op.getPred().getType());
      Value predVal = LLVM::LoadOp::create(rewriter, loc, predFlyTy.getElemTy(), pred);
      auto ifOp = scf::IfOp::create(rewriter, loc, TypeRange{}, predVal, false);
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      auto res = emitCopyBody(rewriter);
      return res;
    }
    return emitCopyBody(rewriter);
  }
};

class GpuLaunchFuncOpLowering : public OpConversionPattern<gpu::LaunchFuncOp> {
public:
  using OpConversionPattern<gpu::LaunchFuncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(gpu::LaunchFuncOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto grid = gpu::KernelDim3{adaptor.getGridSizeX(), adaptor.getGridSizeY(), adaptor.getGridSizeZ()};
    auto block = gpu::KernelDim3{adaptor.getBlockSizeX(), adaptor.getBlockSizeY(), adaptor.getBlockSizeZ()};

    Type asyncTokenType = nullptr;
    if (op.getAsyncToken())
      asyncTokenType = op.getAsyncToken().getType();

    // Use the type-converted operands from the adaptor instead of original operands.
    // The adaptor already contains the converted values (llvm.ptr instead of fly.ptr).
    rewriter.replaceOpWithNewOp<gpu::LaunchFuncOp>(
        op, adaptor.getKernel(), grid, block, adaptor.getDynamicSharedMemorySize(),
        adaptor.getKernelOperands(), asyncTokenType, adaptor.getAsyncDependencies());
    return success();
  }
};

struct FlyToIXDLConversionPass
    : public mlir::impl::FlyToIXDLConversionPassBase<FlyToIXDLConversionPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    target.addLegalDialect<arith::ArithDialect, scf::SCFDialect, vector::VectorDialect,
                           gpu::GPUDialect, func::FuncDialect, LLVM::LLVMDialect>();
    target.addIllegalDialect<fly::FlyDialect>();
    target.addLegalOp<StaticOp, MakeIntTupleOp, MakeLayoutOp, MakeTileOp, MakeComposedLayoutOp, MakeCopyAtomOp>();

    FlyTypeConverter typeConverter;
    
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) { return typeConverter.isSignatureLegal(op.getFunctionType()); });
    target.addDynamicallyLegalOp<gpu::GPUFuncOp>([&](gpu::GPUFuncOp op) { return typeConverter.isSignatureLegal(op.getFunctionType()); });

    target.addDynamicallyLegalOp<gpu::LaunchFuncOp>([&](gpu::LaunchFuncOp op) {
      auto isValueLegal = [&](Value v) {
        if (!v)
          return true;
        return typeConverter.isLegal(v.getType());
      };

      for (Value v : op.getKernelOperands())
        if (!isValueLegal(v))
          return false;

      if (!isValueLegal(op.getDynamicSharedMemorySize()))
        return false;

      for (Value dep : op.getAsyncDependencies())
        if (!isValueLegal(dep))
          return false;
      if (op.getAsyncObject() && !isValueLegal(op.getAsyncObject()))
        return false;

      return true;
    });

    patterns.add<MakePtrOpLowering, MemRefAllocOpLowering, MakeViewOpLowering, AddOffsetOpLowering, 
                 MemRefLoadVecOpLowering, MemRefStoreVecOpLowering, 
                 CopyAtomCallLowering, GpuLaunchFuncOpLowering>(typeConverter, context);

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
    populateFunctionOpInterfaceTypeConversionPattern<gpu::GPUFuncOp>(patterns, typeConverter);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::fly::createFlyToIXDLConversionPass() {
  return std::make_unique<FlyToIXDLConversionPass>();
}
