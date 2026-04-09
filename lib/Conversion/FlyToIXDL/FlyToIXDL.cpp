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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/StringSet.h"

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
  MakePtrOpLowering(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<MakePtrOp>(typeConverter, context) {}

  LogicalResult matchAndRewrite(MakePtrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto flyPtrTy = dyn_cast<fly::PointerType>(op.getResult().getType());
    if (!flyPtrTy)
      return failure();

    Location loc = op.getLoc();
    AddressSpace addrSpace = flyPtrTy.getAddressSpace().getValue();

    if (addrSpace == AddressSpace::Register) {
      auto dictAttrs = op.getDictAttrs();
      if (!dictAttrs)
        return rewriter.notifyMatchFailure(op, "register make_ptr requires dictAttrs");
      auto allocSize = dictAttrs->getAs<IntegerAttr>("allocaSize");
      if (!allocSize)
        return rewriter.notifyMatchFailure(op, "register make_ptr requires allocaSize");
      unsigned llvmAS = mapAddressSpace(AddressSpace::Register);
      auto llvmPtrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), llvmAS);
      Value nElems = arith::ConstantIntOp::create(rewriter, loc, allocSize.getInt(), 64);
      Value ptr = LLVM::AllocaOp::create(rewriter, loc, llvmPtrTy, flyPtrTy.getElemTy(), nElems, 0);
      rewriter.replaceOp(op, ptr);
      return success();
    }

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

    return rewriter.notifyMatchFailure(op, "unsupported make_ptr variant for IXDL");
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
    if (isa<fly::CoordTensorType>(op.getResult().getType())) {
      if (!op.getResult().use_empty())
        return rewriter.notifyMatchFailure(op, "coord_tensor result should have no uses");
      rewriter.eraseOp(op);
      return success();
    }
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

class GetIterOpLowering : public OpConversionPattern<GetIterOp> {
public:
  GetIterOpLowering(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<GetIterOp>(typeConverter, context) {}

  LogicalResult matchAndRewrite(GetIterOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getMemref());
    return success();
  }
};

class ApplySwizzleOpLowering : public OpConversionPattern<ApplySwizzleOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ApplySwizzleOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getPtr());
    return success();
  }
};

class RecastIterOpLowering : public OpConversionPattern<RecastIterOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(RecastIterOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getSrc());
    return success();
  }
};

class IntToPtrOpLowering : public OpConversionPattern<IntToPtrOp> {
public:
  IntToPtrOpLowering(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<IntToPtrOp>(typeConverter, context) {}

  LogicalResult matchAndRewrite(IntToPtrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto flyPtrTy = dyn_cast<fly::PointerType>(op.getResult().getType());
    if (!flyPtrTy)
      return failure();
    auto resultTy = dyn_cast<LLVM::LLVMPointerType>(getTypeConverter()->convertType(flyPtrTy));
    if (!resultTy)
      return failure();
    rewriter.replaceOpWithNewOp<LLVM::IntToPtrOp>(op, resultTy, adaptor.getSrc());
    return success();
  }
};

class PtrToIntOpLowering : public OpConversionPattern<PtrToIntOp> {
public:
  PtrToIntOpLowering(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<PtrToIntOp>(typeConverter, context) {}

  LogicalResult matchAndRewrite(PtrToIntOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultTy)
      return failure();
    rewriter.replaceOpWithNewOp<LLVM::PtrToIntOp>(op, resultTy, adaptor.getPtr());
    return success();
  }
};

class PtrLoadOpLowering : public OpConversionPattern<PtrLoadOp> {
public:
  using OpConversionPattern<PtrLoadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(PtrLoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto flyPtrTy = dyn_cast<fly::PointerType>(op.getPtr().getType());
    if (!flyPtrTy)
      return failure();
    Type elemTy = flyPtrTy.getElemTy();
    Value loaded = LLVM::LoadOp::create(rewriter, op.getLoc(), elemTy, adaptor.getPtr());
    rewriter.replaceOp(op, loaded);
    return success();
  }
};

class PtrStoreOpLowering : public OpConversionPattern<PtrStoreOp> {
public:
  using OpConversionPattern<PtrStoreOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(PtrStoreOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto flyPtrTy = dyn_cast<fly::PointerType>(op.getPtr().getType());
    if (!flyPtrTy)
      return failure();
    LLVM::StoreOp::create(rewriter, op.getLoc(), adaptor.getValue(), adaptor.getPtr());
    rewriter.eraseOp(op);
    return success();
  }
};

class GetDynSharedOpLowering : public OpConversionPattern<GetDynSharedOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(GetDynSharedOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto flyPtrTy = cast<fly::PointerType>(op.getResult().getType());
    unsigned addrSpace = mapAddressSpace(flyPtrTy.getAddressSpace().getValue());

    auto moduleOp = op->getParentOfType<gpu::GPUModuleOp>();
    if (!moduleOp)
      return op->emitError("get_dyn_shared must be inside a gpu.module");

    LLVM::GlobalOp sharedGlobal = getOrCreateDynSharedGlobal(rewriter, moduleOp, loc, addrSpace);

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);

    auto basePtr = LLVM::AddressOfOp::create(rewriter, loc, sharedGlobal);
    Type ptrType = basePtr->getResultTypes()[0];

    auto i8Ty = IntegerType::get(rewriter.getContext(), 8);
    Value sharedPtr =
        LLVM::GEPOp::create(rewriter, loc, ptrType, i8Ty, basePtr, ArrayRef<LLVM::GEPArg>{0});

    rewriter.replaceOp(op, sharedPtr);
    return success();
  }

private:
  static LLVM::GlobalOp getOrCreateDynSharedGlobal(ConversionPatternRewriter &rewriter,
                                                   gpu::GPUModuleOp moduleOp, Location loc,
                                                   unsigned addrSpace) {
    llvm::StringSet<> existingNames;
    for (auto globalOp : moduleOp.getBody()->getOps<LLVM::GlobalOp>()) {
      existingNames.insert(globalOp.getSymName());
      if (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(globalOp.getType())) {
        if (globalOp.getAddrSpace() == addrSpace && arrayType.getNumElements() == 0 &&
            globalOp.getAlignment().value_or(0) == 1024)
          return globalOp;
      }
    }

    unsigned counter = 0;
    SmallString<128> symName = SymbolTable::generateSymbolName<128>(
        "__dynamic_shared_", [&](StringRef candidate) { return existingNames.contains(candidate); },
        counter);

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());

    auto zeroArrayTy = LLVM::LLVMArrayType::get(IntegerType::get(rewriter.getContext(), 8), 0);

    auto globalOp = LLVM::GlobalOp::create(rewriter, loc, zeroArrayTy,
                                           /*isConstant=*/false, LLVM::Linkage::External, symName,
                                           /*value=*/Attribute(),
                                           /*alignment=*/1024, addrSpace);
    globalOp.setDsoLocal(true);
    return globalOp;
  }
};

class ExtractAlignedPointerAsIndexLowering
    : public OpConversionPattern<ExtractAlignedPointerAsIndexOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ExtractAlignedPointerAsIndexOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src = adaptor.getSource();
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      resultType = op.getResult().getType();
    if (src.getType() != resultType)
      src = LLVM::AddrSpaceCastOp::create(rewriter, op.getLoc(), resultType, src);
    rewriter.replaceOp(op, src);
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
      int64_t copyBytes = numValSrc * copyAtom.getValBits() / 8;
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
    target.addLegalOp<StaticOp, MakeIntTupleOp, MakeLayoutOp, MakeTileOp, MakeComposedLayoutOp,
                      MakeMmaAtomOp, MakeCopyAtomOp, MakeTiledCopyOp, MakeTiledMmaOp>();

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

    patterns.add<MakePtrOpLowering, GetDynSharedOpLowering>(typeConverter, context);
    patterns.add<IntToPtrOpLowering, PtrToIntOpLowering>(typeConverter, context);
    patterns.add<GetIterOpLowering, ApplySwizzleOpLowering, RecastIterOpLowering>(typeConverter,
                                                                                  context);
    patterns.add<MemRefAllocOpLowering, MakeViewOpLowering, AddOffsetOpLowering>(typeConverter,
                                                                                 context);
    patterns.add<MemRefLoadVecOpLowering, MemRefStoreVecOpLowering>(typeConverter, context);
    patterns.add<PtrLoadOpLowering, PtrStoreOpLowering>(typeConverter, context);
    patterns.add<CopyAtomCallLowering, GpuLaunchFuncOpLowering>(typeConverter, context);
    patterns.add<ExtractAlignedPointerAsIndexLowering>(typeConverter, context);

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
