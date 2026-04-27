
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/IXDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "flydsl/Conversion/FlyToIXDL/FlyToIXDL.h"
#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/IntTupleUtils.h"
#include "flydsl/Dialect/Fly/Utils/LayoutUtils.h"
#include "flydsl/Dialect/FlyIXDL/IR/Dialect.h"

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
    return IXDL::kGlobalMemorySpace;
  case AddressSpace::Shared:
    return IXDL::kSharedMemorySpace;
  case AddressSpace::Register:
    return IXDL::kPrivateMemroySpace;
  case AddressSpace::BufferDesc:
    return 0;
  }
  return 0;
}


static FailureOr<Value> toI64(Value v, Location loc,
                              ConversionPatternRewriter &rewriter) {
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

static FailureOr<Value> materializeScalarIndex(
    Value intTuple, Location loc, ConversionPatternRewriter &rewriter) {
  auto tupleTy = dyn_cast<fly::IntTupleType>(intTuple.getType());
  if (!tupleTy)
    return failure();
  IntTupleAttr profile = tupleTy.getAttr();
  if (!profile.isLeaf())
    return failure();
  if (auto intAttr = dyn_cast<IntAttr>(profile.getValue())) {
    if (intAttr.isStatic())
      return arith::ConstantIndexOp::create(rewriter, loc, intAttr.getValue())
          .getResult();
  }
  if (profile.getLeafAsInt().isNone())
    return arith::ConstantIndexOp::create(rewriter, loc, 0).getResult();
  if (Operation *defOp = intTuple.getDefiningOp()) {
    if (defOp->getName().getStringRef() == "fly.make_int_tuple" &&
        defOp->getNumOperands() == 1) {
      Value v = defOp->getOperand(0);
      if (v.getType().isIndex())
        return v;
      if (v.getType().isSignlessInteger())
        return arith::IndexCastOp::create(rewriter, loc,
                                          rewriter.getIndexType(), v)
            .getResult();
    }
  }
  return failure();
}

// ---------- Generic Fly op lowerings (backend-agnostic) ----------

class MakePtrOpLowering : public OpConversionPattern<MakePtrOp> {
public:
  using OpConversionPattern<MakePtrOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(MakePtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto flyPtrTy = dyn_cast<fly::PointerType>(op.getResult().getType());
    if (!flyPtrTy)
      return failure();
    AddressSpace addrSpace = flyPtrTy.getAddressSpace().getValue();
    auto args = adaptor.getArgs();
    if (addrSpace == AddressSpace::BufferDesc)
      return rewriter.notifyMatchFailure(
          op, "BufferDesc not supported in IXDL lowering");
    auto resultTy = dyn_cast<LLVM::LLVMPointerType>(
        getTypeConverter()->convertType(flyPtrTy));
    if (!resultTy)
      return failure();
    if (args.size() == 1) {
      Value src = args[0];
      if (src.getType() == resultTy) {
        rewriter.replaceOp(op, src);
        return success();
      }
      rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, resultTy, src);
      return success();
    }
    return rewriter.notifyMatchFailure(op,
                                       "unsupported make_ptr operand count");
  }
};

class MemRefAllocOpLowering : public OpConversionPattern<MemRefAllocaOp> {
public:
  using OpConversionPattern<MemRefAllocaOp>::OpConversionPattern;

  mutable unsigned smemCounter = 0;

  LogicalResult
  matchAndRewrite(MemRefAllocaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto flyMemRefTy = dyn_cast<fly::MemRefType>(op.getResult().getType());
    if (!flyMemRefTy)
      return failure();
    LayoutAttr layoutAttr = cast<LayoutAttr>(flyMemRefTy.getLayout());
    auto elemTy = flyMemRefTy.getElemTy();
    LayoutBuilder<LayoutAttr> builder(rewriter.getContext());
    IntTupleAttr totalSize = layoutCosize(builder, layoutAttr);
    assert(totalSize.isStatic() && totalSize.isLeaf());
    auto convertedPtrTy = dyn_cast<LLVM::LLVMPointerType>(
        getTypeConverter()->convertType(flyMemRefTy));
    if (!convertedPtrTy)
      return failure();
    auto loc = op.getLoc();
    int64_t nElemsVal = totalSize.getLeafAsInt().getValue();

    AddressSpace addrSpace = flyMemRefTy.getAddressSpace().getValue();
    if (addrSpace == AddressSpace::Shared) {
      return lowerSharedMemoryAlloc(op, rewriter, loc, elemTy, nElemsVal,
                                    convertedPtrTy);
    }

    Value nElems =
        arith::ConstantIntOp::create(rewriter, loc, nElemsVal, 64)
            .getResult();
    Value ptr = LLVM::AllocaOp::create(rewriter, loc, convertedPtrTy, elemTy,
                                       nElems, 0);
    int64_t elemBits = 0;
    if (auto ft = dyn_cast<FloatType>(elemTy))
      elemBits = ft.getWidth();
    else if (auto it = dyn_cast<IntegerType>(elemTy))
      elemBits = it.getWidth();
    if (elemBits > 0) {
      int64_t totalBytes = nElemsVal * elemBits / 8;
      Value len = arith::ConstantIntOp::create(rewriter, loc, totalBytes, 64)
                      .getResult();
      Value zero =
          arith::ConstantIntOp::create(rewriter, loc, 0, 8).getResult();
      LLVM::MemsetOp::create(rewriter, loc, ptr, zero, len, false);
    }
    rewriter.replaceOp(op, ptr);
    return success();
  }

private:
  LogicalResult
  lowerSharedMemoryAlloc(MemRefAllocaOp op, ConversionPatternRewriter &rewriter,
                         Location loc, Type elemTy, int64_t nElems,
                         LLVM::LLVMPointerType ptrTy) const {
    Block *moduleBody = nullptr;
    if (auto gpuMod = op->getParentOfType<gpu::GPUModuleOp>())
      moduleBody = gpuMod.getBody();
    else if (auto mod = op->getParentOfType<ModuleOp>())
      moduleBody = mod.getBody();
    if (!moduleBody)
      return rewriter.notifyMatchFailure(op, "no enclosing module");

    std::string symName =
        "__smem_alloc_" + std::to_string(smemCounter++);

    auto arrayTy = LLVM::LLVMArrayType::get(elemTy, nElems);
    unsigned smemAS = ptrTy.getAddressSpace();

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleBody);
      LLVM::GlobalOp::create(rewriter, loc, arrayTy, /*isConstant=*/false,
                             LLVM::Linkage::Internal, symName,
                             Attribute{}, /*alignment=*/0, smemAS);
    }

    Value addr = LLVM::AddressOfOp::create(rewriter, loc, ptrTy, symName);
    rewriter.replaceOp(op, addr);
    return success();
  }
};

class GetIterOpLowering : public OpConversionPattern<GetIterOp> {
public:
  using OpConversionPattern<GetIterOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(GetIterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value mem = adaptor.getMemref();
    Type resTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!resTy)
      return failure();
    assert(mem.getType() == resTy);
    rewriter.replaceOp(op, mem);
    return success();
  }
};

class AddOffsetOpLowering : public OpConversionPattern<AddOffsetOp> {
public:
  using OpConversionPattern<AddOffsetOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AddOffsetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value base = adaptor.getPtr();
    auto flyPtrTy = dyn_cast<fly::PointerType>(op.getPtr().getType());
    if (!flyPtrTy)
      return failure();
    auto offsetIdx = materializeScalarIndex(op.getOffset(), loc, rewriter);
    if (failed(offsetIdx))
      return failure();
    if (flyPtrTy.getAddressSpace().getValue() == AddressSpace::BufferDesc)
      return rewriter.notifyMatchFailure(
          op, "BufferDesc not supported in IXDL lowering");
    auto basePtrTy = dyn_cast<LLVM::LLVMPointerType>(base.getType());
    if (!basePtrTy)
      return failure();
    auto resultTy = dyn_cast<LLVM::LLVMPointerType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultTy)
      return failure();
    FailureOr<Value> offsetI64 = toI64(*offsetIdx, loc, rewriter);
    if (failed(offsetI64))
      return failure();
    Type elemTy = flyPtrTy.getElemTy();
    Value gep = LLVM::GEPOp::create(rewriter, loc, resultTy, elemTy, base,
                                     ValueRange{*offsetI64});
    rewriter.replaceOp(op, gep);
    return success();
  }
};

class MakeViewOpLowering : public OpConversionPattern<MakeViewOp> {
public:
  using OpConversionPattern<MakeViewOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(MakeViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value base = adaptor.getIter();
    Type resultTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultTy)
      return failure();
    if (base.getType() == resultTy) {
      rewriter.replaceOp(op, base);
      return success();
    }
    if (isa<LLVM::LLVMPointerType>(base.getType()) &&
        isa<LLVM::LLVMPointerType>(resultTy)) {
      rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, resultTy, base);
      return success();
    }
    return failure();
  }
};

class MemRefLoadVecOpLowering
    : public OpConversionPattern<MemRefLoadVecOp> {
public:
  using OpConversionPattern<MemRefLoadVecOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(MemRefLoadVecOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getMemref();
    if (!isa<LLVM::LLVMPointerType>(input.getType()))
      return failure();
    auto resVecTy = dyn_cast<VectorType>(op.getResult().getType());
    if (!resVecTy)
      return failure();
    Value loaded =
        LLVM::LoadOp::create(rewriter, op.getLoc(), resVecTy, input);
    rewriter.replaceOp(op, loaded);
    return success();
  }
};

class MemRefStoreVecOpLowering
    : public OpConversionPattern<MemRefStoreVecOp> {
public:
  using OpConversionPattern<MemRefStoreVecOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(MemRefStoreVecOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value dest = adaptor.getMemref();
    Value valueToStore = adaptor.getVector();
    if (!isa<LLVM::LLVMPointerType>(dest.getType()))
      return failure();
    if (!isa<VectorType>(valueToStore.getType()))
      return failure();
    LLVM::StoreOp::create(rewriter, op.getLoc(), valueToStore, dest);
    rewriter.eraseOp(op);
    return success();
  }
};

class MemRefLoadOpLowering : public OpConversionPattern<MemRefLoadOp> {
public:
  using OpConversionPattern<MemRefLoadOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(MemRefLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value mem = adaptor.getMemref();
    if (!isa<LLVM::LLVMPointerType>(mem.getType()))
      return failure();
    auto idxVal =
        materializeScalarIndex(op.getIndices(), op.getLoc(), rewriter);
    if (failed(idxVal))
      return failure();
    FailureOr<Value> idxI64 = toI64(*idxVal, op.getLoc(), rewriter);
    if (failed(idxI64))
      return failure();
    auto flyMemRefTy = dyn_cast<fly::MemRefType>(op.getMemref().getType());
    if (!flyMemRefTy)
      return failure();
    Type elemTy = flyMemRefTy.getElemTy();
    auto ptrTy = cast<LLVM::LLVMPointerType>(mem.getType());
    Value gep = LLVM::GEPOp::create(rewriter, op.getLoc(), ptrTy, elemTy, mem,
                                     ValueRange{*idxI64});
    Value loaded = LLVM::LoadOp::create(rewriter, op.getLoc(),
                                         op.getResult().getType(), gep);
    rewriter.replaceOp(op, loaded);
    return success();
  }
};

class MemRefStoreOpLowering : public OpConversionPattern<MemRefStoreOp> {
public:
  using OpConversionPattern<MemRefStoreOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(MemRefStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value mem = adaptor.getMemref();
    if (!isa<LLVM::LLVMPointerType>(mem.getType()))
      return failure();
    auto idxVal =
        materializeScalarIndex(op.getIndices(), op.getLoc(), rewriter);
    if (failed(idxVal))
      return failure();
    FailureOr<Value> idxI64 = toI64(*idxVal, op.getLoc(), rewriter);
    if (failed(idxI64))
      return failure();
    auto flyMemRefTy = dyn_cast<fly::MemRefType>(op.getMemref().getType());
    if (!flyMemRefTy)
      return failure();
    Type elemTy = flyMemRefTy.getElemTy();
    auto ptrTy = cast<LLVM::LLVMPointerType>(mem.getType());
    Value gep = LLVM::GEPOp::create(rewriter, op.getLoc(), ptrTy, elemTy, mem,
                                     ValueRange{*idxI64});
    LLVM::StoreOp::create(rewriter, op.getLoc(), adaptor.getValue(), gep);
    rewriter.eraseOp(op);
    return success();
  }
};

// ---------- IXDL-specific Copy atom lowering ----------

class CopyAtomCallLowering : public OpConversionPattern<CopyAtomCall> {
public:
  using OpConversionPattern<CopyAtomCall>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CopyAtomCall op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type copyAtomType = op.getCopyAtom().getType();
    auto copyAtom = dyn_cast<CopyAtomType>(copyAtomType);
    if (!copyAtom)
      return rewriter.notifyMatchFailure(op, "copyAtom is not CopyAtomType");

    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();
    Value pred = adaptor.getPred();

    auto srcFlyTy = dyn_cast<fly::MemRefType>(op.getSrc().getType());
    auto dstFlyTy = dyn_cast<fly::MemRefType>(op.getDst().getType());
    if (!srcFlyTy || !dstFlyTy)
      return rewriter.notifyMatchFailure(op, "expected Fly memref types");
    if (srcFlyTy.getElemTy() != dstFlyTy.getElemTy())
      return rewriter.notifyMatchFailure(op, "src/dst element types mismatch");

    Location loc = op.getLoc();
    Type copyOpType = copyAtom.getCopyOp();

    auto emitCopyBody = [&](ConversionPatternRewriter &rw) -> LogicalResult {
      if (isa<CopyOpUniversalCopyType>(copyOpType))
        return lowerUniversalCopy(op, rw, loc, copyAtom, srcFlyTy, src, dst);
      if (isa<fly_ixdl::CopyOpIvcore11_SLBLoadType>(copyOpType) ||
          isa<fly_ixdl::CopyOpIvcore11_DescStoreType>(copyOpType) ||
          isa<fly_ixdl::CopyOpIvcore11_SMELoadType>(copyOpType))
        return lowerIXDLCopy(op, rw, loc, copyAtom, srcFlyTy, src, dst);
      return rw.notifyMatchFailure(op, "unsupported CopyOp type for IXDL");
    };

    if (pred) {
      auto predFlyTy = dyn_cast<fly::MemRefType>(op.getPred().getType());
      if (!predFlyTy)
        return rewriter.notifyMatchFailure(op, "pred not a Fly memref type");
      Type predElemTy = predFlyTy.getElemTy();
      Value predVal = LLVM::LoadOp::create(rewriter, loc, predElemTy, pred);
      auto ifOp =
          scf::IfOp::create(rewriter, loc, TypeRange{}, predVal, false);
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      return emitCopyBody(rewriter);
    }
    return emitCopyBody(rewriter);
  }

private:
  LogicalResult lowerUniversalCopy(CopyAtomCall op,
                                   ConversionPatternRewriter &rewriter,
                                   Location loc, CopyAtomType copyAtomTy,
                                   fly::MemRefType srcFlyTy, Value src,
                                   Value dst) const {
    LayoutBuilder<LayoutAttr> attrBuilder(rewriter.getContext());
    auto thrValLayoutSrc =
        dyn_cast<LayoutAttr>(copyAtomTy.getThrValLayoutSrc());
    if (!thrValLayoutSrc)
      return rewriter.notifyMatchFailure(op, "getThrValLayoutSrc null");
    IntAttr numValSrcAttr =
        intTupleProduct(attrBuilder, thrValLayoutSrc.getShape().at(1))
            .getLeafAsInt();
    if (!numValSrcAttr.isStatic())
      return rewriter.notifyMatchFailure(op, "NumValSrc not static");
    int64_t numValSrc = numValSrcAttr.getValue();
    Type elemTy = srcFlyTy.getElemTy();

    if (numValSrc == 1) {
      Value v = LLVM::LoadOp::create(rewriter, loc, elemTy, src);
      LLVM::StoreOp::create(rewriter, loc, v, dst);
    } else {
      auto vecTy = VectorType::get({numValSrc}, elemTy);
      Value v = LLVM::LoadOp::create(rewriter, loc, vecTy, src);
      LLVM::StoreOp::create(rewriter, loc, v, dst);
    }
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult lowerIXDLCopy(CopyAtomCall op,
                              ConversionPatternRewriter &rewriter, Location loc,
                              CopyAtomType copyAtomTy,
                              fly::MemRefType srcFlyTy, Value src,
                              Value dst) const {
    LayoutBuilder<LayoutAttr> attrBuilder(rewriter.getContext());
    auto thrValLayoutSrc =
        dyn_cast<LayoutAttr>(copyAtomTy.getThrValLayoutSrc());
    if (!thrValLayoutSrc)
      return rewriter.notifyMatchFailure(op, "getThrValLayoutSrc null");
    IntAttr numValSrcAttr =
        intTupleProduct(attrBuilder, thrValLayoutSrc.getShape().at(1))
            .getLeafAsInt();
    if (!numValSrcAttr.isStatic())
      return rewriter.notifyMatchFailure(op, "NumValSrc not static");
    int64_t numValSrc = numValSrcAttr.getValue();
    Type elemTy = srcFlyTy.getElemTy();

    if (numValSrc == 1) {
      Value v = LLVM::LoadOp::create(rewriter, loc, elemTy, src);
      LLVM::StoreOp::create(rewriter, loc, v, dst);
    } else {
      auto vecTy = VectorType::get({numValSrc}, elemTy);
      Value v = LLVM::LoadOp::create(rewriter, loc, vecTy, src);
      LLVM::StoreOp::create(rewriter, loc, v, dst);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

// ---------- IXDL-specific MMA atom lowering ----------

class MmaAtomCallLowering : public OpConversionPattern<MmaAtomCall> {
public:
  using OpConversionPattern<MmaAtomCall>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MmaAtomCall op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type mmaAtomType = op.getMmaAtom().getType();
    if (!isa<MmaAtomTypeInterface>(mmaAtomType))
      return rewriter.notifyMatchFailure(op, "expected MmaAtomTypeInterface");

    Location loc = op.getLoc();
    Value dPtr = adaptor.getD();
    Value aPtr = adaptor.getA();
    Value bPtr = adaptor.getB();
    Value cPtr = adaptor.getC();

    if (!isa<LLVM::LLVMPointerType>(dPtr.getType()) ||
        !isa<LLVM::LLVMPointerType>(aPtr.getType()) ||
        !isa<LLVM::LLVMPointerType>(bPtr.getType()) ||
        !isa<LLVM::LLVMPointerType>(cPtr.getType()))
      return rewriter.notifyMatchFailure(op, "expected llvm.ptr operands");

    if (auto universalFma = dyn_cast<MmaAtomUniversalFMAType>(mmaAtomType))
      return lowerUniversalFMA(op, rewriter, loc, universalFma, dPtr, aPtr,
                               bPtr, cPtr);
    if (auto ivcore11Mmad =
            dyn_cast<fly_ixdl::MmaAtomIvcore11_MMADType>(mmaAtomType))
      return lowerIvcore11MMAD(op, rewriter, loc, ivcore11Mmad, dPtr, aPtr,
                               bPtr, cPtr);

    return rewriter.notifyMatchFailure(op, "unsupported MmaAtom type");
  }

private:
  LogicalResult lowerUniversalFMA(MmaAtomCall op,
                                  ConversionPatternRewriter &rewriter,
                                  Location loc,
                                  MmaAtomUniversalFMAType atomTy, Value dPtr,
                                  Value aPtr, Value bPtr,
                                  Value cPtr) const {
    Type elemTy = atomTy.getElemTy();
    Value a = LLVM::LoadOp::create(rewriter, loc, elemTy, aPtr);
    Value b = LLVM::LoadOp::create(rewriter, loc, elemTy, bPtr);
    Value c = LLVM::LoadOp::create(rewriter, loc, elemTy, cPtr);
    Value mul = LLVM::FMulOp::create(rewriter, loc, elemTy, a, b);
    Value res = LLVM::FAddOp::create(rewriter, loc, elemTy, mul, c);
    LLVM::StoreOp::create(rewriter, loc, res, dPtr);
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult lowerIvcore11MMAD(MmaAtomCall op,
                                  ConversionPatternRewriter &rewriter,
                                  Location loc,
                                  fly_ixdl::MmaAtomIvcore11_MMADType atomTy,
                                  Value dPtr, Value aPtr, Value bPtr,
                                  Value cPtr) const {
    MLIRContext *ctx = rewriter.getContext();
    int32_t m = atomTy.getM();
    int32_t n = atomTy.getN();
    int32_t k = atomTy.getK();
    Type elemTyA = atomTy.getElemTyA();
    Type elemTyB = atomTy.getElemTyB();
    Type elemTyAcc = atomTy.getElemTyAcc();

    Type abVecTyA, abVecTyB, accVecTy;
    IXDL::MMADTypes mmadTypeA, mmadTypeB;

    if (elemTyA.isF16()) {
      abVecTyA = VectorType::get({4}, Float16Type::get(ctx));
      mmadTypeA = IXDL::MMADTypes::f16;
    } else if (elemTyA.isBF16()) {
      abVecTyA = VectorType::get({4}, BFloat16Type::get(ctx));
      mmadTypeA = IXDL::MMADTypes::bf16;
    } else if (elemTyA.isInteger(8)) {
      abVecTyA = VectorType::get({4}, IntegerType::get(ctx, 8));
      mmadTypeA = IXDL::MMADTypes::s8;
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported A element type");
    }

    if (elemTyB.isF16()) {
      abVecTyB = VectorType::get({4}, Float16Type::get(ctx));
      mmadTypeB = IXDL::MMADTypes::f16;
    } else if (elemTyB.isBF16()) {
      abVecTyB = VectorType::get({4}, BFloat16Type::get(ctx));
      mmadTypeB = IXDL::MMADTypes::bf16;
    } else if (elemTyB.isInteger(8)) {
      abVecTyB = VectorType::get({4}, IntegerType::get(ctx, 8));
      mmadTypeB = IXDL::MMADTypes::s8;
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported B element type");
    }

    if (elemTyAcc.isF32())
      accVecTy = VectorType::get({4}, Float32Type::get(ctx));
    else if (elemTyAcc.isInteger(32))
      accVecTy = VectorType::get({4}, IntegerType::get(ctx, 32));
    else
      return rewriter.notifyMatchFailure(op, "unsupported acc element type");

    Value aVec = LLVM::LoadOp::create(rewriter, loc, abVecTyA, aPtr);
    Value bVec = LLVM::LoadOp::create(rewriter, loc, abVecTyB, bPtr);
    Value cVec = LLVM::LoadOp::create(rewriter, loc, accVecTy, cPtr);

    std::array<int64_t, 3> shape = {m, n, k};
    std::array<IXDL::MMADTypes, 2> multiplicandTypes = {mmadTypeA, mmadTypeB};
    std::array<IXDL::MMADLayout, 2> multiplicandLayouts = {
        IXDL::MMADLayout::row, IXDL::MMADLayout::col};

    Value result = IXDL::MmadOp::create(
        rewriter, loc, accVecTy, ValueRange{aVec}, ValueRange{bVec},
        ValueRange{cVec}, shape, multiplicandTypes, multiplicandLayouts);

    LLVM::StoreOp::create(rewriter, loc, result, dPtr);
    rewriter.eraseOp(op);
    return success();
  }
};

class GpuLaunchFuncOpLowering
    : public OpConversionPattern<gpu::LaunchFuncOp> {
public:
  using OpConversionPattern<gpu::LaunchFuncOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gpu::LaunchFuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto kernelRef = adaptor.getKernel();
    auto grid = gpu::KernelDim3{adaptor.getGridSizeX(), adaptor.getGridSizeY(),
                                adaptor.getGridSizeZ()};
    auto block = gpu::KernelDim3{adaptor.getBlockSizeX(),
                                 adaptor.getBlockSizeY(),
                                 adaptor.getBlockSizeZ()};
    std::optional<gpu::KernelDim3> clusterSize = std::nullopt;
    if (adaptor.getClusterSizeX() && adaptor.getClusterSizeY() &&
        adaptor.getClusterSizeZ())
      clusterSize =
          gpu::KernelDim3{adaptor.getClusterSizeX(), adaptor.getClusterSizeY(),
                          adaptor.getClusterSizeZ()};
    Type asyncTokenType = nullptr;
    if (Value tok = op.getAsyncToken())
      asyncTokenType = tok.getType();
    if (Value asyncObj = adaptor.getAsyncObject()) {
      if (!adaptor.getAsyncDependencies().empty())
        return rewriter.notifyMatchFailure(
            op, "both asyncObject and asyncDependencies");
      rewriter.replaceOpWithNewOp<gpu::LaunchFuncOp>(
          op, kernelRef, grid, block, adaptor.getDynamicSharedMemorySize(),
          adaptor.getKernelOperands(), asyncObj, clusterSize);
      return success();
    }
    rewriter.replaceOpWithNewOp<gpu::LaunchFuncOp>(
        op, kernelRef, grid, block, adaptor.getDynamicSharedMemorySize(),
        adaptor.getKernelOperands(), asyncTokenType,
        adaptor.getAsyncDependencies(), clusterSize);
    return success();
  }
};

class ExtractAlignedPointerAsIndexLowering
    : public OpConversionPattern<ExtractAlignedPointerAsIndexOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ExtractAlignedPointerAsIndexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value src = adaptor.getSource();
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      resultType = op.getResult().getType();
    if (src.getType() != resultType)
      src = rewriter.create<LLVM::AddrSpaceCastOp>(op.getLoc(), resultType,
                                                    src);
    rewriter.replaceOp(op, src);
    return success();
  }
};

class FlyTypeConverterIXDL : public TypeConverter {
public:
  FlyTypeConverterIXDL() {
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

class FlyToIXDLConversionPass
    : public mlir::impl::FlyToIXDLConversionPassBase<FlyToIXDLConversionPass> {
public:
  using mlir::impl::FlyToIXDLConversionPassBase<
      FlyToIXDLConversionPass>::FlyToIXDLConversionPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(getContext());

    target.addLegalDialect<arith::ArithDialect, scf::SCFDialect,
                           vector::VectorDialect, gpu::GPUDialect,
                           func::FuncDialect, LLVM::LLVMDialect,
                           IXDL::IXDLDialect, fly_ixdl::FlyIXDLDialect>();
    target.addIllegalDialect<fly::FlyDialect>();

    target.addLegalOp<StaticOp, MakeIntTupleOp, MakeLayoutOp, MakeTileOp,
                      MakeComposedLayoutOp>();
    target.addLegalOp<MakeMmaAtomOp, MakeCopyAtomOp>();

    FlyTypeConverterIXDL typeConverter;

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<gpu::GPUFuncOp>([&](gpu::GPUFuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<gpu::LaunchFuncOp>(
        [&](gpu::LaunchFuncOp op) {
          for (Value v : op.getKernelOperands())
            if (!typeConverter.isLegal(v.getType()))
              return false;
          if (auto v = op.getDynamicSharedMemorySize();
              v && !typeConverter.isLegal(v.getType()))
            return false;
          for (Value dep : op.getAsyncDependencies())
            if (!typeConverter.isLegal(dep.getType()))
              return false;
          if (auto v = op.getAsyncObject();
              v && !typeConverter.isLegal(v.getType()))
            return false;
          return true;
        });

    patterns.add<MakePtrOpLowering>(typeConverter, context);
    patterns.add<MemRefAllocOpLowering>(typeConverter, context);
    patterns.add<GetIterOpLowering>(typeConverter, context);
    patterns.add<AddOffsetOpLowering>(typeConverter, context);
    patterns.add<MakeViewOpLowering>(typeConverter, context);
    patterns.add<MemRefLoadVecOpLowering>(typeConverter, context);
    patterns.add<MemRefStoreVecOpLowering>(typeConverter, context);
    patterns.add<MemRefLoadOpLowering>(typeConverter, context);
    patterns.add<MemRefStoreOpLowering>(typeConverter, context);
    patterns.add<CopyAtomCallLowering>(typeConverter, context);
    patterns.add<MmaAtomCallLowering>(typeConverter, context);
    patterns.add<GpuLaunchFuncOpLowering>(typeConverter, context);
    patterns.add<ExtractAlignedPointerAsIndexLowering>(typeConverter, context);

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateFunctionOpInterfaceTypeConversionPattern<gpu::GPUFuncOp>(
        patterns, typeConverter);

    if (failed(
            applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace impl {
std::unique_ptr<::mlir::Pass> createFlyToIXDLConversionPass() {
  return std::make_unique<FlyToIXDLConversionPass>();
}
} // namespace impl
