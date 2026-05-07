// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/IXDLDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

#include "flydsl/Conversion/FlyToROCDL/FlyToROCDL.h"
#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/IntTupleUtils.h"
#include "flydsl/Dialect/Fly/Utils/LayoutUtils.h"
#include "flydsl/Dialect/Fly/Utils/PointerUtils.h"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#include "flydsl/Dialect/FlyROCDL/Utils/BufferFatPtr.h"

namespace mlir {
#define GEN_PASS_DEF_FLYTOROCDLCONVERSIONPASS
#define GEN_PASS_DEF_FLYROCDLCLUSTERATTRPASS
#include "flydsl/Conversion/FlyToROCDL/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::fly;

namespace {

unsigned mapToLLVMAddressSpace(AddressSpace addrSpace) {
  switch (addrSpace) {
  case AddressSpace::Global:
    return 1;
  case AddressSpace::Shared:
    return 3;
  case AddressSpace::Register:
    return 5;
  case AddressSpace::BufferDesc:
    return 8;
  default:
    assert(false && "Unsupported address space");
    return 0;
  }
}

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
        return rewriter.notifyMatchFailure(op, "register make_ptr requires allocSize in ptrAttrs");
      unsigned llvmAS = mapToLLVMAddressSpace(AddressSpace::Register);
      auto llvmPtrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), llvmAS);
      Value nElems = arith::ConstantIntOp::create(rewriter, loc, allocSize.getInt(), 64);
      Type elemTy = projectToLLVMCompatibleElemTy(flyPtrTy.getElemTy());
      Value ptr = LLVM::AllocaOp::create(rewriter, loc, llvmPtrTy, elemTy, nElems, 0);
      rewriter.replaceOp(op, ptr);
      return success();
    } else if (addrSpace == AddressSpace::BufferDesc) {
      auto args = adaptor.getArgs();
      if (args.size() != 4)
        return rewriter.notifyMatchFailure(
            op, "buffer_rsrc make_ptr expects 4 args: base, stride, numRecords, flags");

      Value base = args[0];
      Value stride = args[1];
      Value numRecords = args[2];
      Value flags = args[3];

      auto rsrcPtrTy = LLVM::LLVMPointerType::get(rewriter.getContext(),
                                                  mapToLLVMAddressSpace(AddressSpace::BufferDesc));
      Value bufferRsrc = ROCDL::MakeBufferRsrcOp::create(rewriter, loc, rsrcPtrTy, base, stride,
                                                         numRecords, flags);
      rewriter.replaceOp(op, BufferFatPtr::pack(rewriter, loc, bufferRsrc));
      return success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported make_ptr address space");
  }
};

class GetDynSharedOpLowering : public OpConversionPattern<GetDynSharedOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(GetDynSharedOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto flyPtrTy = cast<fly::PointerType>(op.getResult().getType());
    unsigned addrSpace = mapToLLVMAddressSpace(flyPtrTy.getAddressSpace().getValue());

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

class AddOffsetOpLowering : public OpConversionPattern<AddOffsetOp> {
public:
  AddOffsetOpLowering(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<AddOffsetOp>(typeConverter, context) {}

  LogicalResult matchAndRewrite(AddOffsetOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value base = adaptor.getPtr();
    Value offset = adaptor.getOffset();

    auto flyPtrTy = dyn_cast<fly::PointerType>(op.getPtr().getType());
    if (!flyPtrTy)
      return failure();

    auto offsetTy = dyn_cast<fly::IntTupleType>(offset.getType());
    IntTupleAttr offsetAttr = offsetTy.getAttr();
    if (!offsetAttr.isLeaf())
      return rewriter.notifyMatchFailure(op, "offset must be a leaf int tuple");

    Value offsetVal;
    auto offsetInt = offsetAttr.extractIntFromLeaf();
    if (offsetInt.isStatic()) {
      offsetVal = arith::ConstantIntOp::create(rewriter, loc, offsetInt.getValue(), 32);
    } else {
      Operation *defOp = offset.getDefiningOp();
      offsetVal = defOp->getOperand(0);
    }

    if (flyPtrTy.getAddressSpace().getValue() == AddressSpace::BufferDesc) {
      BufferFatPtr bp(flyPtrTy, base);
      rewriter.replaceOp(op, bp.addOffset(rewriter, loc, offsetVal));
      return success();
    }

    auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(base.getType());
    if (!ptrTy)
      return failure();

    Type elemTy = projectToLLVMCompatibleElemTy(flyPtrTy.getElemTy());
    Value gep = LLVM::GEPOp::create(rewriter, loc, ptrTy, elemTy, base, ValueRange{offsetVal});
    rewriter.replaceOp(op, gep);
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
    } else {
      Value base = adaptor.getIter();
      rewriter.replaceOp(op, base);
      return success();
    }
  }
};

class PtrLoadOpLowering : public OpConversionPattern<PtrLoadOp> {
public:
  using OpConversionPattern<PtrLoadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(PtrLoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value ptr = adaptor.getPtr();

    auto flyPtrTy = dyn_cast<fly::PointerType>(op.getPtr().getType());
    if (!flyPtrTy)
      return failure();

    Type loadTy = op.getResult().getType();

    if (auto vecTy = dyn_cast<VectorType>(loadTy)) {
      auto swizzle = flyPtrTy.getSwizzle();
      if (!swizzle.isTrivialSwizzle()) {
        int64_t vecBytes =
            vecTy.getNumElements() * vecTy.getElementType().getIntOrFloatBitWidth() / 8;
        int64_t baseBytes = int64_t{1} << swizzle.getBase();
        if (baseBytes % vecBytes != 0)
          return rewriter.notifyMatchFailure(
              op, "vector ptr.load byte size must divide swizzle base granularity");
      }
    }

    if (flyPtrTy.getAddressSpace().getValue() == AddressSpace::BufferDesc) {
      BufferFatPtr bp(flyPtrTy, ptr);
      Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
      ArrayAttr noAttrs;
      Value loaded = ROCDL::RawPtrBufferLoadOp::create(
          rewriter, loc, loadTy, bp.bufferRsrc(rewriter, loc), bp.swizzleByteOffset(rewriter, loc),
          zero, zero, noAttrs, noAttrs, noAttrs);
      rewriter.replaceOp(op, loaded);
      return success();
    } else {
      ptr = applySwizzleOnPtr(rewriter, loc, cast<TypedValue<LLVM::LLVMPointerType>>(ptr),
                              flyPtrTy.getSwizzle());
      Value loaded = LLVM::LoadOp::create(rewriter, loc, loadTy, ptr);
      rewriter.replaceOp(op, loaded);
      return success();
    }
  }
};

class PtrStoreOpLowering : public OpConversionPattern<PtrStoreOp> {
public:
  using OpConversionPattern<PtrStoreOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(PtrStoreOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value ptr = adaptor.getPtr();
    Value value = adaptor.getValue();

    auto flyPtrTy = dyn_cast<fly::PointerType>(op.getPtr().getType());
    if (!flyPtrTy)
      return failure();

    if (auto vecTy = dyn_cast<VectorType>(value.getType())) {
      auto swizzle = flyPtrTy.getSwizzle();
      if (!swizzle.isTrivialSwizzle()) {
        int64_t vecBytes =
            vecTy.getNumElements() * vecTy.getElementType().getIntOrFloatBitWidth() / 8;
        int64_t baseBytes = int64_t{1} << swizzle.getBase();
        if (baseBytes % vecBytes != 0)
          return rewriter.notifyMatchFailure(
              op, "vector ptr.store byte size must divide swizzle base granularity");
      }
    }

    if (flyPtrTy.getAddressSpace().getValue() == AddressSpace::BufferDesc) {
      BufferFatPtr bp(flyPtrTy, ptr);
      Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
      ArrayAttr noAttrs;
      ROCDL::RawPtrBufferStoreOp::create(rewriter, loc, value, bp.bufferRsrc(rewriter, loc),
                                         bp.swizzleByteOffset(rewriter, loc), zero, zero, noAttrs,
                                         noAttrs, noAttrs);
      rewriter.eraseOp(op);
      return success();
    } else {
      ptr = applySwizzleOnPtr(rewriter, loc, cast<TypedValue<LLVM::LLVMPointerType>>(ptr),
                              flyPtrTy.getSwizzle());
      LLVM::StoreOp::create(rewriter, loc, value, ptr);
      rewriter.eraseOp(op);
      return success();
    }
  }
};

class MakeCopyAtomOpLowering : public OpConversionPattern<MakeCopyAtomOp> {
public:
  using OpConversionPattern<MakeCopyAtomOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(MakeCopyAtomOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto copyAtomTy = dyn_cast<CopyAtomType>(op.getResult().getType());
    if (!copyAtomTy)
      return rewriter.notifyMatchFailure(op, "not a CopyAtomType");
    Type convertedTy = getTypeConverter()->convertType(copyAtomTy);

    auto statefulOp = dyn_cast<StatefulOpTypeInterface>(copyAtomTy.getCopyOp());
    if (statefulOp) {
      Value state = statefulOp.getDefaultState(rewriter, op.getLoc());
      rewriter.replaceOp(op, state);
    } else {
      rewriter.replaceOpWithNewOp<LLVM::UndefOp>(op, convertedTy);
    }
    return success();
  }
};

class MakeMmaAtomOpLowering : public OpConversionPattern<MakeMmaAtomOp> {
public:
  using OpConversionPattern<MakeMmaAtomOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(MakeMmaAtomOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto mmaAtomTy = dyn_cast<MmaAtomType>(op.getResult().getType());
    if (!mmaAtomTy)
      return rewriter.notifyMatchFailure(op, "not a MmaAtomType");
    Type convertedTy = getTypeConverter()->convertType(mmaAtomTy);
    auto statefulOp = dyn_cast<StatefulOpTypeInterface>(mmaAtomTy.getMmaOp());
    if (statefulOp) {
      Value state = statefulOp.getDefaultState(rewriter, op.getLoc());
      rewriter.replaceOp(op, state);
    } else {
      rewriter.replaceOpWithNewOp<LLVM::UndefOp>(op, convertedTy);
    }
    return success();
  }
};

class MakeTiledCopyOpLowering : public OpConversionPattern<MakeTiledCopyOp> {
public:
  using OpConversionPattern<MakeTiledCopyOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(MakeTiledCopyOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getCopyAtom());
    return success();
  }
};

class MakeTiledMmaOpLowering : public OpConversionPattern<MakeTiledMmaOp> {
public:
  using OpConversionPattern<MakeTiledMmaOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(MakeTiledMmaOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getMmaAtom());
    return success();
  }
};

class AtomSetValueOpLowering : public OpConversionPattern<AtomSetValueOp> {
public:
  using OpConversionPattern<AtomSetValueOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(AtomSetValueOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type origAtomTy = op.getAtom().getType();
    StringAttr fieldAttr = op.getFieldAttr();
    Location loc = op.getLoc();

    Value structVal = adaptor.getAtom();
    Value fieldVal = adaptor.getValue();
    Value result;

    if (auto copyAtomTy = dyn_cast<CopyAtomType>(origAtomTy)) {
      if (!copyAtomTy.isStateful())
        return rewriter.notifyMatchFailure(op, "CopyAtom is not stateful");
      result = copyAtomTy.setAtomState(rewriter, loc, structVal, fieldAttr, fieldVal);
    } else if (auto mmaAtomTy = dyn_cast<MmaAtomType>(origAtomTy)) {
      if (!mmaAtomTy.isStateful())
        return rewriter.notifyMatchFailure(op, "MmaAtom is not stateful");
      result = mmaAtomTy.setAtomState(rewriter, loc, structVal, fieldAttr, fieldVal);
    } else {
      return rewriter.notifyMatchFailure(op, "atom is not CopyAtomType or MmaAtomType");
    }

    if (!result)
      return rewriter.notifyMatchFailure(op, "setAtomState failed");

    rewriter.replaceOp(op, result);
    return success();
  }
};

class CopyAtomCallLowering : public OpConversionPattern<CopyAtomCall> {
public:
  using OpConversionPattern<CopyAtomCall>::OpConversionPattern;

  LogicalResult matchAndRewrite(CopyAtomCall op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type copyAtomType = op.getCopyAtom().getType();
    auto copyAtom = dyn_cast<CopyAtomType>(copyAtomType);
    if (!copyAtom)
      return rewriter.notifyMatchFailure(op, "copyAtom is not CopyAtomType");

    Value copyAtomVal = adaptor.getCopyAtom();
    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();
    Value pred = adaptor.getPred();

    auto srcMemTy = dyn_cast<fly::MemRefType>(op.getSrc().getType());
    auto dstMemTy = dyn_cast<fly::MemRefType>(op.getDst().getType());

    if (!srcMemTy || !dstMemTy)
      return rewriter.notifyMatchFailure(op, "expected MemRef types on original op");
    if (srcMemTy.getElemTy() != dstMemTy.getElemTy())
      return rewriter.notifyMatchFailure(op, "src/dst element types mismatch");

    Location loc = op.getLoc();

    Type predMemTy = nullptr;
    if (pred) {
      predMemTy = dyn_cast<fly::MemRefType>(op.getPred().getType());
      if (!predMemTy)
        return rewriter.notifyMatchFailure(op, "pred is not a MemRef type");
    }

    if (pred) {
      if (failed(copyAtom.emitAtomCall(rewriter, loc, copyAtomType, srcMemTy, dstMemTy, predMemTy,
                                       copyAtomVal, src, dst, pred)))
        return failure();
    } else {
      if (failed(copyAtom.emitAtomCall(rewriter, loc, copyAtomType, srcMemTy, dstMemTy, copyAtomVal,
                                       src, dst)))
        return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

class CopyAtomCallSSALowering : public OpConversionPattern<CopyAtomCallSSA> {
public:
  using OpConversionPattern<CopyAtomCallSSA>::OpConversionPattern;

  LogicalResult matchAndRewrite(CopyAtomCallSSA op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type copyAtomType = op.getCopyAtom().getType();
    auto copyAtom = dyn_cast<CopyAtomType>(copyAtomType);
    if (!copyAtom)
      return rewriter.notifyMatchFailure(op, "copyAtom is not CopyAtomType");

    Location loc = op.getLoc();
    bool hasResult = op.getResults().size() > 0;
    Type srcTy = op.getSrc().getType();
    Value pred = adaptor.getPred();

    Type resultTy = hasResult ? op.getResult(0).getType() : Type{};
    Type dstTy = op.getDst() ? op.getDst().getType() : Type{};

    FailureOr<Value> result;
    if (pred) {
      result = copyAtom.emitAtomCallSSA(rewriter, loc, resultTy, copyAtomType, srcTy, dstTy,
                                        op.getPred().getType(), adaptor.getCopyAtom(),
                                        adaptor.getSrc(), adaptor.getDst(), pred);
    } else {
      result = copyAtom.emitAtomCallSSA(rewriter, loc, resultTy, copyAtomType, srcTy, dstTy,
                                        adaptor.getCopyAtom(), adaptor.getSrc(), adaptor.getDst());
    }
    if (failed(result))
      return failure();

    if (hasResult)
      rewriter.replaceOp(op, *result);
    else
      rewriter.eraseOp(op);
    return success();
  }
};

class MmaAtomCallLowering : public OpConversionPattern<MmaAtomCall> {
public:
  using OpConversionPattern<MmaAtomCall>::OpConversionPattern;

  LogicalResult matchAndRewrite(MmaAtomCall op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto mmaAtomTy = dyn_cast<MmaAtomType>(op.getMmaAtom().getType());
    if (!mmaAtomTy)
      return rewriter.notifyMatchFailure(op, "expected MmaAtomType for mmaAtom operand");

    Location loc = op.getLoc();

    Value dPtr = adaptor.getD();
    Value aPtr = adaptor.getA();
    Value bPtr = adaptor.getB();
    Value cPtr = adaptor.getC();

    if (!isa<LLVM::LLVMPointerType>(dPtr.getType()) ||
        !isa<LLVM::LLVMPointerType>(aPtr.getType()) ||
        !isa<LLVM::LLVMPointerType>(bPtr.getType()) || !isa<LLVM::LLVMPointerType>(cPtr.getType()))
      return rewriter.notifyMatchFailure(op, "expected llvm.ptr operands after type conversion");

    auto dMemTy = dyn_cast<fly::MemRefType>(op.getD().getType());
    auto aMemTy = dyn_cast<fly::MemRefType>(op.getA().getType());
    auto bMemTy = dyn_cast<fly::MemRefType>(op.getB().getType());
    auto cMemTy = dyn_cast<fly::MemRefType>(op.getC().getType());
    if (!dMemTy || !aMemTy || !bMemTy || !cMemTy)
      return rewriter.notifyMatchFailure(op, "expected Fly memref types on original op");

    if (failed(mmaAtomTy.emitAtomCall(rewriter, loc, mmaAtomTy, dMemTy, aMemTy, bMemTy, cMemTy,
                                      adaptor.getMmaAtom(), dPtr, aPtr, bPtr, cPtr)))
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

class MmaAtomCallSSALowering : public OpConversionPattern<MmaAtomCallSSA> {
public:
  using OpConversionPattern<MmaAtomCallSSA>::OpConversionPattern;

  LogicalResult matchAndRewrite(MmaAtomCallSSA op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto mmaAtomTy = dyn_cast<MmaAtomType>(op.getMmaAtom().getType());
    if (!mmaAtomTy)
      return rewriter.notifyMatchFailure(op, "expected MmaAtomType for mmaAtom operand");

    Location loc = op.getLoc();
    bool hasResult = op.getResults().size() > 0;

    Type resultTy = hasResult ? op.getResult(0).getType() : Type{};
    Type dTy = op.getD() ? op.getD().getType() : Type{};
    Value dPtr = hasResult ? Value{} : adaptor.getD();

    auto result =
        mmaAtomTy.emitAtomCallSSA(rewriter, loc, resultTy, mmaAtomTy, dTy, op.getA().getType(),
                                  op.getB().getType(), op.getC().getType(), adaptor.getMmaAtom(),
                                  dPtr, adaptor.getA(), adaptor.getB(), adaptor.getC());
    if (failed(result))
      return failure();

    if (hasResult)
      rewriter.replaceOp(op, *result);
    else
      rewriter.eraseOp(op);
    return success();
  }
};

class CpAsyncCommitGroupLowering : public OpConversionPattern<CpAsyncCommitGroupOp> {
public:
  using OpConversionPattern<CpAsyncCommitGroupOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(CpAsyncCommitGroupOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IXDL::CpAsyncCommitGroupOp>(op);
    return success();
  }
};

class CpAsyncWaitGroupLowering : public OpConversionPattern<CpAsyncWaitGroupOp> {
public:
  using OpConversionPattern<CpAsyncWaitGroupOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(CpAsyncWaitGroupOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IXDL::CpAsyncWaitGroupOp>(op, adaptor.getNAttr());
    return success();
  }
};

class PipebarReqLowering : public OpConversionPattern<PipebarReqOp> {
public:
  using OpConversionPattern<PipebarReqOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(PipebarReqOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IXDL::PipebarReqOp>(op, adaptor.getIdAttr());
    return success();
  }
};

class PipebarWaitLowering : public OpConversionPattern<PipebarWaitOp> {
public:
  using OpConversionPattern<PipebarWaitOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(PipebarWaitOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IXDL::PipebarWaitOp>(op, adaptor.getIdAttr());
    return success();
  }
};

class SlWaitcntLowering : public OpConversionPattern<SlWaitcntOp> {
public:
  using OpConversionPattern<SlWaitcntOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(SlWaitcntOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IXDL::SlWaitcntOp>(op, adaptor.getCntAttr());
    return success();
  }
};

/// Lower `gpu.launch_func` kernel operands so that any `!fly.memref` values are
/// replaced by their type-converted builtin `memref` values. This prevents
/// `unrealized_conversion_cast` materializations from remaining live after
/// partial conversion (e.g., when the surrounding `func.func` signature has
/// been converted to builtin memrefs).
class GpuLaunchFuncOpLowering : public OpConversionPattern<gpu::LaunchFuncOp> {
public:
  using OpConversionPattern<gpu::LaunchFuncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(gpu::LaunchFuncOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto kernelRef = adaptor.getKernel();

    auto grid =
        gpu::KernelDim3{adaptor.getGridSizeX(), adaptor.getGridSizeY(), adaptor.getGridSizeZ()};
    auto block =
        gpu::KernelDim3{adaptor.getBlockSizeX(), adaptor.getBlockSizeY(), adaptor.getBlockSizeZ()};

    std::optional<gpu::KernelDim3> clusterSize = std::nullopt;
    if (adaptor.getClusterSizeX() && adaptor.getClusterSizeY() && adaptor.getClusterSizeZ()) {
      clusterSize = gpu::KernelDim3{adaptor.getClusterSizeX(), adaptor.getClusterSizeY(),
                                    adaptor.getClusterSizeZ()};
    }

    // Preserve async token result type when present.
    Type asyncTokenType = nullptr;
    if (Value tok = op.getAsyncToken())
      asyncTokenType = tok.getType();

    // There are two relevant builder signatures in this MLIR:
    // - (kernel, ..., asyncTokenType, asyncDependencies, clusterSize)
    // - (kernel, ..., asyncObject, clusterSize)
    // Pick the one that matches the original op structure.
    if (Value asyncObj = adaptor.getAsyncObject()) {
      if (!adaptor.getAsyncDependencies().empty())
        return rewriter.notifyMatchFailure(
            op, "launch_func has both asyncObject and asyncDependencies");

      rewriter.replaceOpWithNewOp<gpu::LaunchFuncOp>(
          op, kernelRef, grid, block, adaptor.getDynamicSharedMemorySize(),
          adaptor.getKernelOperands(), asyncObj, clusterSize);
      return success();
    }

    rewriter.replaceOpWithNewOp<gpu::LaunchFuncOp>(
        op, kernelRef, grid, block, adaptor.getDynamicSharedMemorySize(),
        adaptor.getKernelOperands(), asyncTokenType, adaptor.getAsyncDependencies(), clusterSize);
    return success();
  }
};

class FlyTypeConverter : public TypeConverter {
public:
  FlyTypeConverter() {
    addConversion([](Type type) { return type; });

    addConversion([&](FloatType floatTy) -> std::optional<Type> {
      if (floatTy.getWidth() < 16)
        return IntegerType::get(floatTy.getContext(), floatTy.getWidth());
      return std::nullopt;
    });
    addConversion([&](VectorType vecTy) -> std::optional<Type> {
      Type convertedElem = convertType(vecTy.getElementType());
      if (!convertedElem || convertedElem == vecTy.getElementType())
        return std::nullopt;
      return VectorType::get(vecTy.getShape(), convertedElem, vecTy.getScalableDims());
    });
    addConversion([&](fly::MemRefType flyMemRefTy) -> Type {
      if (flyMemRefTy.getAddressSpace().getValue() == AddressSpace::BufferDesc)
        return BufferFatPtr::getType(flyMemRefTy.getContext());
      unsigned as = mapToLLVMAddressSpace(flyMemRefTy.getAddressSpace().getValue());
      return LLVM::LLVMPointerType::get(flyMemRefTy.getContext(), as);
    });
    addConversion([&](fly::PointerType flyPtrTy) -> Type {
      if (flyPtrTy.getAddressSpace().getValue() == AddressSpace::BufferDesc)
        return BufferFatPtr::getType(flyPtrTy.getContext());
      unsigned as = mapToLLVMAddressSpace(flyPtrTy.getAddressSpace().getValue());
      return LLVM::LLVMPointerType::get(flyPtrTy.getContext(), as);
    });
    addConversion([&](fly::CopyAtomType atomTy) -> Type {
      if (atomTy.isStateful())
        return atomTy.getConvertedType(atomTy.getContext());
      return LLVM::LLVMStructType::getLiteral(atomTy.getContext(), {});
    });
    addConversion([&](fly::MmaAtomType atomTy) -> Type {
      if (atomTy.isStateful())
        return atomTy.getConvertedType(atomTy.getContext());
      return LLVM::LLVMStructType::getLiteral(atomTy.getContext(), {});
    });
    addConversion(
        [&](fly::TiledCopyType tiledTy) -> Type { return convertType(tiledTy.getCopyAtom()); });
    addConversion(
        [&](fly::TiledMmaType tiledTy) -> Type { return convertType(tiledTy.getMmaAtom()); });
  }
};

class ExtractAlignedPointerAsIndexLowering
    : public OpConversionPattern<ExtractAlignedPointerAsIndexOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ExtractAlignedPointerAsIndexOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // fly.memref is a bare pointer; after type conversion the operand is llvm.ptr<AS>.
    // Cast to the result type (e.g. llvm.ptr<0>) if address spaces differ.
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

static std::optional<int64_t> getI32Constant(Value value) {
  if (auto cst = value.getDefiningOp<arith::ConstantIntOp>())
    return cst.value();
  return std::nullopt;
}

static bool matchI32Constant(Value value, int64_t expected) {
  std::optional<int64_t> cst = getI32Constant(value);
  return cst && *cst == expected;
}

static bool matchMulBy(Value value, int64_t factor, Value &operand) {
  auto mul = value.getDefiningOp<arith::MulIOp>();
  if (!mul)
    return false;
  if (matchI32Constant(mul.getRhs(), factor)) {
    operand = mul.getLhs();
    return true;
  }
  if (matchI32Constant(mul.getLhs(), factor)) {
    operand = mul.getRhs();
    return true;
  }
  return false;
}

static bool matchRemBy(Value value, int64_t divisor, Value &operand) {
  auto rem = value.getDefiningOp<arith::RemSIOp>();
  if (!rem || !matchI32Constant(rem.getRhs(), divisor))
    return false;
  operand = rem.getLhs();
  return true;
}

static bool matchDivBy(Value value, int64_t divisor, Value &operand) {
  auto div = value.getDefiningOp<arith::DivSIOp>();
  if (!div || !matchI32Constant(div.getRhs(), divisor))
    return false;
  operand = div.getLhs();
  return true;
}

static bool matchLane(Value value) {
  Value threadId;
  return matchRemBy(value, 64, threadId);
}

// Match the IX11 MMAD A-operand lane layout:
//   (lane % 16) * 2 + (lane / 16) * 32
// which is exactly lane * 2 for lane in [0, 64).
static bool matchAOperandLaneTimesTwo(Value value) {
  auto add = value.getDefiningOp<arith::AddIOp>();
  if (!add)
    return false;

  auto match = [](Value lhs, Value rhs) {
    Value rem16, div16;
    if (!matchMulBy(lhs, 2, rem16) || !matchMulBy(rhs, 32, div16))
      return false;
    Value laneFromRem, laneFromDiv;
    if (!matchRemBy(rem16, 16, laneFromRem) || !matchDivBy(div16, 16, laneFromDiv))
      return false;
    return laneFromRem == laneFromDiv && matchLane(laneFromRem);
  };

  return match(add.getLhs(), add.getRhs()) || match(add.getRhs(), add.getLhs());
}

static bool matchAddConst(Value value, Value &base, int64_t &constant) {
  constant = 0;
  for (int depth = 0; depth < 4; ++depth) {
    auto add = value.getDefiningOp<arith::AddIOp>();
    if (!add)
      break;
    if (auto rhs = getI32Constant(add.getRhs())) {
      constant += *rhs;
      value = add.getLhs();
      continue;
    }
    if (auto lhs = getI32Constant(add.getLhs())) {
      constant += *lhs;
      value = add.getRhs();
      continue;
    }
    break;
  }
  base = value;
  return true;
}

static bool matchNoopAOperandSwizzle(Value value, Value &laneTimesTwo, int64_t &constant) {
  auto xorOp = value.getDefiningOp<arith::XOrIOp>();
  if (!xorOp) {
    Value base;
    int64_t cst = 0;
    if (!matchAddConst(value, base, cst) || !matchAOperandLaneTimesTwo(base))
      return false;
    laneTimesTwo = base;
    constant = cst;
    return true;
  }

  auto shr = xorOp.getRhs().getDefiningOp<arith::ShRUIOp>();
  if (!shr)
    return false;
  auto andOp = shr.getLhs().getDefiningOp<arith::AndIOp>();
  if (!andOp)
    return false;
  if (andOp.getLhs() != xorOp.getLhs())
    return false;
  if (!matchI32Constant(andOp.getRhs(), 256) || !matchI32Constant(shr.getRhs(), 2))
    return false;

  Value base;
  int64_t cst = 0;
  if (!matchAddConst(xorOp.getLhs(), base, cst) || !matchAOperandLaneTimesTwo(base))
    return false;
  // For lane*2 + {0, 128}, S<1,6,2> has no effect because bit 8 is clear.
  if (cst != 0 && cst != 128)
    return false;
  laneTimesTwo = base;
  constant = cst;
  return true;
}

static bool matchAOperandS2ROffset(Value offset, Value &baseElems, int64_t &constantElems) {
  auto add = offset.getDefiningOp<arith::AddIOp>();
  if (!add)
    return false;

  Value laneTimesTwo;
  if (matchNoopAOperandSwizzle(add.getRhs(), laneTimesTwo, constantElems)) {
    baseElems = add.getLhs();
    return true;
  }
  if (matchNoopAOperandSwizzle(add.getLhs(), laneTimesTwo, constantElems)) {
    baseElems = add.getRhs();
    return true;
  }
  return false;
}

static bool matchBOperandInner(Value value, Value &lane) {
  auto add = value.getDefiningOp<arith::AddIOp>();
  if (!add)
    return false;

  auto match = [](Value lhs, Value rhs, Value &lane) {
    Value rem4, div4;
    if (!matchMulBy(lhs, 2, rem4) || !matchMulBy(rhs, 128, div4))
      return false;
    Value laneMod16FromRem, laneMod16FromDiv;
    if (!matchRemBy(rem4, 4, laneMod16FromRem) || !matchDivBy(div4, 4, laneMod16FromDiv))
      return false;
    if (laneMod16FromRem != laneMod16FromDiv)
      return false;
    Value candidateLane;
    if (!matchRemBy(laneMod16FromRem, 16, candidateLane) || !matchLane(candidateLane))
      return false;
    lane = candidateLane;
    return true;
  };

  return match(add.getLhs(), add.getRhs(), lane) || match(add.getRhs(), add.getLhs(), lane);
}

// Match the IX11 MMAD B-operand col_xfb8 lane layout before swizzle:
//   ((lane % 16) % 4) * 2 + ((lane % 16) / 4) * 128 + (lane / 16) * 32
static bool matchBOperandUnswizzled(Value value, Value &lane) {
  auto add = value.getDefiningOp<arith::AddIOp>();
  if (!add)
    return false;

  auto match = [](Value lhs, Value rhs, Value &lane) {
    Value innerLane;
    if (!matchBOperandInner(lhs, innerLane))
      return false;

    Value div16;
    if (!matchMulBy(rhs, 32, div16))
      return false;
    Value laneFromDiv;
    if (!matchDivBy(div16, 16, laneFromDiv) || laneFromDiv != innerLane)
      return false;

    lane = innerLane;
    return true;
  };

  return match(add.getLhs(), add.getRhs(), lane) || match(add.getRhs(), add.getLhs(), lane);
}

static bool matchBOperandSwizzle(Value value, Value &lane, int64_t &constantElems) {
  auto xorOp = value.getDefiningOp<arith::XOrIOp>();
  if (!xorOp)
    return false;

  auto shr = xorOp.getRhs().getDefiningOp<arith::ShRUIOp>();
  if (!shr)
    return false;
  auto andOp = shr.getLhs().getDefiningOp<arith::AndIOp>();
  if (!andOp || andOp.getLhs() != xorOp.getLhs())
    return false;
  if (!matchI32Constant(andOp.getRhs(), 384) || !matchI32Constant(shr.getRhs(), 4))
    return false;

  Value base;
  int64_t cst = 0;
  if (!matchAddConst(xorOp.getLhs(), base, cst) || !matchBOperandUnswizzled(base, lane))
    return false;
  if (cst != 0 && cst != 8 && cst != 16 && cst != 24)
    return false;

  constantElems = cst;
  return true;
}

static bool matchBOperandS2ROffset(Value offset, Value &baseElems, int64_t &constantElems) {
  auto add = offset.getDefiningOp<arith::AddIOp>();
  if (!add)
    return false;

  Value lane;
  if (matchBOperandSwizzle(add.getRhs(), lane, constantElems)) {
    baseElems = add.getLhs();
    return true;
  }
  if (matchBOperandSwizzle(add.getLhs(), lane, constantElems)) {
    baseElems = add.getRhs();
    return true;
  }
  return false;
}

class IX11AOperandBlkloadFriendlyLoad final : public OpRewritePattern<LLVM::LoadOp> {
public:
  using OpRewritePattern<LLVM::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::LoadOp op, PatternRewriter &rewriter) const override {
    auto vecTy = dyn_cast<VectorType>(op.getResult().getType());
    if (!vecTy || vecTy.getNumElements() != 2 ||
        vecTy.getElementType().getIntOrFloatBitWidth() != 16)
      return failure();

    auto gep = op.getAddr().getDefiningOp<LLVM::GEPOp>();
    if (!gep)
      return failure();
    auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(gep.getType());
    if (!ptrTy || ptrTy.getAddressSpace() != 3)
      return failure();

    SmallVector<Value> dynamicIndices(gep.getDynamicIndices());
    if (dynamicIndices.size() != 1)
      return failure();

    Value baseElems;
    int64_t constantElems = 0;
    if (!matchAOperandS2ROffset(dynamicIndices.front(), baseElems, constantElems))
      return failure();
    if ((constantElems & 1) != 0)
      return failure();

    Location loc = op.getLoc();
    Type i32Ty = rewriter.getI32Type();
    Value two = arith::ConstantIntOp::create(rewriter, loc, 2, 32);
    Value baseDwords = arith::DivSIOp::create(rewriter, loc, baseElems, two);
    Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    Value uniformBaseDwords =
        IXDL::ReadlaneOp::create(rewriter, loc, i32Ty, baseDwords, zero);
    Value laneId = IXDL::LaneIdOp::create(rewriter, loc, i32Ty);
    Value dwordOffset = arith::AddIOp::create(rewriter, loc, uniformBaseDwords, laneId);
    Value i32Ptr =
        LLVM::GEPOp::create(rewriter, loc, ptrTy, i32Ty, gep.getBase(), ValueRange{dwordOffset});
    if (constantElems != 0) {
      int64_t constantBytes = constantElems * (vecTy.getElementType().getIntOrFloatBitWidth() / 8);
      i32Ptr = LLVM::GEPOp::create(
          rewriter, loc, ptrTy, rewriter.getI8Type(), i32Ptr,
          ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(constantBytes)});
    }
    auto loaded = LLVM::LoadOp::create(rewriter, loc, i32Ty, i32Ptr);
    if (op.getAlignmentAttr())
      loaded.setAlignmentAttr(op.getAlignmentAttr());
    Value casted = LLVM::BitcastOp::create(rewriter, loc, op.getResult().getType(), loaded);
    rewriter.replaceOp(op, casted);
    return success();
  }
};

class IX11BOperandAddressFriendlyLoad final : public OpRewritePattern<LLVM::LoadOp> {
public:
  using OpRewritePattern<LLVM::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::LoadOp op, PatternRewriter &rewriter) const override {
    auto vecTy = dyn_cast<VectorType>(op.getResult().getType());
    if (!vecTy || vecTy.getNumElements() != 2 ||
        vecTy.getElementType().getIntOrFloatBitWidth() != 16)
      return failure();

    auto gep = op.getAddr().getDefiningOp<LLVM::GEPOp>();
    if (!gep)
      return failure();
    auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(gep.getType());
    if (!ptrTy || ptrTy.getAddressSpace() != 3)
      return failure();

    SmallVector<Value> dynamicIndices(gep.getDynamicIndices());
    if (dynamicIndices.size() != 1)
      return failure();

    Value baseElems;
    int64_t constantElems = 0;
    if (!matchBOperandS2ROffset(dynamicIndices.front(), baseElems, constantElems))
      return failure();
    if ((constantElems & 1) != 0)
      return failure();

    Location loc = op.getLoc();
    Type i32Ty = rewriter.getI32Type();
    Value laneId = IXDL::LaneIdOp::create(rewriter, loc, i32Ty);
    Value four = arith::ConstantIntOp::create(rewriter, loc, 4, 32);
    Value laneShifted = arith::ShLIOp::create(rewriter, loc, laneId, four);
    Value mask = arith::ConstantIntOp::create(rewriter, loc, 192, 32);
    Value laneHighBits = arith::AndIOp::create(rewriter, loc, laneShifted, mask);
    Value laneDwords = arith::OrIOp::create(rewriter, loc, laneId, laneHighBits);
    if (constantElems != 0) {
      Value cst = arith::ConstantIntOp::create(rewriter, loc, constantElems / 2, 32);
      laneDwords = arith::XOrIOp::create(rewriter, loc, laneDwords, cst);
    }

    Value two = arith::ConstantIntOp::create(rewriter, loc, 2, 32);
    Value baseDwords = arith::DivSIOp::create(rewriter, loc, baseElems, two);
    Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    Value uniformBaseDwords =
        IXDL::ReadlaneOp::create(rewriter, loc, i32Ty, baseDwords, zero);
    Value dwordOffset = arith::AddIOp::create(rewriter, loc, uniformBaseDwords, laneDwords);
    Value i32Ptr =
        LLVM::GEPOp::create(rewriter, loc, ptrTy, i32Ty, gep.getBase(), ValueRange{dwordOffset});
    auto loaded = LLVM::LoadOp::create(rewriter, loc, i32Ty, i32Ptr);
    if (op.getAlignmentAttr())
      loaded.setAlignmentAttr(op.getAlignmentAttr());
    Value casted = LLVM::BitcastOp::create(rewriter, loc, op.getResult().getType(), loaded);
    rewriter.replaceOp(op, casted);
    return success();
  }
};

static void cseIXDLAddressOpsInRegion(Region &region) {
  for (Block &block : region) {
    SmallVector<Operation *> scalarOps;
    Value laneId;
    SmallVector<IXDL::ReadlaneOp> readlanes;

    for (Operation &op : llvm::make_early_inc_range(block)) {
      for (Region &nested : op.getRegions())
        cseIXDLAddressOpsInRegion(nested);

      if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp, arith::RemSIOp,
              arith::ShLIOp, arith::ShRUIOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp>(&op)) {
        bool replaced = false;
        for (Operation *prev : scalarOps) {
          if (prev->getName() != op.getName())
            continue;
          if (prev->getResult(0).getType() != op.getResult(0).getType())
            continue;
          if (!llvm::equal(prev->getOperands(), op.getOperands()))
            continue;
          if (prev->getAttrDictionary() != op.getAttrDictionary())
            continue;
          op.getResult(0).replaceAllUsesWith(prev->getResult(0));
          op.erase();
          replaced = true;
          break;
        }
        if (replaced)
          continue;
        scalarOps.push_back(&op);
        continue;
      }

      if (auto curLaneId = dyn_cast<IXDL::LaneIdOp>(&op)) {
        Value cur = curLaneId->getResult(0);
        if (laneId && laneId.getType() == cur.getType()) {
          cur.replaceAllUsesWith(laneId);
          curLaneId->erase();
        } else {
          laneId = cur;
        }
        continue;
      }

      if (auto curReadlane = dyn_cast<IXDL::ReadlaneOp>(&op)) {
        Value src = curReadlane->getOperand(0);
        Value lane = curReadlane->getOperand(1);
        Type resultTy = curReadlane->getResult(0).getType();
        for (IXDL::ReadlaneOp prevReadlane : readlanes) {
          if (prevReadlane->getResult(0).getType() != resultTy)
            continue;
          if (prevReadlane->getOperand(0) != src || prevReadlane->getOperand(1) != lane)
            continue;
          curReadlane->getResult(0).replaceAllUsesWith(prevReadlane->getResult(0));
          curReadlane->erase();
          curReadlane = nullptr;
          break;
        }
        if (curReadlane)
          readlanes.push_back(curReadlane);
      }
    }
  }
}

static void cseIXDLAddressOps(Operation *op) {
  for (Region &region : op->getRegions())
    cseIXDLAddressOpsInRegion(region);
}

class FlyToROCDLConversionPass
    : public mlir::impl::FlyToROCDLConversionPassBase<FlyToROCDLConversionPass> {
public:
  using mlir::impl::FlyToROCDLConversionPassBase<
      FlyToROCDLConversionPass>::FlyToROCDLConversionPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    ConversionTarget target(getContext());

    target.addLegalDialect<arith::ArithDialect, scf::SCFDialect, vector::VectorDialect,
                           gpu::GPUDialect, func::FuncDialect, LLVM::LLVMDialect,
                           ROCDL::ROCDLDialect, IXDL::IXDLDialect>();
    target.addIllegalDialect<fly::FlyDialect, fly_rocdl::FlyROCDLDialect>();

    // Constructors
    target.addLegalOp<StaticOp, MakeIntTupleOp, MakeLayoutOp, MakeComposedLayoutOp>();

    FlyTypeConverter typeConverter;

    // Ensure function signatures are type-converted; otherwise conversions may rely on
    // inserted unrealized casts that remain live.
    target.addDynamicallyLegalOp<func::FuncOp>(
        [&](func::FuncOp op) { return typeConverter.isSignatureLegal(op.getFunctionType()); });
    target.addDynamicallyLegalOp<gpu::GPUFuncOp>(
        [&](gpu::GPUFuncOp op) { return typeConverter.isSignatureLegal(op.getFunctionType()); });

    // IMPORTANT: `gpu.launch_func` itself is in a legal dialect, but its kernel operands may
    // still carry illegal `!fly.memref` types. If we don't mark it dynamically illegal in that
    // case, partial conversion won't try to rewrite it, leaving `unrealized_conversion_cast`
    // users alive and causing legalization failure.
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

      // Async operands are part of the operand list; keep them consistent as well.
      for (Value dep : op.getAsyncDependencies())
        if (!isValueLegal(dep))
          return false;
      if (!isValueLegal(op.getAsyncObject()))
        return false;

      // Dimensions are typically index and already legal; no need to special-case.
      return true;
    });

    patterns.add<MakePtrOpLowering, GetDynSharedOpLowering>(typeConverter, context);
    patterns.add<IntToPtrOpLowering, PtrToIntOpLowering>(typeConverter, context);
    patterns.add<ApplySwizzleOpLowering, RecastIterOpLowering>(typeConverter, context);
    patterns.add<AddOffsetOpLowering>(typeConverter, context);
    patterns.add<MakeViewOpLowering>(typeConverter, context);
    patterns.add<PtrLoadOpLowering, PtrStoreOpLowering>(typeConverter, context);
    patterns.add<MakeCopyAtomOpLowering, MakeMmaAtomOpLowering>(typeConverter, context);
    patterns.add<MakeTiledCopyOpLowering, MakeTiledMmaOpLowering>(typeConverter, context);
    patterns.add<AtomSetValueOpLowering>(typeConverter, context);
    patterns.add<CopyAtomCallLowering, MmaAtomCallLowering>(typeConverter, context);
    patterns.add<CopyAtomCallSSALowering, MmaAtomCallSSALowering>(typeConverter, context);
    patterns.add<CpAsyncCommitGroupLowering, CpAsyncWaitGroupLowering, PipebarReqLowering,
                 PipebarWaitLowering, SlWaitcntLowering>(typeConverter, context);
    patterns.add<GpuLaunchFuncOpLowering>(typeConverter, context);

    // TODO: deprecated in the future
    patterns.add<ExtractAlignedPointerAsIndexLowering>(typeConverter, context);

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
    populateFunctionOpInterfaceTypeConversionPattern<gpu::GPUFuncOp>(patterns, typeConverter);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    RewritePatternSet ix11S2RPatterns(context);
    ix11S2RPatterns.add<IX11AOperandBlkloadFriendlyLoad, IX11BOperandAddressFriendlyLoad>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(ix11S2RPatterns))))
      signalPassFailure();

    cseIXDLAddressOps(getOperation());
  }
};

// ---------------------------------------------------------------------------
// FlyROCDLClusterAttrPass — inject amdgpu-cluster-dims into llvm.func
// passthrough.  Run inside gpu.module() AFTER convert-gpu-to-rocdl.
//
// The upstream ROCDL dialect does not translate `rocdl.cluster_dims` to the
// LLVM IR function attribute `amdgpu-cluster-dims`.  This pass bridges the
// gap by converting the discardable attribute that `GPUFuncOpLowering`
// copied from gpu.func into an LLVM passthrough entry that the LLVM IR
// emitter honours.
// ---------------------------------------------------------------------------
class FlyROCDLClusterAttrPass
    : public mlir::impl::FlyROCDLClusterAttrPassBase<FlyROCDLClusterAttrPass> {
public:
  using mlir::impl::FlyROCDLClusterAttrPassBase<
      FlyROCDLClusterAttrPass>::FlyROCDLClusterAttrPassBase;

  void runOnOperation() override {
    getOperation()->walk([&](LLVM::LLVMFuncOp func) {
      auto clusterAttr = func->getAttrOfType<StringAttr>("rocdl.cluster_dims");
      if (!clusterAttr)
        return;

      MLIRContext *ctx = func.getContext();

      // Build the new passthrough entry: ["amdgpu-cluster-dims", "2,2,1"].
      auto key = StringAttr::get(ctx, "amdgpu-cluster-dims");
      auto entry = ArrayAttr::get(ctx, {key, clusterAttr});

      // Append to existing passthrough list (if any).
      SmallVector<Attribute, 4> passthroughAttrs;
      if (auto existing = func.getPassthroughAttr())
        passthroughAttrs.append(existing.begin(), existing.end());
      passthroughAttrs.push_back(entry);

      func.setPassthroughAttr(ArrayAttr::get(ctx, passthroughAttrs));
      func->removeAttr("rocdl.cluster_dims");
    });
  }
};

} // namespace
