#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ReplaceConstant.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Passes/PassBuilder.h"
#if LLVM_VERSION_MAJOR >= 22
#include "llvm/Plugins/PassPlugin.h"
#else
#include "llvm/Passes/PassPlugin.h"
#endif
#include "llvm/Support/Path.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizer.h"
#include "llvm/Transforms/Instrumentation/MemorySanitizer.h"
#include "llvm/Transforms/Instrumentation/SanitizerCoverage.h"
#include "llvm/Transforms/Instrumentation/ThreadSanitizer.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

namespace {
struct MarkSanitized : PassInfoMixin<MarkSanitized> {
  Attribute::AttrKind Kind;
  explicit MarkSanitized(Attribute::AttrKind Kind) : Kind(Kind) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    for (Function &F : M)
      if (!F.isDeclaration()) {
        F.addFnAttr(Kind);
        F.addFnAttr("frame-pointer", "all");
      }
    return PreservedAnalyses::none();
  }
};

// Emits checks for LLVM operations whose out-of-range inputs yield poison in
// Numba-generated IR, reported through the real UBSan runtime handlers.
// Layouts follow compiler-rt/lib/ubsan/ubsan_handlers.h and ubsan_value.h.
class UndefinedBehaviorChecks : public PassInfoMixin<UndefinedBehaviorChecks> {
  Module *M = nullptr;
  LLVMContext *C = nullptr;
  Type *I16 = nullptr, *I32 = nullptr, *I64 = nullptr;
  PointerType *Ptr = nullptr;
  StringMap<GlobalVariable *> Descriptors;
  StringMap<GlobalVariable *> Filenames;

  Constant *typeDescriptor(uint16_t Kind, uint16_t Info, StringRef Name) {
    std::string Key = (Twine(Kind) + ":" + Twine(Info) + ":" + Name).str();
    auto &Slot = Descriptors[Key];
    if (!Slot) {
      Constant *Text = ConstantDataArray::getString(*C, Name);
      auto *Type = StructType::get(*C, {I16, I16, Text->getType()});
      Slot = new GlobalVariable(
          *M, Type, true, GlobalValue::PrivateLinkage,
          ConstantStruct::get(Type, {ConstantInt::get(I16, Kind),
                                     ConstantInt::get(I16, Info), Text}),
          "__numba_ubsan_type");
      Slot->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
      Slot->setAlignment(Align(2));
    }
    return Slot;
  }

  Constant *integerType(unsigned Bits, bool Signed) {
    std::string Name = (Twine("'") + (Signed ? "int" : "uint") + Twine(Bits) + "'").str();
    return typeDescriptor(0, (Log2_32(Bits) << 1) | Signed, Name);
  }

  Constant *floatType(Type *T) {
    unsigned Bits = T->getPrimitiveSizeInBits();
    return typeDescriptor(1, Bits, (Twine("'float") + Twine(Bits) + "'").str());
  }

  // Writable: the runtime atomically marks a location as already reported.
  Constant *data(Instruction &I, ArrayRef<Constant *> Types) {
    std::string Filename = "<numba jit: no debug location>";
    unsigned Line = 0, Column = 0;
    if (const DILocation *Loc = I.getDebugLoc()) {
      SmallString<256> Path(Loc->getFilename());
      if (!sys::path::is_absolute(Path) && !Loc->getDirectory().empty()) {
        Path = Loc->getDirectory();
        sys::path::append(Path, Loc->getFilename());
      }
      if (!Path.empty())
        Filename = std::string(Path);
      Line = Loc->getLine();
      Column = Loc->getColumn();
    }
    auto &Name = Filenames[Filename];
    if (!Name) {
      Constant *Text = ConstantDataArray::getString(*C, Filename);
      Name = new GlobalVariable(*M, Text->getType(), true,
                                GlobalValue::PrivateLinkage, Text,
                                "__numba_ubsan_file");
      Name->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    }
    SmallVector<Type *, 5> Fields = {Ptr, I32, I32};
    SmallVector<Constant *, 5> Values = {Name, ConstantInt::get(I32, Line),
                                         ConstantInt::get(I32, Column)};
    for (Constant *T : Types) {
      Fields.push_back(Ptr);
      Values.push_back(T);
    }
    auto *Type = StructType::get(*C, Fields);
    return new GlobalVariable(*M, Type, false, GlobalValue::PrivateLinkage,
                              ConstantStruct::get(Type, Values),
                              "__numba_ubsan_data");
  }

  Value *handle(IRBuilder<> &B, Value *V) {
    Type *T = V->getType();
    if (T->isFloatingPointTy())
      V = B.CreateBitCast(V, IntegerType::get(*C, T->getPrimitiveSizeInBits()));
    return B.CreateZExtOrTrunc(V, I64);
  }

  void report(Instruction &I, Value *Bad, StringRef Handler,
              ArrayRef<Constant *> Types, ArrayRef<Value *> Args) {
    Instruction *Then = SplitBlockAndInsertIfThen(Bad, &I, false);
    IRBuilder<> B(Then);
    B.SetCurrentDebugLocation(I.getDebugLoc());
    SmallVector<Type *, 3> Params = {Ptr};
    SmallVector<Value *, 3> Values = {data(I, Types)};
    for (Value *Arg : Args) {
      Params.push_back(I64);
      Values.push_back(handle(B, Arg));
    }
    FunctionCallee Callee = M->getOrInsertFunction(
        Handler, FunctionType::get(Type::getVoidTy(*C), Params, false));
    B.CreateCall(Callee, Values);
  }

  bool shift(BinaryOperator &I) {
    auto *T = dyn_cast<IntegerType>(I.getType());
    if (!T || T->getBitWidth() > 64)
      return false;
    Value *Amount = I.getOperand(1);
    if (auto *K = dyn_cast<ConstantInt>(Amount))
      if (K->getValue().ult(T->getBitWidth()))
        return false;
    IRBuilder<> B(&I);
    Value *Bad = B.CreateICmpUGE(Amount, ConstantInt::get(T, T->getBitWidth()));
    Constant *Type = integerType(T->getBitWidth(), true);
    report(I, Bad, "__ubsan_handle_shift_out_of_bounds", {Type, Type},
           {I.getOperand(0), Amount});
    return true;
  }

  bool floatCast(CastInst &I, bool Signed) {
    Type *Source = I.getSrcTy();
    auto *Target = dyn_cast<IntegerType>(I.getDestTy());
    if (!Source->isFloatingPointTy() || !Target || Target->getBitWidth() > 64)
      return false;
    const fltSemantics &Semantics = Source->getFltSemantics();
    unsigned Width = Target->getBitWidth();
    // Mirrors Clang's -fsanitize=float-cast-overflow bounds computation.
    APFloat Min(Semantics, APFloat::uninitialized);
    if (Min.convertFromAPInt(APSInt::getMinValue(Width, !Signed), Signed,
                             APFloat::rmTowardZero) & APFloat::opOverflow)
      Min = APFloat::getInf(Semantics, true);
    else
      Min.subtract(APFloat(Semantics, 1), APFloat::rmTowardNegative);
    APFloat Max(Semantics, APFloat::uninitialized);
    if (Max.convertFromAPInt(APSInt::getMaxValue(Width, !Signed), Signed,
                             APFloat::rmTowardZero) & APFloat::opOverflow)
      Max = APFloat::getInf(Semantics, false);
    else
      Max.add(APFloat(Semantics, 1), APFloat::rmTowardPositive);
    IRBuilder<> B(&I);
    llvm::Value *Operand = I.getOperand(0);
    llvm::Value *InRange = B.CreateAnd(
        B.CreateFCmpOGT(Operand, ConstantFP::get(Source, Min)),
        B.CreateFCmpOLT(Operand, ConstantFP::get(Source, Max)));
    report(I, B.CreateNot(InRange), "__ubsan_handle_float_cast_overflow",
           {floatType(Source), integerType(Width, Signed)}, {Operand});
    return true;
  }

public:
  PreservedAnalyses run(Module &Module, ModuleAnalysisManager &) {
    M = &Module;
    C = &Module.getContext();
    I16 = Type::getInt16Ty(*C);
    I32 = Type::getInt32Ty(*C);
    I64 = Type::getInt64Ty(*C);
    Ptr = PointerType::getUnqual(*C);
    SmallVector<Instruction *, 32> Work;
    for (Function &F : Module)
      for (Instruction &I : instructions(F))
        if (isa<FPToSIInst>(I) || isa<FPToUIInst>(I) ||
            I.getOpcode() == Instruction::Shl ||
            I.getOpcode() == Instruction::LShr ||
            I.getOpcode() == Instruction::AShr)
          Work.push_back(&I);
    bool Changed = false;
    for (Instruction *I : Work) {
      if (auto *Cast = dyn_cast<FPToSIInst>(I))
        Changed |= floatCast(*Cast, true);
      else if (auto *Cast = dyn_cast<FPToUIInst>(I))
        Changed |= floatCast(*Cast, false);
      else
        Changed |= shift(*cast<BinaryOperator>(I));
    }
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};
// With -fsanitize-memory-param-retval (Clang's default), arguments are noundef
// and MSan checks them at the call site. Numba omits noundef, so values leaving
// instrumented JIT code for uninstrumented runtimes would escape unchecked.
struct CheckExternalArguments : PassInfoMixin<CheckExternalArguments> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    bool Changed = false;
    for (Function &F : M)
      for (Instruction &I : instructions(F)) {
        auto *Call = dyn_cast<CallBase>(&I);
        Function *Callee = Call ? Call->getCalledFunction() : nullptr;
        if (!Callee || !Callee->isDeclaration() || Callee->isIntrinsic())
          continue;
        for (unsigned Index = 0; Index < Call->arg_size(); ++Index) {
          Type *T = Call->getArgOperand(Index)->getType();
          if (T->isIntegerTy() || T->isFloatingPointTy() || T->isPointerTy()) {
            Call->addParamAttr(Index, Attribute::NoUndef);
            Changed = true;
          }
        }
      }
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

// Uninstrumented callees (CPython, Numba's C runtime) write through pointers
// without updating shadow memory. Treat stack objects whose address escapes to
// such a call as initialized afterwards, avoiding false positives on out-params.
struct UnpoisonExternalOutputs : PassInfoMixin<UnpoisonExternalOutputs> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    const DataLayout &Layout = M.getDataLayout();
    auto *Ptr = PointerType::getUnqual(M.getContext());
    auto *Size = Type::getInt64Ty(M.getContext());
    FunctionCallee Unpoison = M.getOrInsertFunction(
        "__msan_unpoison", FunctionType::get(Type::getVoidTy(M.getContext()),
                                             {Ptr, Size}, false));
    bool Changed = false;
    for (Function &F : M) {
      SmallVector<std::pair<CallBase *, AllocaInst *>, 16> Work;
      for (Instruction &I : instructions(F)) {
        auto *Call = dyn_cast<CallBase>(&I);
        Function *Callee = Call ? Call->getCalledFunction() : nullptr;
        if (!Callee || !Callee->isDeclaration() || Callee->isIntrinsic() ||
            Callee->getName().starts_with("__msan_"))
          continue;
        SmallPtrSet<AllocaInst *, 4> Seen;
        for (Value *Arg : Call->args())
          if (Arg->getType()->isPointerTy())
            if (auto *Alloca = dyn_cast<AllocaInst>(getUnderlyingObject(Arg)))
              if (Alloca->getAllocationSize(Layout) && Seen.insert(Alloca).second)
                Work.push_back({Call, Alloca});
      }
      for (auto [Call, Alloca] : Work) {
        IRBuilder<> B(Call->getNextNode());
        B.CreateCall(Unpoison, {Alloca, ConstantInt::get(
            Size, Alloca->getAllocationSize(Layout)->getFixedValue())});
        Changed = true;
      }
    }
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

// SanitizerCoverage registers its per-function counter and PC-table arrays with
// libFuzzer through linker-synthesized __start_/__stop_ section symbols, which
// MCJIT never defines. Pack them into one contiguous array per module (same
// order, so counter i still corresponds to PC-table entry i) and register the
// packed arrays' bounds instead.
struct PackSanitizerCoverage : PassInfoMixin<PackSanitizerCoverage> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    SmallVector<GlobalVariable *, 32> Counters, Tables;
    for (GlobalVariable &G : M.globals()) {
      if (G.getSection() == "__sancov_cntrs")
        Counters.push_back(&G);
      else if (G.getSection() == "__sancov_pcs")
        Tables.push_back(&G);
    }
    if (Counters.empty())
      return PreservedAnalyses::all();
    if (Counters.size() != Tables.size())
      report_fatal_error("sanitizer coverage counters and PC tables differ");
    LLVMContext &C = M.getContext();
    auto *I8 = Type::getInt8Ty(C);
    auto *I64 = Type::getInt64Ty(C);
    auto *Ptr = PointerType::getUnqual(C);
    uint64_t Total = 0;
    SmallVector<Constant *, 256> Entries;
    for (size_t Index = 0; Index < Counters.size(); ++Index) {
      uint64_t Size = cast<ArrayType>(Counters[Index]->getValueType())->getNumElements();
      Constant *Init = Tables[Index]->getInitializer();
      if (cast<ArrayType>(Init->getType())->getNumElements() != 2 * Size)
        report_fatal_error("sanitizer coverage PC table does not match counters");
      for (uint64_t Entry = 0; Entry < 2 * Size; ++Entry)
        Entries.push_back(Init->getAggregateElement(Entry));
      Total += Size;
    }
    auto *CountersType = ArrayType::get(I8, Total);
    auto *AllCounters = new GlobalVariable(
        M, CountersType, false, GlobalValue::PrivateLinkage,
        ConstantAggregateZero::get(CountersType), "__numba_sancov_counters");
    auto *TableType = ArrayType::get(Ptr, 2 * Total);
    auto *AllTables = new GlobalVariable(
        M, TableType, true, GlobalValue::PrivateLinkage,
        ConstantArray::get(TableType, Entries), "__numba_sancov_pcs");
    AllTables->setAlignment(Align(8));
    // As in Clang: coverage arrays are not program data; keep them out of ASan.
    GlobalValue::SanitizerMetadata Metadata;
    Metadata.NoAddress = true;
    AllCounters->setSanitizerMetadata(Metadata);
    AllTables->setSanitizerMetadata(Metadata);
    if (GlobalVariable *Lowest = M.getNamedGlobal("__sancov_lowest_stack"))
      if (Lowest->use_empty())
        Lowest->eraseFromParent();

    DenseSet<Constant *> Old(Counters.begin(), Counters.end());
    Old.insert(Tables.begin(), Tables.end());
    removeFromUsedLists(M, [&](Constant *C) { return Old.contains(C); });
    uint64_t Offset = 0;
    for (GlobalVariable *G : Counters) {
      G->replaceAllUsesWith(ConstantExpr::getInBoundsGetElementPtr(
          I8, AllCounters, ConstantInt::get(I64, Offset)));
      Offset += cast<ArrayType>(G->getValueType())->getNumElements();
    }
    auto Bound = [&](StringRef Name, Constant *Value) {
      if (GlobalVariable *G = M.getNamedGlobal(Name)) {
        G->replaceAllUsesWith(Value);
        G->eraseFromParent();
      }
    };
    Bound("__start___sancov_cntrs", AllCounters);
    Bound("__stop___sancov_cntrs", ConstantExpr::getInBoundsGetElementPtr(
                                       I8, AllCounters, ConstantInt::get(I64, Total)));
    Bound("__start___sancov_pcs", AllTables);
    Bound("__stop___sancov_pcs", ConstantExpr::getInBoundsGetElementPtr(
                                     Ptr, AllTables, ConstantInt::get(I64, 2 * Total)));
    for (GlobalVariable *G : Counters)
      G->eraseFromParent();
    for (GlobalVariable *G : Tables)
      G->eraseFromParent();
    return PreservedAnalyses::none();
  }
};

// MCJIT's RuntimeDyld does not correctly resolve initial-exec TLS relocations
// (R_X86_64_GOTTPOFF) against the sanitizer runtime's thread-local shadow
// state. Each function instead obtains the current thread's address from a
// getter in the sanitizer-built bridge, which uses dynamic-linker TLS.
struct ReplaceSanitizerThreadLocals : PassInfoMixin<ReplaceSanitizerThreadLocals> {
  StringRef Prefix;
  explicit ReplaceSanitizerThreadLocals(StringRef Prefix) : Prefix(Prefix) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    SmallVector<GlobalVariable *, 8> Globals;
    for (GlobalVariable &G : M.globals())
      if (G.isThreadLocal() && G.isDeclaration() && G.getName().starts_with(Prefix))
        Globals.push_back(&G);
    if (Globals.empty())
      return PreservedAnalyses::all();
    SmallVector<Constant *, 8> Constants(Globals.begin(), Globals.end());
    convertUsersOfConstantsToInstructions(Constants);
    auto *Ptr = PointerType::getUnqual(M.getContext());
    for (GlobalVariable *G : Globals) {
      FunctionCallee Getter = M.getOrInsertFunction(
          ("numba_tls_" + G->getName()).str(), FunctionType::get(Ptr, false));
      DenseMap<Function *, Value *> Addresses;
      for (Use &U : make_early_inc_range(G->uses())) {
        auto *I = dyn_cast<Instruction>(U.getUser());
        if (!I)
          report_fatal_error("unexpected non-instruction use of sanitizer TLS");
        Function *F = I->getFunction();
        Value *&Address = Addresses[F];
        if (!Address) {
          IRBuilder<> B(&*F->getEntryBlock().getFirstInsertionPt());
          Address = B.CreateCall(Getter);
        }
        if (auto *II = dyn_cast<IntrinsicInst>(I);
            II && II->getIntrinsicID() == Intrinsic::threadlocal_address) {
          II->replaceAllUsesWith(Address);
          II->eraseFromParent();
        } else {
          U.set(Address);
        }
      }
    }
    return PreservedAnalyses::none();
  }
};
} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "numba-sanitizers", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "numba-asan") {
                    MPM.addPass(MarkSanitized(Attribute::SanitizeAddress));
                    AddressSanitizerOptions Options;
                    Options.UseAfterScope = true;
                    // MCJIT does not provide ELF start/stop section symbols.
                    MPM.addPass(AddressSanitizerPass(
                        Options, /*UseGlobalGC=*/false,
                        /*UseOdrIndicator=*/false));
                    return true;
                  }
                  if (Name == "numba-tsan") {
                    MPM.addPass(MarkSanitized(Attribute::SanitizeThread));
                    MPM.addPass(ModuleThreadSanitizerPass());
                    MPM.addPass(createModuleToFunctionPassAdaptor(
                        ThreadSanitizerPass()));
                    return true;
                  }
                  if (Name == "numba-msan") {
                    MPM.addPass(MarkSanitized(Attribute::SanitizeMemory));
                    MPM.addPass(CheckExternalArguments());
                    // Eager checks: callees never read parameter shadow that
                    // uninstrumented callers (CPython, Numba) cannot provide.
                    MPM.addPass(MemorySanitizerPass(MemorySanitizerOptions(
                        /*TrackOrigins=*/2, /*Recover=*/false,
                        /*Kernel=*/false, /*EagerChecks=*/true)));
                    MPM.addPass(UnpoisonExternalOutputs());
                    MPM.addPass(ReplaceSanitizerThreadLocals("__msan_"));
                    return true;
                  }
                  if (Name == "numba-ubsan") {
                    MPM.addPass(UndefinedBehaviorChecks());
                    return true;
                  }
                  if (Name == "numba-sancov") {
                    // Clang's -fsanitize=fuzzer-no-link, minus stack-depth
                    // (its initial-exec TLS counter is unsupported by MCJIT).
                    SanitizerCoverageOptions Options;
                    Options.CoverageType = SanitizerCoverageOptions::SCK_Edge;
                    Options.IndirectCalls = true;
                    Options.TraceCmp = true;
                    Options.Inline8bitCounters = true;
                    Options.PCTable = true;
                    MPM.addPass(SanitizerCoveragePass(Options));
                    MPM.addPass(PackSanitizerCoverage());
                    return true;
                  }
                  return false;
                });
          }};
}
