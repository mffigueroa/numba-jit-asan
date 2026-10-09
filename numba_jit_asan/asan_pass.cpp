#include "llvm/IR/Attributes.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizer.h"

using namespace llvm;

namespace {
struct MarkAddressSanitized : PassInfoMixin<MarkAddressSanitized> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    for (Function &F : M)
      if (!F.isDeclaration()) {
        F.addFnAttr(Attribute::SanitizeAddress);
        F.addFnAttr("frame-pointer", "all");
      }
    return PreservedAnalyses::none();
  }
};
} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "numba-asan", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name != "numba-asan")
                    return false;
                  MPM.addPass(MarkAddressSanitized());
                  AddressSanitizerOptions Options;
                  Options.UseAfterScope = true;
                  // MCJIT does not provide ELF start/stop section symbols.
                  MPM.addPass(AddressSanitizerPass(
                      Options, /*UseGlobalGC=*/false,
                      /*UseOdrIndicator=*/false));
                  return true;
                });
          }};
}
