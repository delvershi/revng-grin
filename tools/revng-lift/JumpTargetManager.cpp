/// \file jumptargetmanager.cpp
/// \brief This file handles the possible jump targets encountered during
///        translation and the creation and management of the respective
///        BasicBlock.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include "revng/Support/Assert.h"
#include <cstdint>
#include <fstream>
#include <queue>
#include <sstream>

// Boost includes
#include <boost/icl/interval_set.hpp>
#include <boost/icl/right_open_interval.hpp>
#include <boost/type_traits/is_same.hpp>

// LLVM includes
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Endian.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Cloning.h"

// Local libraries includes
#include "revng/ADT/Queue.h"
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/ReachingDefinitions/ReachingDefinitionsPass.h"
#include "revng/Support/CommandLine.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/revng.h"

// Local includes
#include "JumpTargetManager.h"
#include "SET.h"
#include "SimplifyComparisonsPass.h"
#include "SubGraph.h"

using namespace llvm;

namespace {

Logger<> JTCountLog("jtcount");

cl::opt<bool> Statistics("Statistics",
                       cl::desc("Count rewriting information"),
                       cl::cat(MainCategory));
cl::opt<bool> FAST("fast",
                       cl::desc("fast rewriting"),
                       cl::cat(MainCategory));
cl::opt<bool> SUPERFAST("super-fast",
                       cl::desc("fast rewriting"),
                       cl::cat(MainCategory));



cl::opt<bool> NoOSRA("no-osra", cl::desc(" OSRA"), cl::cat(MainCategory));
cl::alias A1("O",
             cl::desc("Alias for -no-osra"),
             cl::aliasopt(NoOSRA),
             cl::cat(MainCategory));

RegisterPass<TranslateDirectBranchesPass> X("translate-db",
                                            "Translate Direct Branches"
                                            " Pass",
                                            false,
                                            false);

// TODO: this is kind of an abuse
Logger<> Verify("verify");
Logger<> RegisterJTLog("registerjt");

} // namespace

char TranslateDirectBranchesPass::ID = 0;

static bool isSumJump(StoreInst *PCWrite);

void TranslateDirectBranchesPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addUsedIfAvailable<SETPass>();
  AU.setPreservesAll();
}

/// \brief Purges everything is after a call to exitTB (except the call itself)
static void exitTBCleanup(Instruction *ExitTBCall) {
  BasicBlock *BB = ExitTBCall->getParent();

  // Cleanup everything it's aftewards starting from the end
  Instruction *ToDelete = &*(--BB->end());
  while (ToDelete != ExitTBCall) {
    if (auto DeadBranch = dyn_cast<BranchInst>(ToDelete))
      purgeBranch(BasicBlock::iterator(DeadBranch));
    else
      ToDelete->eraseFromParent();

    ToDelete = &*(--BB->end());
  }
}

bool TranslateDirectBranchesPass::pinJTs(Function &F) {
  const auto *SET = getAnalysisIfAvailable<SETPass>();
  if (SET == nullptr || SET->jumps().size() == 0)
    return false;

  LLVMContext &Context = getContext(&F);
  Value *PCReg = JTM->pcReg();
  auto *RegType = cast<IntegerType>(PCReg->getType()->getPointerElementType());
  auto C = [RegType](uint64_t A) { return ConstantInt::get(RegType, A); };
  BasicBlock *AnyPC = JTM->anyPC();
  BasicBlock *UnexpectedPC = JTM->unexpectedPC();
  // TODO: enforce CFG

  for (const auto &Jump : SET->jumps()) {
    StoreInst *PCWrite = Jump.Instruction;
    bool Approximate = Jump.Approximate;
    const std::vector<uint64_t> &Destinations = Jump.Destinations;

    // We don't care if we already handled this call too exitTB in the past,
    // information should become progressively more precise, so let's just
    // remove everything after this call and put a new handler
    CallInst *CallExitTB = JTM->findNextExitTB(PCWrite);

    revng_assert(CallExitTB != nullptr);
    revng_assert(PCWrite->getParent()->getParent() == &F);
    revng_assert(JTM->isPCReg(PCWrite->getPointerOperand()));
    revng_assert(Destinations.size() != 0);

    auto *ExitTBArg = ConstantInt::get(Type::getInt32Ty(Context),
                                       Destinations.size());
    uint64_t OldTargetsCount = getLimitedValue(CallExitTB->getArgOperand(0));

    // TODO: we should check Destinations.size() >= OldTargetsCount
    // TODO: we should also check the destinations are actually the same

    BasicBlock *FailBB = Approximate ? AnyPC : UnexpectedPC;
    BasicBlock *BB = CallExitTB->getParent();

    // Kill everything is after the call to exitTB
    exitTBCleanup(CallExitTB);

    // Mark this call to exitTB as handled
    CallExitTB->setArgOperand(0, ExitTBArg);

    IRBuilder<> Builder(BB);
    auto PCLoad = Builder.CreateLoad(PCReg);
    if (Destinations.size() == 1) {
      auto *Comparison = Builder.CreateICmpEQ(C(Destinations[0]), PCLoad);
      Builder.CreateCondBr(Comparison,
                           JTM->getBlockAt(Destinations[0]),
                           FailBB);
    } else {
      auto *Switch = Builder.CreateSwitch(PCLoad, FailBB, Destinations.size());
      for (uint64_t Destination : Destinations)
        Switch->addCase(C(Destination), JTM->getBlockAt(Destination));
    }

    // Move all the markers right before the branch instruction
    Instruction *Last = BB->getTerminator();
    auto It = CallExitTB->getIterator();
    while (isMarker(&*It)) {
      // Get the marker instructions
      Instruction *I = &*It;

      // Move the iterator back
      It--;

      // Move the last moved instruction (initially the terminator)
      I->moveBefore(Last);

      Last = I;
    }

    // Notify new branches only if the amount of possible targets actually
    // increased
    if (Destinations.size() > OldTargetsCount)
      JTM->newBranch();
  }

  return true;
}

bool TranslateDirectBranchesPass::pinConstantStore(Function &F) {
  auto &Context = F.getParent()->getContext();

  Function *ExitTB = JTM->exitTB();
  auto ExitTBIt = ExitTB->use_begin();
  while (ExitTBIt != ExitTB->use_end()) {
    // Take note of the use and increment the iterator immediately: this allows
    // us to erase the call to exit_tb without unexpected behaviors
    Use &ExitTBUse = *ExitTBIt++;
    if (auto Call = dyn_cast<CallInst>(ExitTBUse.getUser())) {
      if (Call->getCalledFunction() == ExitTB) {
        // Look for the last write to the PC
        StoreInst *PCWrite = JTM->getPrevPCWrite(Call);

        // Is destination a constant?
        if (PCWrite == nullptr) {
          //forceFallthroughAfterHelper(Call);
        } else {
          //uint64_t NextPC = JTM->getNextPC(PCWrite);
          //if (NextPC != 0 && not NoOSRA && isSumJump(PCWrite))
          //  JTM->registerJT(NextPC, JTReason::SumJump);

          auto *Address = dyn_cast<ConstantInt>(PCWrite->getValueOperand());
          if (Address != nullptr) {
            // Compute the actual PC and get the associated BasicBlock
            uint64_t TargetPC = Address->getSExtValue();
	    if(JTM->isIllegalStaticAddr(TargetPC))
                continue;
            auto *TargetBlock = JTM->obtainJTBB(TargetPC, JTReason::DirectJump);
	    if(TargetBlock==nullptr)
		continue;

            // Remove unreachable right after the exit_tb
            BasicBlock::iterator CallIt(Call);
            BasicBlock::iterator BlockEnd = Call->getParent()->end();
            CallIt++;
            revng_assert(CallIt != BlockEnd && isa<UnreachableInst>(&*CallIt));
            CallIt->eraseFromParent();

            // Cleanup of what's afterwards (only a unconditional jump is
            // allowed)
            CallIt = BasicBlock::iterator(Call);
            BlockEnd = Call->getParent()->end();
            if (++CallIt != BlockEnd)
              purgeBranch(CallIt);

            if (TargetBlock != nullptr) {
              // A target was found, jump there
              BranchInst::Create(TargetBlock, Call);
              JTM->newBranch();
            } else {
              // We're jumping to an invalid location, abort everything
              // TODO: emit a warning
              CallInst::Create(F.getParent()->getFunction("abort"), {}, Call);
              new UnreachableInst(Context, Call);
            }
            Call->eraseFromParent();
          }
        }
      } else {
        revng_unreachable("Unexpected instruction using the PC");
      }
    } else {
      revng_unreachable("Unhandled usage of the PC");
    }
  }

  return true;
}

bool TranslateDirectBranchesPass::forceFallthroughAfterHelper(CallInst *Call) {
  // If someone else already took care of the situation, quit
  if (getLimitedValue(Call->getArgOperand(0)) > 0)
    return false;

  auto *PCReg = JTM->pcReg();
  auto PCRegTy = PCReg->getType()->getPointerElementType();
  bool ForceFallthrough = false;

  BasicBlock::reverse_iterator It(++Call->getReverseIterator());
  auto *BB = Call->getParent();
  auto EndIt = BB->rend();
  while (!ForceFallthrough) {
    while (It != EndIt) {
      Instruction *I = &*It;
      if (auto *Store = dyn_cast<StoreInst>(I)) {
        if (Store->getPointerOperand() == PCReg) {
          // We found a PC-store, give up
          return false;
        }
      } else if (auto *Call = dyn_cast<CallInst>(I)) {
        if (Function *Callee = Call->getCalledFunction()) {
          if (Callee->getName().startswith("helper_")) {
            // We found a call to an helper
            ForceFallthrough = true;
            break;
          }
        }
      }
      It++;
    }

    if (!ForceFallthrough) {
      // Proceed only to unique predecessor, if present
      if (auto *Pred = BB->getUniquePredecessor()) {
        BB = Pred;
        It = BB->rbegin();
        EndIt = BB->rend();
      } else {
        // We have multiple predecessors, give up
        return false;
      }
    }
  }

  exitTBCleanup(Call);
  JTM->newBranch();

  IRBuilder<> Builder(Call->getParent());
  Call->setArgOperand(0, Builder.getInt32(1));

  // Create the fallthrough jump
  uint64_t NextPC = JTM->getNextPC(Call);
  Value *NextPCConst = Builder.getIntN(PCRegTy->getIntegerBitWidth(), NextPC);

  // Get the fallthrough basic block and emit a conditional branch, if not
  // possible simply jump to anyPC
  BasicBlock *NextPCBB = JTM->registerJT(NextPC, JTReason::PostHelper);
  if (NextPCBB != nullptr) {
    Builder.CreateCondBr(Builder.CreateICmpEQ(Builder.CreateLoad(PCReg),
                                              NextPCConst),
                         NextPCBB,
                         JTM->anyPC());
  } else {
    Builder.CreateBr(JTM->anyPC());
  }

  return true;
}

bool TranslateDirectBranchesPass::runOnModule(Module &M) {
  Function &F = *M.getFunction("root");
  pinConstantStore(F);
  //pinJTs(F);
  return true;
}

uint64_t TranslateDirectBranchesPass::getNextPC(Instruction *TheInstruction) {
  DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();

  BasicBlock *Block = TheInstruction->getParent();
  BasicBlock::reverse_iterator It(++TheInstruction->getReverseIterator());

  while (true) {
    BasicBlock::reverse_iterator Begin(Block->rend());

    // Go back towards the beginning of the basic block looking for a call to
    // newpc
    CallInst *Marker = nullptr;
    for (; It != Begin; It++) {
      if ((Marker = dyn_cast<CallInst>(&*It))) {
        // TODO: comparing strings is not very elegant
        if (Marker->getCalledFunction()->getName() == "newpc") {
          uint64_t PC = getLimitedValue(Marker->getArgOperand(0));
          uint64_t Size = getLimitedValue(Marker->getArgOperand(1));
          revng_assert(Size != 0);
          return PC + Size;
        }
      }
    }

    auto *Node = DT.getNode(Block);
    revng_assert(Node != nullptr,
                 "BasicBlock not in the dominator tree, is it reachable?");

    Block = Node->getIDom()->getBlock();
    It = Block->rbegin();
  }

  revng_unreachable("Can't find the PC marker");
}

Constant *JumpTargetManager::readConstantPointer(Constant *Address,
                                                 Type *PointerTy,
                                                 BinaryFile::Endianess E) {
  Constant *ConstInt = readConstantInt(Address,
                                       Binary.architecture().pointerSize() / 8,
                                       E);
  if (ConstInt != nullptr) {
    return Constant::getIntegerValue(PointerTy, ConstInt->getUniqueInteger());
  } else {
    return nullptr;
  }
}

ConstantInt *JumpTargetManager::readConstantInt(Constant *ConstantAddress,
                                                unsigned Size,
                                                BinaryFile::Endianess E) {
  const DataLayout &DL = TheModule.getDataLayout();

  if (ConstantAddress->getType()->isPointerTy()) {
    using CE = ConstantExpr;
    auto IntPtrTy = Type::getIntNTy(Context,
                                    Binary.architecture().pointerSize());
    ConstantAddress = CE::getPtrToInt(ConstantAddress, IntPtrTy);
  }

  uint64_t Address = getZExtValue(ConstantAddress, DL);
  UnusedCodePointers.erase(Address);
  registerReadRange(Address, Size);

  auto Result = Binary.readRawValue(Address, Size, E);

  if (Result.hasValue())
    return ConstantInt::get(IntegerType::get(Context, Size * 8),
                            Result.getValue());
  else
    return nullptr;
}

template<typename T>
static cl::opt<T> *
getOption(StringMap<cl::Option *> &Options, const char *Name) {
  return static_cast<cl::opt<T> *>(Options[Name]);
}

JumpTargetManager::JumpTargetManager(Function *TheFunction,
                                     Value *PCReg,
                                     const BinaryFile &Binary) :
  TheModule(*TheFunction->getParent()),
  Context(TheModule.getContext()),
  TheFunction(TheFunction),
  OriginalInstructionAddresses(),
  JumpTargets(),
  assign_gadge(),
  PCReg(PCReg),
  ExitTB(nullptr),
  Dispatcher(nullptr),
  DispatcherSwitch(nullptr),
  Binary(Binary),
  NoReturn(Binary.architecture()),
  CurrentCFGForm(CFGForm::UnknownFormCFG) {
  FunctionType *ExitTBTy = FunctionType::get(Type::getVoidTy(Context),
                                             { Type::getInt32Ty(Context) },
                                             false);
  ExitTB = cast<Function>(TheModule.getOrInsertFunction("exitTB", ExitTBTy));
  createDispatcher(TheFunction, PCReg);

  for (auto &Segment : Binary.segments()){
    Segment.insertExecutableRanges(std::back_inserter(ExecutableRanges));
    if(Segment.IsExecutable){
      codeSeg_StartAddr = Segment.StartVirtualAddress;
    }
    if(Segment.IsWriteable and !Segment.IsExecutable){
      DataSegmStartAddr = Segment.StartVirtualAddress; 
      DataSegmEndAddr = Segment.EndVirtualAddress;
    }
  }
  ro_StartAddr = 0;
  ro_EndAddr = 0;
  text_StartAddr = 0;
  if(Binary.rodataStartAddr){
    ro_StartAddr = Binary.rodataStartAddr; 
    ro_EndAddr = Binary.ehframeEndAddr;
    revng_assert(ro_StartAddr<=ro_EndAddr);
  }
  if(Binary.textStartAddr)
    text_StartAddr = Binary.textStartAddr;

  // Configure GlobalValueNumbering
  StringMap<cl::Option *> &Options(cl::getRegisteredOptions());
  getOption<bool>(Options, "enable-load-pre")->setInitialValue(false);
  getOption<unsigned>(Options, "memdep-block-scan-limit")->setInitialValue(100);
  // getOption<bool>(Options, "enable-pre")->setInitialValue(false);
  // getOption<uint32_t>(Options, "max-recurse-depth")->setInitialValue(10);
  haveBB = 0;
  elf_name = nullptr;
  range = 0;
 
  auto PointerPath = "GlobalPointer.log";
  std::ofstream PointerAddrInfoStream(PointerPath);
  PointerAddrInfoStream << "Global Pointer Base Addr:\n";

  auto Path = "JumpTable.log";
  std::ofstream JumpTableAddrInfoStream(Path);
  JumpTableAddrInfoStream << "Jump Table Addr:\n";
}

JumpTargetManager::JumpTargetManager(Function *TheFunction,
                                     Value *PCReg,
                                     const BinaryFile &Binary,
                                     GlobalVariable *ELFName) :
  TheModule(*TheFunction->getParent()),
  Context(TheModule.getContext()),
  TheFunction(TheFunction),
  OriginalInstructionAddresses(),
  JumpTargets(),
  assign_gadge(),
  PCReg(PCReg),
  ExitTB(nullptr),
  Dispatcher(nullptr),
  DispatcherSwitch(nullptr),
  Binary(Binary),
  NoReturn(Binary.architecture()),
  CurrentCFGForm(CFGForm::UnknownFormCFG) {
  FunctionType *ExitTBTy = FunctionType::get(Type::getVoidTy(Context),
                                             { Type::getInt32Ty(Context) },
                                             false);
  elf_name = ELFName;
  ExitTB = cast<Function>(TheModule.getOrInsertFunction("exitTB", ExitTBTy));
  createDispatcher(TheFunction, PCReg);

  for (auto &Segment : Binary.segments()){
    Segment.insertExecutableRanges(std::back_inserter(ExecutableRanges));
    if(Segment.IsExecutable){
      codeSeg_StartAddr = Segment.StartVirtualAddress;
    }
    if(Segment.IsWriteable and !Segment.IsExecutable){
      DataSegmStartAddr = Segment.StartVirtualAddress; 
      DataSegmEndAddr = Segment.EndVirtualAddress;
    }
  }
  ro_StartAddr = 0;
  ro_EndAddr = 0;
  text_StartAddr = 0;
  if(Binary.rodataStartAddr){
    ro_StartAddr = Binary.rodataStartAddr; 
    ro_EndAddr = Binary.ehframeEndAddr;
    revng_assert(ro_StartAddr<=ro_EndAddr);
  }
  if(Binary.textStartAddr)
    text_StartAddr = Binary.textStartAddr;

  // Configure GlobalValueNumbering
  StringMap<cl::Option *> &Options(cl::getRegisteredOptions());
  getOption<bool>(Options, "enable-load-pre")->setInitialValue(false);
  getOption<unsigned>(Options, "memdep-block-scan-limit")->setInitialValue(100);
  // getOption<bool>(Options, "enable-pre")->setInitialValue(false);
  // getOption<uint32_t>(Options, "max-recurse-depth")->setInitialValue(10);
  haveBB = 0;
  range = 0;
 
  auto PointerPath = "GlobalPointer.log";
  std::ofstream PointerAddrInfoStream(PointerPath);
  PointerAddrInfoStream << "Global Pointer Base Addr:\n";

  auto Path = "JumpTable.log";
  std::ofstream JumpTableAddrInfoStream(Path);
  JumpTableAddrInfoStream << "Jump Table Addr:\n";
}

static bool isBetterThan(const Label *NewCandidate, const Label *OldCandidate) {
  if (OldCandidate == nullptr)
    return true;

  if (NewCandidate->address() > OldCandidate->address())
    return true;

  if (NewCandidate->address() == OldCandidate->address()) {
    StringRef OldName = OldCandidate->symbolName();
    if (OldName.size() == 0)
      return true;
  }

  return false;
}

// TODO: move this in BinaryFile?
std::string
JumpTargetManager::nameForAddress(uint64_t Address, uint64_t Size) const {
  std::stringstream Result;
  const auto &SymbolMap = Binary.labels();

  auto It = SymbolMap.find(interval::right_open(Address, Address + Size));
  if (It != SymbolMap.end()) {
    // We have to look for (in order):
    //
    // * Exact match
    // * Contained (non 0-sized)
    // * Contained (0-sized)
    const Label *ExactMatch = nullptr;
    const Label *ContainedNonZeroSized = nullptr;
    const Label *ContainedZeroSized = nullptr;

    for (const Label *L : It->second) {
      // Consider symbols only
      if (not L->isSymbol())
        continue;

      if (L->matches(Address, Size)) {

        // It's an exact match
        ExactMatch = L;
        break;

      } else if (not L->isSizeVirtual() and L->contains(Address, Size)) {

        // It's contained in a not 0-sized symbol
        if (isBetterThan(L, ContainedNonZeroSized))
          ContainedNonZeroSized = L;

      } else if (L->isSizeVirtual() and L->contains(Address, 0)) {

        // It's contained in a 0-sized symbol
        if (isBetterThan(L, ContainedZeroSized))
          ContainedZeroSized = L;
      }
    }

    const Label *Chosen = nullptr;
    if (ExactMatch != nullptr)
      Chosen = ExactMatch;
    else if (ContainedNonZeroSized != nullptr)
      Chosen = ContainedNonZeroSized;
    else if (ContainedZeroSized != nullptr)
      Chosen = ContainedZeroSized;

    if (Chosen != nullptr and Chosen->symbolName().size() != 0) {
      // Use the symbol name
      Result << Chosen->symbolName().str();

      // And, if necessary, an offset
      if (Address != Chosen->address())
        Result << ".0x" << std::hex << (Address - Chosen->address());

      return Result.str();
    }
  }

  // We don't have a symbol to use, just return the address
  Result << "0x" << std::hex << Address;
  return Result.str();
}

void JumpTargetManager::harvestGlobalData() {
  // Register symbols
  for (auto &P : Binary.labels())
    for (const Label *L : P.second)
      if (L->isSymbol() and L->isCode())
        registerJT(L->address(), JTReason::FunctionSymbol);

  // Register landing pads, if available
  // TODO: should register them in UnusedCodePointers?
  for (uint64_t LandingPad : Binary.landingPads())
    registerJT(LandingPad, JTReason::GlobalData);

  for (uint64_t CodePointer : Binary.codePointers())
    registerJT(CodePointer, JTReason::GlobalData);

  for (auto &Segment : Binary.segments()) {
    const Constant *Initializer = Segment.Variable->getInitializer();
    if (isa<ConstantAggregateZero>(Initializer))
      continue;

    auto *Data = cast<ConstantDataArray>(Initializer);
    uint64_t StartVirtualAddress = Segment.StartVirtualAddress;
    const unsigned char *DataStart = Data->getRawDataValues().bytes_begin();
    const unsigned char *DataEnd = Data->getRawDataValues().bytes_end();

    using endianness = support::endianness;
    if (Binary.architecture().pointerSize() == 64) {
      if (Binary.architecture().isLittleEndian())
        findCodePointers<uint64_t, endianness::little>(StartVirtualAddress,
                                                       DataStart,
                                                       DataEnd);
      else
        findCodePointers<uint64_t, endianness::big>(StartVirtualAddress,
                                                    DataStart,
                                                    DataEnd);
    } else if (Binary.architecture().pointerSize() == 32) {
      if (Binary.architecture().isLittleEndian())
        findCodePointers<uint32_t, endianness::little>(StartVirtualAddress,
                                                       DataStart,
                                                       DataEnd);
      else
        findCodePointers<uint32_t, endianness::big>(StartVirtualAddress,
                                                    DataStart,
                                                    DataEnd);
    }
  }

  revng_log(JTCountLog,
            "JumpTargets found in global data: " << std::dec
                                                 << Unexplored.size());
}

template<typename value_type, unsigned endian>
void JumpTargetManager::findCodePointers(uint64_t StartVirtualAddress,
                                         const unsigned char *Start,
                                         const unsigned char *End) {
  using support::endianness;
  using support::endian::read;
  for (auto Pos = Start; Pos < End - sizeof(value_type); Pos++) {
    uint64_t Value = read<value_type, static_cast<endianness>(endian), 1>(Pos);
    BasicBlock *Result = registerJT(Value, JTReason::GlobalData);

    if (Result != nullptr)
      UnusedCodePointers.insert(StartVirtualAddress + (Pos - Start));
  }
}

/// Handle a new program counter. We might already have a basic block for that
/// program counter, or we could even have a translation for it. Return one of
/// these, if appropriate.
///
/// \param PC the new program counter.
/// \param ShouldContinue an out parameter indicating whether the returned
///        basic block was just a placeholder or actually contains a
///        translation.
///
/// \return the basic block to use from now on, or null if the program counter
///         is not associated to a basic block.
// TODO: make this return a pair
BasicBlock *JumpTargetManager::newPC(uint64_t PC, bool &ShouldContinue) {
  // Did we already meet this PC?
  auto JTIt = JumpTargets.find(PC);
  if (JTIt != JumpTargets.end()) {
    // If it was planned to explore it in the future, just to do it now
    for (auto UnexploredIt = Unexplored.begin();
         UnexploredIt != Unexplored.end();
         UnexploredIt++) {

      if (UnexploredIt->first == PC) {
        BasicBlock *Result = UnexploredIt->second;

        // Check if we already have a translation for that
        ShouldContinue = Result->empty();
        if (ShouldContinue) {
          // We don't, OK let's explore it next
          Unexplored.erase(UnexploredIt);
        } else {
          // We do, it will be purged at the next `peek`
          revng_assert(ToPurge.count(Result) != 0);
        }

        return Result;
      }
    }

    // It wasn't planned to visit it, so we've already been there, just jump
    // there
    BasicBlock *BB = JTIt->second.head();
    revng_assert(!BB->empty());
    ShouldContinue = false;
    return BB;
  }

  // Check if we already translated this PC even if it's not associated to a
  // basic block (i.e., we have to split its basic block). This typically
  // happens with variable-length instruction encodings.
  if (OriginalInstructionAddresses.count(PC) != 0) {
    ShouldContinue = false;
    InstructionMap::iterator InstrIt = OriginalInstructionAddresses.find(PC);
    Instruction *I = InstrIt->second;
    haveBB = 1;
    return I->getParent();
    //revng_abort("Why this?\n");
    //return registerJT(PC, JTReason::AmbigousInstruction);
  }

  // We don't know anything about this PC
  return nullptr;
}

/// Save the PC-Instruction association for future use (jump target)
void JumpTargetManager::registerInstruction(uint64_t PC,
                                            Instruction *Instruction) {
  // Never save twice a PC
  revng_assert(!OriginalInstructionAddresses.count(PC));
  OriginalInstructionAddresses[PC] = Instruction;
}

CallInst *JumpTargetManager::findNextExitTB(Instruction *Start) {

  struct Visitor
    : public BFSVisitorBase<true, Visitor, SmallVector<BasicBlock *, 4>> {
  public:
    using SuccessorsType = SmallVector<BasicBlock *, 4>;

  public:
    CallInst *Result;
    Function *ExitTB;
    JumpTargetManager *JTM;

  public:
    Visitor(Function *ExitTB, JumpTargetManager *JTM) :
      Result(nullptr),
      ExitTB(ExitTB),
      JTM(JTM) {}

  public:
    VisitAction visit(BasicBlockRange Range) {
      for (Instruction &I : Range) {
        if (auto *Call = dyn_cast<CallInst>(&I)) {
          revng_assert(!(Call->getCalledFunction()->getName() == "newpc"));
          if (Call->getCalledFunction() == ExitTB) {
            revng_assert(Result == nullptr);
            Result = Call;
            return ExhaustQueueAndStop;
          }
        }
      }

      return Continue;
    }

    SuccessorsType successors(BasicBlock *BB) {
      SuccessorsType Successors;
      for (BasicBlock *Successor : make_range(succ_begin(BB), succ_end(BB)))
        if (JTM->isTranslatedBB(Successor))
          Successors.push_back(Successor);
      return Successors;
    }
  };

  Visitor V(ExitTB, this);
  V.run(Start);

  return V.Result;
}

StoreInst *JumpTargetManager::getPrevPCWrite(Instruction *TheInstruction) {
  // Look for the last write to the PC
  BasicBlock::iterator I(TheInstruction);
  BasicBlock::iterator Begin(TheInstruction->getParent()->begin());

  while (I != Begin) {
    I--;
    Instruction *Current = &*I;

    auto *Store = dyn_cast<StoreInst>(Current);
    if (Store != nullptr && Store->getPointerOperand() == PCReg)
      return Store;

    // If we meet a call to an helper, return nullptr
    // TODO: for now we just make calls to helpers, is this is OK even if we
    //       split the translated function in multiple functions?
    if (isa<CallInst>(Current))
      return nullptr;
  }

  // TODO: handle the following case:
  //          pc = x
  //          brcond ?, a, b
  //       a:
  //          pc = y
  //          br b
  //       b:
  //          exitTB
  // TODO: emit warning
  return nullptr;
}

// TODO: this is outdated and we should drop it, we now have OSRA and friends
/// \brief Tries to detect pc += register In general, we assume what we're
/// translating is code emitted by a compiler. This means that usually all the
/// possible jump targets are explicit jump to a constant or are stored
/// somewhere in memory (e.g.  jump tables and vtables). However, in certain
/// cases, mainly due to handcrafted assembly we can have a situation like the
/// following:
///
///     addne pc, pc, \\curbit, lsl #2
///
/// (taken from libgcc ARM's lib1funcs.S, specifically line 592 of
/// `libgcc/config/arm/lib1funcs.S` at commit
/// `f1717362de1e56fe1ffab540289d7d0c6ed48b20`)
///
/// This code basically jumps forward a number of instructions depending on a
/// run-time value. Therefore, without further analysis, potentially, all the
/// coming instructions are jump targets.
///
/// To workaround this issue we use a simple heuristics, which basically
/// consists in making all the coming instructions possible jump targets until
/// the next write to the PC. In the future, we could extend this until the end
/// of the function.
static bool isSumJump(StoreInst *PCWrite) {
  // * Follow the written value recursively
  //   * Is it a `load` or a `constant`? Fine. Don't proceed.
  //   * Is it an `and`? Enqueue the operands in the worklist.
  //   * Is it an `add`? Make all the coming instructions jump targets.
  //
  // This approach has a series of problems:
  //
  // * It doesn't work with delay slots. Delay slots are handled by libtinycode
  //   as follows:
  //
  //       jump lr
  //         store btarget, lr
  //       store 3, r0
  //         store 3, r0
  //         store btarget, pc
  //
  //   Clearly, if we don't follow the loads we miss the situation we're trying
  //   to handle.
  // * It is unclear how this would perform without EarlyCSE and SROA.
  std::queue<Value *> WorkList;
  WorkList.push(PCWrite->getValueOperand());

  while (!WorkList.empty()) {
    Value *V = WorkList.front();
    WorkList.pop();

    if (isa<Constant>(V) || isa<LoadInst>(V)) {
      // Fine
    } else if (auto *BinOp = dyn_cast<BinaryOperator>(V)) {
      switch (BinOp->getOpcode()) {
      case Instruction::Add:
      case Instruction::Or:
        return true;
      case Instruction::Shl:
      case Instruction::LShr:
      case Instruction::AShr:
      case Instruction::And:
        for (auto &Operand : BinOp->operands())
          if (!isa<Constant>(Operand.get()))
            WorkList.push(Operand.get());
        break;
      default:
        // TODO: emit warning
        return false;
      }
    } else {
      // TODO: emit warning
      return false;
    }
  }

  return false;
}

std::pair<uint64_t, uint64_t>
JumpTargetManager::getPC(Instruction *TheInstruction) const {
  CallInst *NewPCCall = nullptr;
  std::set<BasicBlock *> Visited;
  std::queue<BasicBlock::reverse_iterator> WorkList;
  if (TheInstruction->getIterator() == TheInstruction->getParent()->begin())
    WorkList.push(--TheInstruction->getParent()->rend());
  else
    WorkList.push(++TheInstruction->getReverseIterator());

  while (!WorkList.empty()) {
    auto I = WorkList.front();
    WorkList.pop();
    auto *BB = I->getParent();
    auto End = BB->rend();

    // Go through the instructions looking for calls to newpc
    for (; I != End; I++) {
      if (auto Marker = dyn_cast<CallInst>(&*I)) {
        // TODO: comparing strings is not very elegant
        auto *Callee = Marker->getCalledFunction();
        if (Callee != nullptr && Callee->getName() == "newpc") {

          // We found two distinct newpc leading to the requested instruction
          if (NewPCCall != nullptr)
            return { 0, 0 };

          NewPCCall = Marker;
          break;
        }
      }
    }

    // If we haven't find a newpc call yet, continue exploration backward
    if (NewPCCall == nullptr) {
      // If one of the predecessors is the dispatcher, don't explore any further
      for (BasicBlock *Predecessor : predecessors(BB)) {
        // Assert we didn't reach the almighty dispatcher
        revng_assert(!(NewPCCall == nullptr && Predecessor == Dispatcher));
        if (Predecessor == Dispatcher)
          continue;
      }

      for (BasicBlock *Predecessor : predecessors(BB)) {
        // Ignore already visited or empty BBs
        if (!Predecessor->empty()
            && Visited.find(Predecessor) == Visited.end()) {
          WorkList.push(Predecessor->rbegin());
          Visited.insert(Predecessor);
        }
      }
    }
  }

  // Couldn't find the current PC
  if (NewPCCall == nullptr)
    return { 0, 0 };

  uint64_t PC = getLimitedValue(NewPCCall->getArgOperand(0));
  uint64_t Size = getLimitedValue(NewPCCall->getArgOperand(1));
  revng_assert(Size != 0);
  return { PC, Size };
}

void JumpTargetManager::handleSumJump(Instruction *SumJump) {
  // Take the next PC
  uint64_t NextPC = getNextPC(SumJump);
  revng_assert(NextPC != 0);
  BasicBlock *BB = registerJT(NextPC, JTReason::SumJump);
  revng_assert(BB && !BB->empty());

  std::set<BasicBlock *> Visited;
  Visited.insert(Dispatcher);
  std::queue<BasicBlock *> WorkList;
  WorkList.push(BB);
  while (!WorkList.empty()) {
    BB = WorkList.front();
    Visited.insert(BB);
    WorkList.pop();

    BasicBlock::iterator I(BB->begin());
    BasicBlock::iterator End(BB->end());
    while (I != End) {
      // Is it a new PC marker?
      if (auto *Call = dyn_cast<CallInst>(&*I)) {
        Function *Callee = Call->getCalledFunction();
        // TODO: comparing strings is not very elegant
        if (Callee != nullptr && Callee->getName() == "newpc") {
          uint64_t PC = getLimitedValue(Call->getArgOperand(0));

          // If we've found a (direct or indirect) jump, stop
          if (PC != NextPC)
            return;

          // Split and update iterators to proceed
          BB = registerJT(PC, JTReason::SumJump);

          // Do we have a block?
          if (BB == nullptr)
            return;

          I = BB->begin();
          End = BB->end();

          // Updated the expectation for the next PC
          NextPC = PC + getLimitedValue(Call->getArgOperand(1));
        } else if (Call->getCalledFunction() == ExitTB) {
          // We've found an unparsed indirect jump
          return;
        }
      }

      // Proceed to next instruction
      I++;
    }

    // Inspect and enqueue successors
    for (BasicBlock *Successor : successors(BB))
      if (Visited.find(Successor) == Visited.end())
        WorkList.push(Successor);
  }
}

/// \brief Class to iterate over all the BBs associated to a translated PC
class BasicBlockVisitor {
public:
  BasicBlockVisitor(const SwitchInst *Dispatcher) :
    Dispatcher(Dispatcher),
    JumpTargetIndex(0),
    JumpTargetsCount(Dispatcher->getNumSuccessors()),
    DL(Dispatcher->getParent()->getParent()->getParent()->getDataLayout()) {}

  void enqueue(BasicBlock *BB) {
    if (Visited.count(BB))
      return;
    Visited.insert(BB);

    uint64_t PC = getPC(BB);
    if (PC == 0)
      SamePC.push(BB);
    else
      NewPC.push({ BB, PC });
  }

  // TODO: this function assumes 0 is not a valid PC
  std::pair<BasicBlock *, uint64_t> pop() {
    if (!SamePC.empty()) {
      auto Result = SamePC.front();
      SamePC.pop();
      return { Result, 0 };
    } else if (!NewPC.empty()) {
      auto Result = NewPC.front();
      NewPC.pop();
      return Result;
    } else if (JumpTargetIndex < JumpTargetsCount) {
      BasicBlock *BB = Dispatcher->getSuccessor(JumpTargetIndex);
      JumpTargetIndex++;
      return { BB, getPC(BB) };
    } else {
      return { nullptr, 0 };
    }
  }

private:
  // TODO: this function assumes 0 is not a valid PC
  uint64_t getPC(BasicBlock *BB) {
    if (!BB->empty()) {
      if (auto *Call = dyn_cast<CallInst>(&*BB->begin())) {
        Function *Callee = Call->getCalledFunction();
        // TODO: comparing with "newpc" string is sad
        if (Callee != nullptr && Callee->getName() == "newpc") {
          Constant *PCOperand = cast<Constant>(Call->getArgOperand(0));
          return getZExtValue(PCOperand, DL);
        }
      }
    }

    return 0;
  }

private:
  const SwitchInst *Dispatcher;
  unsigned JumpTargetIndex;
  unsigned JumpTargetsCount;
  const DataLayout &DL;
  std::set<BasicBlock *> Visited;
  std::queue<BasicBlock *> SamePC;
  std::queue<std::pair<BasicBlock *, uint64_t>> NewPC;
};

void JumpTargetManager::translateIndirectJumps() {
  if (ExitTB->use_empty())
    return;

  legacy::PassManager AnalysisPM;
  AnalysisPM.add(new TranslateDirectBranchesPass(this));
  AnalysisPM.run(TheModule);

  auto I = ExitTB->use_begin();

  while (I != ExitTB->use_end()) {
    Use &ExitTBUse = *I++;
     
    if (auto *Call = dyn_cast<CallInst>(ExitTBUse.getUser())) {
      if (Call->getCalledFunction() == ExitTB) {
        // Look for the last write to the PC
        StoreInst *PCWrite = getPrevPCWrite(Call);
        if (PCWrite != nullptr) {
        //   revng_assert(!isa<ConstantInt>(PCWrite->getValueOperand()),
        //               "Direct jumps should not be handled here");
        }

        if (PCWrite != nullptr && not NoOSRA && isSumJump(PCWrite))
          handleSumJump(PCWrite);

        if (getLimitedValue(Call->getArgOperand(0)) == 0) {
          exitTBCleanup(Call);
          BranchInst::Create(Dispatcher, Call);
        }

       // BasicBlock::iterator Begin1(Call->getParent()->begin());
       // BasicBlock * Begin1(Call->getParent());
       // errs()<<*Begin1<<"\n";

        Call->eraseFromParent();
      }
    }
  }

  revng_assert(ExitTB->use_empty());
  ExitTB->eraseFromParent();
  ExitTB = nullptr;
}

JumpTargetManager::BlockWithAddress JumpTargetManager::peek() {
  harvest();

  // Purge all the partial translations we know might be wrong
  for (BasicBlock *BB : ToPurge)
    purgeTranslation(BB);
  ToPurge.clear();

  if (Unexplored.empty())
    return NoMoreTargets;
  else {
    BlockWithAddress Result = Unexplored.back();
    Unexplored.pop_back();
    return Result;
  }
}

void JumpTargetManager::unvisit(BasicBlock *BB) {
  if (Visited.find(BB) != Visited.end()) {
    std::vector<BasicBlock *> WorkList;
    WorkList.push_back(BB);

    while (!WorkList.empty()) {
      BasicBlock *Current = WorkList.back();
      WorkList.pop_back();

      Visited.erase(Current);

      for (BasicBlock *Successor : successors(BB)) {
        if (Visited.find(Successor) != Visited.end() && !Successor->empty()) {
          auto *Call = dyn_cast<CallInst>(&*Successor->begin());
          if (Call == nullptr
              || Call->getCalledFunction()->getName() != "newpc") {
            WorkList.push_back(Successor);
          }
        }
      }
    }
  }
}

BasicBlock *JumpTargetManager::getBlockAt(uint64_t PC) {
  auto TargetIt = JumpTargets.find(PC);
  revng_assert(TargetIt != JumpTargets.end());
  return TargetIt->second.head();
}

void JumpTargetManager::purgeTranslation(BasicBlock *Start) {
  OnceQueue<BasicBlock *> Queue;
  Queue.insert(Start);

  // Collect all the descendats, except if we meet a jump target
  while (!Queue.empty()) {
    BasicBlock *BB = Queue.pop();
    for (BasicBlock *Successor : successors(BB)) {
      if (isTranslatedBB(Successor) && !isJumpTarget(Successor)
          && !hasPredecessor(Successor, Dispatcher)) {
        Queue.insert(Successor);
      }
    }
  }

  // Erase all the visited basic blocks
  std::set<BasicBlock *> Visited = Queue.visited();

  // Build a subgraph, so that we can visit it in post order, and purge the
  // content of each basic block
  SubGraph<BasicBlock *> TranslatedBBs(Start, Visited);
  for (auto *Node : post_order(TranslatedBBs)) {
    BasicBlock *BB = Node->get();
    while (!BB->empty())
      eraseInstruction(&*(--BB->end()));
  }

  // Remove Start, since we want to keep it (even if empty)
  Visited.erase(Start);

  errs()<<Start->getName()<<" ---------------------------\n";

  for (BasicBlock *BB : Visited) {
    // We might have some predecessorless basic blocks jumping to us, purge them
    // TODO: why this?
    while (pred_begin(BB) != pred_end(BB)) {
      BasicBlock *Predecessor = *pred_begin(BB);
      revng_assert(pred_empty(Predecessor));
      errs()<<Predecessor->getName()<<" ---------------------------\n";
      Predecessor->eraseFromParent();
    }

    revng_assert(BB->use_empty());
    errs()<<BB->getName()<<" ---------------------------\n";
    BB->eraseFromParent();
  }
}

BasicBlock * JumpTargetManager::obtainJTBB(uint64_t PC, JTReason::Values Reason){
 
  BlockMap::iterator TargetIt = JumpTargets.find(PC);
  if (TargetIt != JumpTargets.end()) {
    // Case 1: there's already a BasicBlock for that address, return it
    BasicBlock *BB = TargetIt->second.head();
    TargetIt->second.setReason(Reason);
    
    unvisit(BB);
    return BB;
  }
  return nullptr;
}

// TODO: register Reason
BasicBlock *
JumpTargetManager::registerJT(uint64_t PC, JTReason::Values Reason) {
  haveBB = 0;
  if (!isExecutableAddress(PC) || !isInstructionAligned(PC))
    return nullptr;

  revng_log(RegisterJTLog,
            "Registering bb." << nameForAddress(PC) << " for "
                              << JTReason::getName(Reason));

  // Do we already have a BasicBlock for this PC?
  BlockMap::iterator TargetIt = JumpTargets.find(PC);
  if (TargetIt != JumpTargets.end()) {
    // Case 1: there's already a BasicBlock for that address, return it
    BasicBlock *BB = TargetIt->second.head();
    TargetIt->second.setReason(Reason);
    
    haveBB = 1;
    
    unvisit(BB);
    return BB;
  }

  // Did we already meet this PC (i.e. do we know what's the associated
  // instruction)?
  BasicBlock *NewBlock = nullptr;
  InstructionMap::iterator InstrIt = OriginalInstructionAddresses.find(PC);
  if (InstrIt != OriginalInstructionAddresses.end()) {
    // Case 2: the address has already been met, but needs to be promoted to
    //         BasicBlock level.
    Instruction *I = InstrIt->second;
    BasicBlock *ContainingBlock = I->getParent();
    if (isFirst(I)) {
      NewBlock = ContainingBlock;
    } else {
      revng_assert(I != nullptr && I->getIterator() != ContainingBlock->end());
  
      std::map<llvm::BasicBlock *,uint32_t>::iterator TargetG = AllGadget.find(ContainingBlock);     
      std::map<llvm::BasicBlock *,uint32_t>::iterator TargetS = AllStaticGadget.find(ContainingBlock); 
      if(TargetG != AllGadget.end() or TargetS!=AllStaticGadget.end()){
        for(auto &g : assign_gadge){
          if(g.second.operation_block == ContainingBlock or
             g.second.static_addr_block == ContainingBlock){
            g.second.operation_block  = nullptr;
            g.second.static_addr_block = nullptr;
            g.second.global_I = nullptr;
            g.second.static_global_I = nullptr;
          }
        }
        AllGadget.erase(ContainingBlock);
        AllStaticGadget.erase(ContainingBlock);
      }
      NewBlock = ContainingBlock->splitBasicBlock(I);
    }

    // Register the basic block and all of its descendants to be purged so that
    // we can retranslate this PC
    // TODO: this might create a problem if QEMU generates control flow that
    //       crosses an instruction boundary
    ToPurge.insert(NewBlock);

    unvisit(NewBlock);
  } else {
    // Case 3: the address has never been met, create a temporary one, register
    // it for future exploration and return it
    NewBlock = BasicBlock::Create(Context, "", TheFunction);
  }

  Unexplored.push_back(BlockWithAddress(PC, NewBlock));

  std::stringstream Name;
  Name << "bb." << nameForAddress(PC);
  NewBlock->setName(Name.str());

  // Create a case for the address associated to the new block
  auto *PCRegType = PCReg->getType();
  auto *SwitchType = cast<IntegerType>(PCRegType->getPointerElementType());
  auto a = ConstantInt::get(SwitchType, PC);
  DispatcherSwitch->addCase(a, NewBlock);

  // Associate the PC with the chosen basic block
  JumpTargets[PC] = JumpTarget(NewBlock, Reason);
  return NewBlock;
}

void JumpTargetManager::registerReadRange(uint64_t Address, uint64_t Size) {
  using interval = boost::icl::interval<uint64_t>;
  ReadIntervalSet += interval::right_open(Address, Address + Size);
}

// TODO: instead of a gigantic switch case we could map the original memory area
//       and write the address of the translated basic block at the jump target
// If this function looks weird it's because it has been designed to be able
// to create the dispatcher in the "root" function or in a standalone function
void JumpTargetManager::createDispatcher(Function *OutputFunction,
                                         Value *SwitchOnPtr) {
  IRBuilder<> Builder(Context);
  QuickMetadata QMD(Context);

  // Create the first block of the dispatcher
  BasicBlock *Entry = BasicBlock::Create(Context,
                                         "dispatcher.entry",
                                         OutputFunction);

  // The default case of the switch statement it's an unhandled cases
  DispatcherFail = BasicBlock::Create(Context,
                                      "dispatcher.default",
                                      OutputFunction);
  Builder.SetInsertPoint(DispatcherFail);

  Module *TheModule = TheFunction->getParent();
  
  Value *PC = Builder.CreateLoad(pcReg()); 
  //Value *ELFName = Builder.CreateLoad(elf_name);
  Value *ELFName = dyn_cast<Value>(elf_name);
  auto *UnknownPCTy = FunctionType::get(Type::getVoidTy(Context), {PC->getType(),ELFName->getType()}, false);
  Constant *UnknownPC = TheModule->getOrInsertFunction("unknownPC",
                                                       UnknownPCTy);
  Builder.CreateCall(cast<Function>(UnknownPC), {PC,ELFName});
  auto *FailUnreachable = Builder.CreateUnreachable();
  FailUnreachable->setMetadata("revng.block.type",
                               QMD.tuple((uint32_t) DispatcherFailureBlock));

  // Switch on the first argument of the function
  Builder.SetInsertPoint(Entry);
  Value *SwitchOn = Builder.CreateLoad(SwitchOnPtr);
  SwitchInst *Switch = Builder.CreateSwitch(SwitchOn, DispatcherFail);
  // The switch is the terminator of the dispatcher basic block
  Switch->setMetadata("revng.block.type",
                      QMD.tuple((uint32_t) DispatcherBlock));

  Dispatcher = Entry;
  DispatcherSwitch = Switch;
  NoReturn.setDispatcher(Dispatcher);

  // Create basic blocks to handle jumps to any PC and to a PC we didn't expect
  AnyPC = BasicBlock::Create(Context, "anypc", OutputFunction);
  UnexpectedPC = BasicBlock::Create(Context, "unexpectedpc", OutputFunction);

  setCFGForm(CFGForm::SemanticPreservingCFG);
}

static void purge(BasicBlock *BB) {
  // Allow up to a single instruction in the basic block
  if (!BB->empty())
    BB->begin()->eraseFromParent();
  revng_assert(BB->empty());
}

std::set<BasicBlock *> JumpTargetManager::computeUnreachable() {
  ReversePostOrderTraversal<BasicBlock *> RPOT(&TheFunction->getEntryBlock());
  std::set<BasicBlock *> Reachable;
  for (BasicBlock *BB : RPOT)
    Reachable.insert(BB);

  // TODO: why is isTranslatedBB(&BB) necessary?
  std::set<BasicBlock *> Unreachable;
  for (BasicBlock &BB : *TheFunction)
    if (Reachable.count(&BB) == 0 and isTranslatedBB(&BB))
      Unreachable.insert(&BB);

  return Unreachable;
}

void JumpTargetManager::setCFGForm(CFGForm::Values NewForm) {
  revng_assert(CurrentCFGForm != NewForm);
  revng_assert(NewForm != CFGForm::UnknownFormCFG);

  std::set<BasicBlock *> Unreachable;

  CFGForm::Values OldForm = CurrentCFGForm;
  CurrentCFGForm = NewForm;

  switch (NewForm) {
  case CFGForm::SemanticPreservingCFG:
    purge(AnyPC);
    BranchInst::Create(dispatcher(), AnyPC);
    // TODO: Here we should have an hard fail, since it's the situation in
    //       which we expected to know where execution could go but we made a
    //       mistake.
    purge(UnexpectedPC);
    BranchInst::Create(dispatcher(), UnexpectedPC);
    break;

  case CFGForm::RecoveredOnlyCFG:
  case CFGForm::NoFunctionCallsCFG:
    purge(AnyPC);
    new UnreachableInst(Context, AnyPC);
    purge(UnexpectedPC);
    new UnreachableInst(Context, UnexpectedPC);
    break;

  default:
    revng_abort("Not implemented yet");
  }

  QuickMetadata QMD(Context);
  AnyPC->getTerminator()->setMetadata("revng.block.type",
                                      QMD.tuple((uint32_t) AnyPCBlock));
  TerminatorInst *UnexpectedPCJump = UnexpectedPC->getTerminator();
  UnexpectedPCJump->setMetadata("revng.block.type",
                                QMD.tuple((uint32_t) UnexpectedPCBlock));

  // If we're entering or leaving the NoFunctionCallsCFG form, update all the
  // branch instruction forming a function call
  if (NewForm == CFGForm::NoFunctionCallsCFG
      || OldForm == CFGForm::NoFunctionCallsCFG) {
    if (auto *FunctionCall = TheModule.getFunction("function_call")) {
      for (User *U : FunctionCall->users()) {
        auto *Call = cast<CallInst>(U);

        // Ignore indirect calls
        // TODO: why this is needed is unclear
        if (isa<ConstantPointerNull>(Call->getArgOperand(0)))
          continue;

        auto *Terminator = cast<TerminatorInst>(nextNonMarker(Call));
        revng_assert(Terminator->getNumSuccessors() == 1);

        // Get the correct argument, the first is the callee, the second the
        // return basic block
        int OperandIndex = NewForm == CFGForm::NoFunctionCallsCFG ? 1 : 0;
        Value *Op = Call->getArgOperand(OperandIndex);
        BasicBlock *NewSuccessor = cast<BlockAddress>(Op)->getBasicBlock();
        Terminator->setSuccessor(0, NewSuccessor);
      }
    }
  }

  rebuildDispatcher();

  if (Verify.isEnabled()) {
    Unreachable = computeUnreachable();
    if (Unreachable.size() != 0) {
      Verify << "The following basic blocks are unreachable after setCFGForm("
             << CFGForm::getName(NewForm) << "):\n";
      for (BasicBlock *BB : Unreachable) {
        Verify << "  " << getName(BB) << " (predecessors:";
        for (BasicBlock *Predecessor : make_range(pred_begin(BB), pred_end(BB)))
          Verify << " " << getName(Predecessor);

        if (uint64_t PC = getBasicBlockPC(BB)) {
          auto It = JumpTargets.find(PC);
          if (It != JumpTargets.end()) {
            Verify << ", reasons:";
            for (const char *Reason : It->second.getReasonNames())
              Verify << " " << Reason;
          }
        }

        Verify << ")\n";
      }
      revng_abort();
    }
  }
}

void JumpTargetManager::rebuildDispatcher() {
  // Remove all cases
  unsigned NumCases = DispatcherSwitch->getNumCases();
  while (NumCases-- > 0)
    DispatcherSwitch->removeCase(DispatcherSwitch->case_begin());

  auto *PCRegType = PCReg->getType()->getPointerElementType();
  auto *SwitchType = cast<IntegerType>(PCRegType);

  // Add all the jump targets if we're using the SemanticPreservingCFG, or
  // only those with no predecessors otherwise
  for (auto &P : JumpTargets) {
    uint64_t PC = P.first;
    BasicBlock *BB = P.second.head();
    if (CurrentCFGForm == CFGForm::SemanticPreservingCFG
        || !hasPredecessors(BB))
      DispatcherSwitch->addCase(ConstantInt::get(SwitchType, PC), BB);
  }

  //
  // Make sure every generated basic block is reachable
  //
  if (CurrentCFGForm != CFGForm::SemanticPreservingCFG) {
    // Compute the set of reachable jump targets
    OnceQueue<BasicBlock *> WorkList;
    for (BasicBlock *BB : DispatcherSwitch->successors())
      WorkList.insert(BB);

    while (not WorkList.empty()) {
      BasicBlock *BB = WorkList.pop();
      for (BasicBlock *Successor : make_range(succ_begin(BB), succ_end(BB)))
        WorkList.insert(Successor);
    }

    std::set<BasicBlock *> Reachable = WorkList.visited();

    // Identify all the unreachable jump targets
    for (auto &P : JumpTargets) {
      uint64_t PC = P.first;
      const JumpTarget &JT = P.second;
      BasicBlock *BB = JT.head();

      // Add to the switch all the unreachable jump targets whose reason is not
      // just direct jump
      if (Reachable.count(BB) == 0
          and not JT.isOnlyReason(JTReason::DirectJump)) {
        DispatcherSwitch->addCase(ConstantInt::get(SwitchType, PC), BB);
      }
    }
  }
}

bool JumpTargetManager::hasPredecessors(BasicBlock *BB) const {
  for (BasicBlock *Pred : predecessors(BB))
    if (isTranslatedBB(Pred))
      return true;
  return false;
}

// Harvesting proceeds trying to avoid to run expensive analyses if not strictly
// necessary, OSRA in particular. To do this we keep in mind two aspects: do we
// have new basic blocks to visit? If so, we avoid any further anyalysis and
// give back control to the translator. If not, we proceed with other analyses
// until we either find a new basic block to translate. If we can't find a new
// block to translate we proceed as long as we are able to create new edges on
// the CFG (not considering the dispatcher).
void JumpTargetManager::harvest() {

//  if (empty()) {
//    for (uint64_t PC : SimpleLiterals)
//      registerJT(PC, JTReason::SimpleLiteral);
//    SimpleLiterals.clear();
//  }
//
//  if (empty()) {
//    // Purge all the generated basic blocks without predecessors
//    std::vector<BasicBlock *> ToDelete;
//    for (BasicBlock &BB : *TheFunction) {
//      if (isTranslatedBB(&BB) and &BB != &TheFunction->getEntryBlock()
//          and pred_begin(&BB) == pred_end(&BB)) {
//        revng_assert(getBasicBlockPC(&BB) == 0);
//        ToDelete.push_back(&BB);
//      }
//    }
//    for (BasicBlock *BB : ToDelete)
//      BB->eraseFromParent();
//
//    // TODO: move me to a commit function
//    // Update the third argument of newpc calls (isJT, i.e., is this instruction
//    // a jump target?)
//    IRBuilder<> Builder(Context);
//    Function *NewPCFunction = TheModule.getFunction("newpc");
//    if (NewPCFunction != nullptr) {
//      for (User *U : NewPCFunction->users()) {
//        auto *Call = cast<CallInst>(U);
//        if (Call->getParent() != nullptr) {
//          // Report the instruction on the coverage CSV
//          using CI = ConstantInt;
//          uint64_t PC = (cast<CI>(Call->getArgOperand(0)))->getLimitedValue();
//
//          bool IsJT = isJumpTarget(PC);
//          Call->setArgOperand(2, Builder.getInt32(static_cast<uint32_t>(IsJT)));
//        }
//      }
//    }
//
//    if (Verify.isEnabled())
//      revng_assert(not verifyModule(TheModule, &dbgs()));
//
//    revng_log(JTCountLog, "Harvesting: SROA, ConstProp, EarlyCSE and SET");
//
//    legacy::FunctionPassManager OptimizingPM(&TheModule);
//    OptimizingPM.add(createSROAPass());
//    OptimizingPM.add(createConstantPropagationPass());
//    OptimizingPM.add(createEarlyCSEPass());
//    OptimizingPM.run(*TheFunction);
//
//    legacy::PassManager PreliminaryBranchesPM;
//    PreliminaryBranchesPM.add(new TranslateDirectBranchesPass(this));
//    PreliminaryBranchesPM.run(TheModule);
//
//    // TODO: eventually, `setCFGForm` should be replaced by using a CustomCFG
//    // To improve the quality of our analysis, keep in the CFG only the edges we
//    // where able to recover (e.g., no jumps to the dispatcher)
//    setCFGForm(CFGForm::RecoveredOnlyCFG);
//
//    NewBranches = 0;
//    legacy::PassManager AnalysisPM;
//    AnalysisPM.add(new SETPass(this, false, &Visited));
//    AnalysisPM.add(new TranslateDirectBranchesPass(this));
//    AnalysisPM.run(TheModule);
//
//    // Restore the CFG
//    setCFGForm(CFGForm::SemanticPreservingCFG);
//
//    revng_log(JTCountLog,
//              std::dec << Unexplored.size() << " new jump targets and "
//                       << NewBranches << " new branches were found");
//  }

//  if (not NoOSRA && empty()) {
//    if (Verify.isEnabled())
//      revng_assert(not verifyModule(TheModule, &dbgs()));
//
//    NoReturn.registerSyscalls(TheFunction);
//
//    do {
//
//      revng_log(JTCountLog,
//                "Harvesting: reset Visited, "
//                  << (NewBranches > 0 ? "SROA, ConstProp, EarlyCSE, " : "")
//                  << "SET + OSRA");
//
//      // TODO: decide what to do with Visited
//      Visited.clear();
//      if (NewBranches > 0) {
//        legacy::FunctionPassManager OptimizingPM(&TheModule);
//        OptimizingPM.add(createSROAPass());
//        OptimizingPM.add(createConstantPropagationPass());
//        OptimizingPM.add(createEarlyCSEPass());
//        OptimizingPM.run(*TheFunction);
//      }
//
//      legacy::PassManager FunctionCallPM;
//      FunctionCallPM.add(new FunctionCallIdentification());
//      FunctionCallPM.run(TheModule);
//
//      createJTReasonMD();
//
//      setCFGForm(CFGForm::RecoveredOnlyCFG);
//
//      NewBranches = 0;
//      legacy::PassManager AnalysisPM;
//      AnalysisPM.add(new SETPass(this, true, &Visited));
//      AnalysisPM.add(new TranslateDirectBranchesPass(this));
//      AnalysisPM.run(TheModule);
//
//      // Restore the CFG
//      setCFGForm(CFGForm::SemanticPreservingCFG);
//
//      revng_log(JTCountLog,
//                std::dec << Unexplored.size() << " new jump targets and "
//                         << NewBranches << " new branches were found");
//
//    } while (empty() && NewBranches > 0);
//  }
  
 // setCFGForm(CFGForm::RecoveredOnlyCFG);

  if (empty()) {
    revng_log(JTCountLog, "We're done looking for jump targets");
  }
}

void JumpTargetManager::pushpartCFGStack(llvm::BasicBlock *dest, 
		                         uint64_t DAddr,
					 llvm::BasicBlock *src,
					 uint64_t SAddr){
  partCFG.push_back(std::make_tuple(dest,DAddr,src,SAddr));
  std::get<0>(nodepCFG) = dest;
  std::get<1>(nodepCFG) = src;
}

void JumpTargetManager::searchpartCFG(std::map<llvm::BasicBlock *,llvm::BasicBlock *> &DONE){
  //Match source BB, to search start entry of one path.
  llvm::Function::iterator it(std::get<1>(nodepCFG));
  llvm::Function::iterator begin(it->getParent()->begin());
  
  for(; it!=begin; it-- ){
    auto bb = dyn_cast<llvm::BasicBlock>(it);
    for(auto p : partCFG){
      if((bb - std::get<0>(p)) == 0){
	if(DONE.find(bb) != DONE.end())
	    break;
        std::get<0>(nodepCFG) = std::get<0>(p);
        std::get<1>(nodepCFG) = std::get<2>(p);
	DONE[std::get<0>(p)] = std::get<2>(p);
        return;
      }
    }
  } 
  std::get<0>(nodepCFG) = nullptr;
  std::get<1>(nodepCFG) = nullptr; 
}

uint32_t JumpTargetManager::belongToUBlock(llvm::BasicBlock *block){
  llvm::StringRef str = block->getName();
  LLVM_NODISCARD size_t nPos1 = llvm::StringRef::npos;
  LLVM_NODISCARD size_t nPos2 = llvm::StringRef::npos;
  llvm::StringRef substr = "";
  nPos1 = str.find_last_of(".");
  nPos2 = str.find_last_of(".",nPos1-1);
  if(nPos1>nPos2){
    substr = str.substr(nPos2 + 1, nPos1 - nPos2 - 1);
  }
  else{
    substr = str.substr(nPos1+1,str.size()-nPos1-1);
  }
  
  // TODO: Get user defined code range.
  llvm::StringRef UserCodeName = "main";
  if(substr.equals(UserCodeName))
    return 1;

  return 0;
}

bool JumpTargetManager::isDataSegmAddr(uint64_t PC){
  return ptc.is_image_addr(PC);  
}

bool JumpTargetManager::isELFDataSegmAddr(uint64_t PC){
  bool flag = false;
  if(ro_StartAddr<=PC and PC<ro_EndAddr)
    flag = true;
  return (ptc.is_image_addr(PC) or flag);  
}

std::pair<bool, uint32_t> JumpTargetManager::islegalAddr(llvm::Value *v){
  uint64_t va = 0;
  StringRef Iargs = v->getName();
  uint32_t registerName = 0;
 
  auto op = StrToInt(Iargs.data());
  //errs()<<op<<"+++\n"; 
  switch(op){
    case RAX:
      va = ptc.regs[R_EAX];
      registerName = RAX;
      errs()<<va<<" :eax\n";
      if(!isDataSegmAddr(va))
        return std::make_pair(0,RAX);
    break;
    case RCX:
      va = ptc.regs[R_ECX];
      registerName = RCX;
      //errs()<<ptc.regs[R_ECX]<<" ++\n";
      if(!isDataSegmAddr(va))
        return std::make_pair(0,RCX);
    break;
    case RDX:
      va = ptc.regs[R_EDX];
      registerName = RDX;
      errs()<<ptc.regs[R_EDX]<<" ++\n";
      if(!isDataSegmAddr(va))
        return std::make_pair(0,RDX);
    break;
    case RBX:
      va = ptc.regs[R_EBX];
      registerName = RBX;
      //errs()<<ptc.regs[R_EBX]<<" ++\n";
      if(!isDataSegmAddr(va))
        return std::make_pair(0,RBX);
    break;
    case RSP:
      va = ptc.regs[R_ESP];
      registerName = RSP;
      //errs()<<ptc.regs[R_ESP]<<" ++\n";
      if(!isDataSegmAddr(va)){
        errs()<<"RSP shouldn't be illegal address!\n";
	return std::make_pair(0,RSP);
      }
    break;
    case RBP:
      va = ptc.regs[R_EBP];
      registerName = RBP;
      errs()<<ptc.regs[R_EBP]<<" ++\n";
      if(!isDataSegmAddr(va))
        return std::make_pair(0,RBP);
    break;
    case RSI:
      va = ptc.regs[R_ESI];
      registerName = RSI;
      //errs()<<ptc.regs[R_ESI]<<" ++\n";
      if(!isDataSegmAddr(va))
        return std::make_pair(0,RSI);
    break;
    case RDI:
      va = ptc.regs[R_EDI];
      registerName = RDI;
      errs()<<ptc.regs[R_EDI]<<" ++\n";
      if(!isDataSegmAddr(va))
        return std::make_pair(0,RDI);
    break;
    case R8:
      va = ptc.regs[R_8];
      registerName = R8;
      if(!isDataSegmAddr(va))
	return std::make_pair(0,R8);
    break;
    case R9:
      va = ptc.regs[R_9];
      registerName = R9;
      if(!isDataSegmAddr(va))
        return std::make_pair(0,R9);
    break;
    case R10:
      va = ptc.regs[R_10];
      registerName = R10;
      if(!isDataSegmAddr(va))
	return std::make_pair(0,R10);
    break;
    case R11:
      va = ptc.regs[R_11];
      registerName = R11;
      if(!isDataSegmAddr(va))
	return std::make_pair(0,R11);
    break;
    case R12:
      va = ptc.regs[R_12];
      registerName = R12;
      if(!isDataSegmAddr(va))
        return std::make_pair(0,R12);
    break;
    case R13:
      va = ptc.regs[R_13];
      registerName = R13;
      if(!isDataSegmAddr(va))
        return std::make_pair(0,R13);
    break;
    case R14:
      va = ptc.regs[R_14];
      registerName = R14;
      if(!isDataSegmAddr(va))
        return std::make_pair(0,R14);
    break;
    case R15:
      va = ptc.regs[R_15];
      registerName = R15;
      if(!isDataSegmAddr(va))
        return std::make_pair(0,R15);
    break;
    default:
      errs()<<"No match register arguments! \n";
  }
  return std::make_pair(1,registerName);
}

JumpTargetManager::LastAssignmentResultWithInst 
JumpTargetManager:: getLastAssignment(llvm::Value *v, 
                                      llvm::User *userInst, 
                                      llvm::BasicBlock *currentBB,
				      TrackbackMode TrackType,
				      uint32_t &NUMOFCONST){
  if(dyn_cast<ConstantInt>(v)){
    return std::make_pair(ConstantValueAssign,nullptr);
  }
  switch(TrackType){
    case FullMode:
    case CrashMode:{
        if(v->getName().equals("rsp"))
          return std::make_pair(ConstantValueAssign,nullptr);       
    }break;
    case InterprocessMode:{
        if(v->getName().equals("rsp")){    
	  NUMOFCONST--;
	  if(NUMOFCONST==0)
	    return std::make_pair(ConstantValueAssign,nullptr);
	}

    }break;
    case TestMode:{
        if(v->getName().equals("rsp")){    
	  NUMOFCONST--;
	  if(NUMOFCONST==0)
	    return std::make_pair(ConstantValueAssign,nullptr);
	}
    }
    case JumpTableMode:{
        auto op = StrToInt(v->getName().data());
        switch(op){ 
	  case RAX:
	  case RBX:
	  case RCX:
	  case RDX:
	  case RSI:
	  case RDI:
          case RSP:
          case RBP:
	  case R8:
	  case R9:
	  case R10:
	  case R11:
	  case R12:
	  case R13:
	  case R14:
          case R15:
	  {
	    if(NUMOFCONST==0) 
	        return std::make_pair(ConstantValueAssign,nullptr);
            NUMOFCONST--;
	  }break;
	}
    }break;
    case RangeMode:{
        auto op = StrToInt(v->getName().data());
        switch(op){ 
	  case RAX:
	  case RBX:
	  case RCX:
	  case RDX:
	  case RSI:
	  case RDI:
          case RSP:
	  case R8:
	  case R9:
	  case R10:
	  case R11:
	  case R12:
	  case R13:
	  case R14:
          case R15:
	  {
	    return std::make_pair(ConstantValueAssign,nullptr);
	  }break;
	}
    }break;
  } 


  errs()<<currentBB->getName()<<"               **************************************\n\n ";
  bool bar = 0;
  std::vector<llvm::Instruction *> vDefUse;
  for(User *vu : v->users()){
    //errs()<<*vu<<"\n";
    if((vu - userInst) == 0)
    	bar = 1;
    auto *vui = dyn_cast<Instruction>(vu);
    if(bar && ((vui->getParent() - currentBB) == 0)){
    	//errs()<<*vu<<" userInst****\n";
        vDefUse.push_back(vui);
    }
    /*
    if(bar && ((vui->getParent() - thisBlock) != 0))
    	break;
    */
  }
  if(vDefUse.empty()){
    bar = 0;
    std::vector<Instruction *> UserOFbv;
    for(auto &Ib : make_range(currentBB->begin(),currentBB->end())){
      for(auto &Ub : Ib.operands()){
        Value *Vb = Ub.get();
        if((v - Vb)==0){
          UserOFbv.push_back(dyn_cast<Instruction>(&Ib));
          break;
        }
      }
    }
    for(auto vuInst : make_range(UserOFbv.rbegin(),UserOFbv.rend())){
      if((dyn_cast<User>(vuInst) - userInst) == 0)
        bar = 1;
      if(bar)
        vDefUse.push_back(vuInst);
    }
    
    if(vDefUse.empty())
      vDefUse = UserOFbv; 
  }

  auto def = dyn_cast<Instruction>(v);
//  if(vDefUse.size() == 1){
//        //vDefUse[0]->getOpcode() == llvm::Instruction::Store
//        if(def){
//          errs()<<*def<<"return def instruction\n";
//          return CurrentBlockValueDef;
//        }
//        else{
//          errs()<<" explort next BasicBlock! return v\n";
//          return NextBlockOperating;
//        }
//  }
  if(vDefUse.size() >= 1){
  	for(auto last : vDefUse){
          /* As user, only store is assigned to Value */ 
          switch(last->getOpcode())
          {
            case llvm::Instruction::Store:{
              auto lastS = dyn_cast<llvm::StoreInst>(last);
              if((lastS->getPointerOperand() - v) == 0){
                  errs()<<*last<<"\n^--last assignment\n";
                  return std::make_pair(CurrentBlockLastAssign,last);
              }
              break;
            } 
            case llvm::Instruction::Load:
            case llvm::Instruction::Select:
            case llvm::Instruction::ICmp:
            case llvm::Instruction::IntToPtr:
            case llvm::Instruction::Add:
            case llvm::Instruction::Sub:
            case llvm::Instruction::And:
            case llvm::Instruction::ZExt:
	    case llvm::Instruction::SExt:
            case llvm::Instruction::Trunc:
	    case llvm::Instruction::Shl:
	    case llvm::Instruction::LShr:
	    case llvm::Instruction::AShr:
            case llvm::Instruction::Or:
	    case llvm::Instruction::Xor:
	    case llvm::Instruction::Br:
	    case llvm::Instruction::Call:
	    case llvm::Instruction::Mul:			  
              continue;
            break;
            default:
              errs()<<"Unkonw instruction: "<<*last<<"\n";
              revng_abort("Unkonw instruction!");
            break;
          }
        }
        if(def){
            errs()<<*def<<"\n^--many or one user, return def instruction\n";
            return std::make_pair(CurrentBlockValueDef,def);
        }
        else{
            errs()<<"--no assignment, to explort next BasicBlock of Value's users\n";
            return std::make_pair(NextBlockOperating,nullptr);
        }
  }
 
  return std::make_pair(UnknowResult,nullptr);   
}

void JumpTargetManager::harvestBlockPCs(std::vector<uint64_t> &BlockPCs){
  if(BlockPCs.empty())
    return;
  int i = 0;
  for(auto pc : BlockPCs){
    if(!haveTranslatedPC(pc, 0) && !isIllegalStaticAddr(pc))    
      StaticAddrs[pc] = false;
    i++;
    if(i>=3)
      break;
  }
}


void JumpTargetManager::registerJumpTable(llvm::BasicBlock *thisBlock, uint64_t thisAddr, int64_t base, int64_t offset){
  if(isExecutableAddress((uint64_t)base))
    return;
    //revng_abort();
  if(!isELFDataSegmAddr((uint64_t)base))
    return;
  JumpTableBase[base] = offset;
  for(uint64_t n = 0;;n++){
    uint64_t addr = (uint64_t)(base + (n << offset));
    uint64_t niube = addr;
    while(isELFDataSegmAddr(addr)){
      auto pre = addr;
      addr = *((uint64_t *)addr);
      if(pre==addr or addr==niube)
          break;
    }

    if(addr==0)
        continue;
    if(isExecutableAddress(addr)){
        
        auto Path = "JumpTable.log";
        std::ofstream JTAddr;
        JTAddr.open(Path,std::ofstream::out | std::ofstream::app);
        JTAddr <<"---------> "<< std::hex << addr <<"\n";
        JTAddr.close();

        harvestBTBasicBlock(thisBlock,thisAddr,addr);
    }
    else
      break;
  }
}

void JumpTargetManager::harvestJumpTableAddr(llvm::BasicBlock *thisBlock, uint64_t thisAddr){
  BasicBlock::iterator begin(thisBlock->begin());
  BasicBlock::iterator end(thisBlock->end());

  auto I = begin;
  uint32_t isJumpTable = 0;
  uint64_t PC = 0;
  llvm::Instruction *shl = nullptr;
  llvm::Instruction *add = nullptr;
  int64_t base = 0;
  int64_t offset = 0;

  for(;I!=end;I++){
    auto op = I->getOpcode();
    if(op == Instruction::Call){
      auto call = dyn_cast<CallInst>(&*I);
      auto callee = call->getCalledFunction();
      if(callee != nullptr and callee->getName() == "newpc"){
        PC = getLimitedValue(call->getArgOperand(0));
        isJumpTable  = 0; 
        shl = nullptr;
        if(offset){
          registerJumpTable(thisBlock,thisAddr,base,offset);
          offset = 0;
          base = 0;
        }
      }
    }
    if(op==Instruction::Shl){
      isJumpTable = 0;
      isJumpTable++;
      shl = dyn_cast<llvm::Instruction>(I);
    }
   
    if(op==Instruction::Add){
      isJumpTable = isJumpTable + 2;
      if(isJumpTable==3 or isJumpTable==5){
        if(shl){
          offset = GetConst(shl, shl->getOperand(1));
          shl = nullptr;
        }
        add = dyn_cast<llvm::Instruction>(I);
        base = base + GetConst(add, add->getOperand(1)); 
       
        auto Path = "JumpTable.log";
        std::ofstream JTAddr;
        JTAddr.open(Path,std::ofstream::out | std::ofstream::app);
        JTAddr <<"0x"<< std::hex << PC <<"\n";
        JTAddr.close();
      }
    }
  }
  
  if(offset)
    registerJumpTable(thisBlock,thisAddr,base,offset);

}

int64_t JumpTargetManager::GetConst(llvm::Instruction *I, llvm::Value *v){
  auto v1 = v;
  auto operateUser = dyn_cast<User>(I);
  auto bb = I->getParent();
  uint32_t NUMOFCONST = 0;
  LastAssignmentResult result;
  llvm::Instruction *lastInst = nullptr;

  auto Path = "JumpTable.log";
  std::ofstream JTAddr;
  JTAddr.open(Path,std::ofstream::out | std::ofstream::app);

  bool nonConst = true;
  while(nonConst){
    std::tie(result,lastInst) = getLastAssignment(v1,operateUser,bb,JumpTableMode,NUMOFCONST);
    switch(result)
    {
      case CurrentBlockValueDef:
      {
        auto load = dyn_cast<llvm::LoadInst>(lastInst);
        if(load==nullptr){
          return 0;
          //revng_assert(load);
        }
        v1 = load->getPointerOperand();
        if(dyn_cast<Constant>(v1)){
          if(dyn_cast<ConstantInt>(v1)){
            auto integer = llvm::cast<llvm::ConstantInt>(v1)->getSExtValue();
            JTAddr << integer <<"\n";
            JTAddr.close();
            return integer;
          }else{
            auto str = v1->getName();
            JTAddr << str.str();
            auto lable = REGLABLE(StrToInt(str.data()));
            if(lable == UndefineOP)
              revng_abort("Unkown register OP!\n");
            JTAddr << " : "<< std::hex<< ptc.regs[lable]<<"\n";
            JTAddr.close();
            return ptc.regs[lable];
          }
        }
        else{
          operateUser = dyn_cast<User>(lastInst);
          break;
        }
      }
      case CurrentBlockLastAssign:
      {
        // Only Store instruction can assign a value for Value rather than defined
        auto store = dyn_cast<llvm::StoreInst>(lastInst);
        v1 = store->getValueOperand();
        if(dyn_cast<Constant>(v1)){
          if(dyn_cast<ConstantInt>(v1)){
            auto integer = llvm::cast<llvm::ConstantInt>(v1)->getSExtValue();
            JTAddr << integer <<"\n";
            JTAddr.close();
            return integer;
          }else{
            auto str = v1->getName();
            JTAddr << str.str();
            auto lable = REGLABLE(StrToInt(str.data()));
            if(lable == UndefineOP)
              revng_abort("Unkown register OP!\n");
            JTAddr << " : "<< std::hex<< ptc.regs[lable]<<"\n";
            JTAddr.close();
            return ptc.regs[lable];
          }
        }
        else{
          operateUser = dyn_cast<User>(lastInst);
          break;
        }
      }
      case ConstantValueAssign:
      {
        //This arguement v is a constant vale.
        if(dyn_cast<ConstantInt>(v)){
          auto integer = llvm::cast<llvm::ConstantInt>(v)->getSExtValue();
          JTAddr << integer <<"\n";
          JTAddr.close();
          return integer;
        }else{
          auto str = v->getName();
          JTAddr << str.str();
          auto lable = REGLABLE(StrToInt(str.data()));
          if(lable == UndefineOP)
            revng_abort("Unkown register OP!\n");
          JTAddr << " : "<< std::hex<< ptc.regs[lable]<<"\n";
          JTAddr.close();
          return ptc.regs[lable];
        }
      }
      case NextBlockOperating:
      case UnknowResult:
          revng_abort("Unknow of result!");
      break;
    }
  }

  return 0;
}

void JumpTargetManager::storeCPURegister(){
  TempCPURegister.clear();
  for(int i=0; i<16;i++) TempCPURegister.push_back(ptc.regs[i]);
}

void JumpTargetManager::recoverCPURegister(){
  for(size_t i=0; i<TempCPURegister.size(); i++) 
    ptc.regs[i] = TempCPURegister[i];
}

void JumpTargetManager::harvestCodePointerInDataSegment(int64_t pos,llvm::Instruction *tmpI, uint32_t tmpOP){
  std::vector<int64_t> GadgeChain;
  GadgeChain.push_back(pos);
  uint64_t pos_base = pos;
  while(assign_gadge[pos_base].second.pre!=-1){
    pos_base = assign_gadge[pos_base].second.pre;
    GadgeChain.push_back(pos_base);
  }
  std::vector<uint64_t> tmpGlobal;
  std::vector<uint64_t> &tmpGlobal1 = tmpGlobal;
  std::vector<int64_t>::reverse_iterator rit = GadgeChain.rbegin();
  for(;rit!=GadgeChain.rend();rit++){
    auto gadget = assign_gadge[*rit].second.static_addr_block;
    bool oper = false;
    if(assign_gadge[*rit].second.operation_block){
      gadget = assign_gadge[*rit].second.operation_block;
      oper = true;
    }
    auto global_I = assign_gadge[*rit].second.global_I;
    auto op = assign_gadge[*rit].second.op;
    auto indirect = assign_gadge[*rit].second.indirect;
    runGlobalGadget(assign_gadge[*rit].first,gadget,oper,global_I,op,indirect,tmpGlobal1);
  }
  if(assign_gadge[pos].second.operation_block){
    auto gadget = assign_gadge[pos].second.static_addr_block;
    bool oper = false;
    auto global_I = tmpI;
    auto op = tmpOP;
    auto indirect = assign_gadge[pos].second.indirect;
    runGlobalGadget(assign_gadge[pos].first,gadget,oper,global_I,op,indirect,tmpGlobal1);
  }
}

void JumpTargetManager::harvestCodePointerInDataSegment(int64_t pos,uint64_t reserve,llvm::Instruction *tmpI, uint32_t tmpOP){
  std::vector<int64_t> GadgeChain;
  GadgeChain.push_back(pos);
  int64_t pos_base = pos;
  while(assign_gadge[pos_base].second.pre!=-1){
    pos_base = assign_gadge[pos_base].second.pre;
    GadgeChain.push_back(pos_base);
  }
  std::vector<uint64_t> tmpGlobal;
  std::vector<uint64_t> &tmpGlobal1 = tmpGlobal;
  std::vector<int64_t>::reverse_iterator rit = GadgeChain.rbegin();
  auto back = assign_gadge[GadgeChain.back()].second.global_addr;
  assign_gadge[GadgeChain.back()].second.global_addr = reserve;
  for(;rit!=GadgeChain.rend();rit++){
    auto gadget = assign_gadge[*rit].second.static_addr_block;
    bool oper = false;
    if(assign_gadge[*rit].second.operation_block){
      gadget = assign_gadge[*rit].second.operation_block;
      oper = true;
    }
    auto global_I = assign_gadge[*rit].second.global_I;
    auto op = assign_gadge[*rit].second.op;
    auto indirect = assign_gadge[*rit].second.indirect;
    runGlobalGadget(assign_gadge[*rit].second.global_addr,gadget,oper,global_I,op,indirect,tmpGlobal1);
  }
  assign_gadge[GadgeChain.back()].second.global_addr = back;
  if(assign_gadge[pos].second.operation_block){
    auto gadget = assign_gadge[pos].second.static_addr_block;
    bool oper = false;
    auto global_I = tmpI;
    auto op = tmpOP;
    auto indirect = assign_gadge[pos].second.indirect;
    runGlobalGadget(assign_gadge[pos].first,gadget,oper,global_I,op,indirect,tmpGlobal1);
  }
}

void JumpTargetManager::harvestCodePointerInDataSegment(int64_t pos){
  std::vector<int64_t> GadgeChain;
  GadgeChain.push_back(pos);
  int64_t pos_base = pos;
  while(assign_gadge[pos_base].second.pre!=-1){
    pos_base = assign_gadge[pos_base].second.pre;
    GadgeChain.push_back(pos_base);
  }
  std::vector<uint64_t> tmpGlobal;
  std::vector<uint64_t> &tmpGlobal1 = tmpGlobal;
  std::vector<int64_t>::reverse_iterator rit = GadgeChain.rbegin();
  for(;rit!=GadgeChain.rend();rit++){
    auto gadget = assign_gadge[*rit].second.static_addr_block;
    bool oper = false;
    if(assign_gadge[*rit].second.operation_block){
      gadget = assign_gadge[*rit].second.operation_block;
      oper = true;
    }
    auto global_I = assign_gadge[*rit].second.global_I;
    auto op = assign_gadge[*rit].second.op;
    auto indirect = assign_gadge[*rit].second.indirect;
    runGlobalGadget(assign_gadge[*rit].first,gadget,oper,global_I,op,indirect,tmpGlobal1);
  }
}

void JumpTargetManager::harvestCodePointerInDataSegment(int64_t pos, uint64_t reserve){
  std::vector<int64_t> GadgeChain;
  GadgeChain.push_back(pos);
  int64_t pos_base = pos;
  while(assign_gadge[pos_base].second.pre!=-1){
    pos_base = assign_gadge[pos_base].second.pre;
    GadgeChain.push_back(pos_base);
  }
  std::vector<uint64_t> tmpGlobal;
  std::vector<uint64_t> &tmpGlobal1 = tmpGlobal;
  std::vector<int64_t>::reverse_iterator rit = GadgeChain.rbegin();
  auto back = assign_gadge[GadgeChain.back()].second.global_addr;
  assign_gadge[GadgeChain.back()].second.global_addr = reserve;
  for(;rit!=GadgeChain.rend();rit++){
    auto gadget = assign_gadge[*rit].second.static_addr_block;
    bool oper = false;
    if(assign_gadge[*rit].second.operation_block){
      gadget = assign_gadge[*rit].second.operation_block;
      oper = true;
    }
    auto global_I = assign_gadge[*rit].second.global_I;
    auto op = assign_gadge[*rit].second.op;
    auto indirect = assign_gadge[*rit].second.indirect;
    runGlobalGadget(assign_gadge[*rit].second.global_addr,gadget,oper,global_I,op,indirect,tmpGlobal1);
  }
  assign_gadge[GadgeChain.back()].second.global_addr = back;
}

void JumpTargetManager::runGlobalGadget(uint64_t basePC,
                                        llvm::BasicBlock * gadget,
                                        bool oper,
                                        llvm::Instruction * global_I,
                                        uint32_t op,
                                        bool indirect,
                                        std::vector<uint64_t> &tmpGlobal){
    if(gadget==nullptr or global_I==nullptr)
      return;
    auto thisAddr = getInstructionPC(&*(gadget->begin()));
    if(thisAddr==0)
      return;
    auto current_pc = getInstructionPC(global_I);
    std::vector<uint64_t> tempVec; 
    std::vector<uint64_t> &tempVec1 = tempVec;

    if(tmpGlobal.empty())
      tmpGlobal.push_back(basePC);

    uint32_t opt = UndefineOP;
    uint64_t virtualAddr = 0;
    uint32_t crash = 0;
    bool recover = false;
    if(current_pc != thisAddr){
      std::tie(opt,virtualAddr) = getLastOperandandNextPC(&*(gadget->begin()));
      if(virtualAddr==0)
        return;
    } 
    if(current_pc == thisAddr or opt==UndefineOP or opt==R_ESP){
      /* If the instruction to operate global data is entry address,
       * we consider that no instruction operates offset, and offset value
       * has been designated in global_I. */
      if(isDataSegmAddr(ptc.regs[R_ESP])){
        ptc.storeStack();
        recover = true;
      }
      storeCPURegister();
      for(auto base : tmpGlobal){
        if(op!=UndefineOP)
          ptc.regs[op] = base;
        ConstOffsetExec(gadget,thisAddr,current_pc,oper,global_I,op,indirect,crash,tempVec1);
        crash++;
      }
      tmpGlobal.clear();
      tmpGlobal = tempVec1;
      recoverCPURegister();
      if(recover)
        ptc.recoverStack();
      return; 
    }

    /* If the instruction to operate global data isn't entry address of block, 
     * then we consider all instructions before this instruction will be the instruction
     * to operate offset value. */
    if(isDataSegmAddr(ptc.regs[R_ESP])){
      ptc.storeStack();
      recover = true;
    }
    storeCPURegister();
    for(auto base:tmpGlobal){
      if(op!=UndefineOP)
        ptc.regs[op] = base;
      VarOffsetExec(gadget,thisAddr,virtualAddr,current_pc,oper,op,opt,indirect,crash,tempVec1);
      crash++;
    }
    tmpGlobal.clear();
    tmpGlobal = tempVec1;
    recoverCPURegister();
    if(recover)
      ptc.recoverStack();
}

void JumpTargetManager::ConstOffsetExec(llvm::BasicBlock *gadget,
                                        uint64_t thisAddr, 
                                        uint64_t current_pc, 
                                        bool oper,
                                        llvm::Instruction * global_I, 
                                        uint32_t op, 
                                        bool indirect,
                                        uint32_t crash,
                                        std::vector<uint64_t>& tempVec){
  size_t pagesize = 0;        
  uint32_t gadgetCrash = 0; 
  uint64_t check = 0;
 
  auto Path = "GlobalPointer.log";
  std::ofstream BaseAddr;
  BaseAddr.open(Path,std::ofstream::out | std::ofstream::app);
  BaseAddr <<"PC: "<<std::hex<< current_pc <<" : \n";
  if(op!=UndefineOP)
    BaseAddr <<"base: "<<std::hex<< ptc.regs[op] <<" : \n";
  for(;;){
    //Static addresses are indirect jump target address.
    int64_t tmpPC = ptc.exec(thisAddr);
    if(tmpPC==-1){
      if(!crash){ 
        if(!gadgetCrash){
          break;
          //revng_assert(tmpPC!=-1);
          
        }
        else
          break;
      }
      else
        break;
    }
    gadgetCrash++;
    if(oper){
      auto data = getGlobalDatafromDestRegs(gadget);
      if(!isGlobalData(data))
        break;
      bool have = false;
      for(auto t:tempVec){
        if(t==data){
          have = true;
          break;
        }
      }
      if(have==false){
        tempVec.push_back(data);
        BaseAddr <<"  data:    0x"<< std::hex << data <<"\n";
      }
      pagesize++;
      if(pagesize>256)
        break;
      if(!haveDef2OP(global_I,op))
        break;
      continue;
    }
    // Static addresses stored in registers.
    if(!indirect)
      tmpPC = getStaticAddrfromDestRegs(gadget,current_pc);
    pagesize++;
    if(!haveDef2OP(global_I,op) or pagesize>256)
      break;
    if(tmpPC==0)
      continue;
    if(!isExecutableAddress(tmpPC))
      break;    
    if(pagesize<5 or pagesize==5){
      check += tmpPC;
    }
    if(pagesize==5){
      if((check-tmpPC*5)==0) 
        break;
    }
    harvestBTBasicBlock(gadget,thisAddr,tmpPC);
   
    BaseAddr <<"    0x"<< std::hex << tmpPC <<"\n";     
  }
  BaseAddr.close();
}

void JumpTargetManager::VarOffsetExec(llvm::BasicBlock *gadget,
                                      uint64_t thisAddr,
                                      uint64_t virtualAddr,
                                      uint64_t current_pc,
                                      bool oper,
                                      uint32_t op,
                                      uint32_t opt,
                                      bool indirect,
                                      uint32_t crash,
                                      std::vector<uint64_t>& tempVec){
  size_t pagesize = 0;
  uint32_t gadgetCrash = 0;
  uint64_t check = 0;
        
  auto Path = "GlobalPointer.log";
  std::ofstream BaseAddr;
  BaseAddr.open(Path,std::ofstream::out | std::ofstream::app);
  BaseAddr <<"PC: "<<std::hex<< virtualAddr <<" : \n";  
  if(op!=UndefineOP)
    BaseAddr <<"base: "<<std::hex<< ptc.regs[op] <<" : \n";
  for(int i=0; ;i++){
    ptc.regs[opt] = i; 
    //Static addresses are indirect jump target address.
    int64_t tmpPC = ptc.exec(virtualAddr);
    if(tmpPC==-1){
      if(!crash){ 
        if(!gadgetCrash){
          break;
          //revng_assert(tmpPC!=-1);
        }
        else
          break;
      }
      else
        break;
    }
    gadgetCrash++;
    if(oper){
      //TODO:vector data ?
      auto data = getGlobalDatafromDestRegs(gadget);
      if(!isGlobalData(data))
        break;
      bool have = false;
      for(auto t:tempVec){
        if(t==data){
          have = true;
          break;
        }
      }
      if(have==false){
        tempVec.push_back(data); 
        BaseAddr <<"  data    0x"<< std::hex << data <<"\n";
      }
      pagesize++;
      if(pagesize>256)
        break;
      continue;
    }
    // Static addresses stored in registers.
    if(!indirect)
      tmpPC = getStaticAddrfromDestRegs(gadget,current_pc);    
    pagesize++;
    if(pagesize>256)
      break;
    if(tmpPC==0)
      continue; 
    if(!isExecutableAddress(tmpPC))
      break;   
    if(pagesize<5 or pagesize==5){
      check += tmpPC;
    } 
    if(pagesize==5){
      if((check-tmpPC*5)==0)
        break;
    }

    harvestBTBasicBlock(gadget,thisAddr,tmpPC);
   
    BaseAddr <<"    0x"<< std::hex << tmpPC <<"\n"; 
  }
  BaseAddr.close();    
}

void JumpTargetManager::harvestStaticAddr(llvm::BasicBlock *thisBlock){
//  if(!isDataSegmAddr(ptc.regs[R_ESP]))
//    return;
  if(thisBlock==nullptr)
    return;

  BasicBlock::reverse_iterator I(thisBlock->rbegin());
  BasicBlock::reverse_iterator rend(thisBlock->rend());
  bool staticFlag = 1;

  auto branch = dyn_cast<BranchInst>(&*I); 
  if(branch && !branch->isConditional())
    staticFlag = 0;

  for(; I!=rend; I++){
    if(staticFlag){
      auto callI = dyn_cast<CallInst>(&*I);
      if(callI){
        auto *Callee = callI->getCalledFunction();
  	if(Callee != nullptr && Callee->getName() == "newpc")
          staticFlag = 0;
      }
    }
    if(!staticFlag){
      if(I->getOpcode()==Instruction::Call){
        auto other = dyn_cast<CallInst>(&*I);
	if(other){
          auto *Callee = other->getCalledFunction();
	  if(Callee != nullptr && Callee->getName() != "newpc")
	    staticFlag = 1;
	}
      }
      if(I->getOpcode()==Instruction::Store){
        auto store = dyn_cast<llvm::StoreInst>(&*I);
        auto v = store->getValueOperand();
        if(dyn_cast<ConstantInt>(v)){
          auto pc = getLimitedValue(v);
          // Harvest entry addresses stored in data segment
          StaticAddrsMap::iterator TargetIt = JumpTableBase.find(pc);
          if(isGlobalData(pc) and TargetIt==JumpTableBase.end()){
            AssignGadge AG(pc);
            assign_gadge.push_back({pc,AG});
            AllGlobalAddr[pc] = 1;
            int64_t pos = assign_gadge.size()-1; 
            if(haveBinaryOperation(&*I)){
              assign_gadge[pos].second.operation_block = thisBlock;
              auto Itt = dyn_cast<llvm::Instruction>(&*I);
              assign_gadge[pos].second.global_I = Itt;
              //if thisBlock is indirect or StaticAddr is stored in registers. 
              if(*ptc.isIndirect or *ptc.isIndirectJmp or 
                  (getStaticAddrfromDestRegs(&*I) and 
                                 !isCase1(&*I,pc) and 
                                 !isCase2(&*I)
                  ))
              {
                assign_gadge[pos].second.static_addr_block = thisBlock;
                assign_gadge[pos].second.operation_block = nullptr;
                AllStaticGadget[thisBlock] = 1;
                if(*ptc.isIndirect or *ptc.isIndirectJmp) 
                    assign_gadge[pos].second.indirect = true;
                harvestCodePointerInDataSegment(pos);
              }else{
                // There have global data by logical operations in thisBlock registers
                auto result = getGlobalDatafromRegs(&*I,pos);
                if(result){
                  AllGadget[thisBlock] = 1;
                  assign_gadge[pos].second.end = false;
                }
                if(!result){
                  assign_gadge[pos].second.operation_block = nullptr;
                  assign_gadge[pos].second.global_I = nullptr; 
                }
              }
            }
          }
	  if(!haveTranslatedPC(pc, 0) and !isIllegalStaticAddr(pc)){
	    StaticAddrs[pc] = false;
            //errs()<<format_hex(pc,0)<<"  147852369\n";
          }
        }
      }
    }
  }
}

void JumpTargetManager::handleGlobalDataGadget(llvm::BasicBlock *thisBlock,std::map<uint32_t, uint64_t> &GloData){
  BasicBlock::iterator it(thisBlock->begin());
  BasicBlock::iterator end(thisBlock->end());

  for(;it!=end;it++){
    if(it->getOpcode()==Instruction::Load){
      auto load = dyn_cast<llvm::LoadInst>(it);
      auto v = load->getPointerOperand();
      if(dyn_cast<Constant>(v)){
        StringRef name = v->getName();
        auto number = StrToInt(name.data());
        auto reg = REGLABLE(number); 
        if(reg==UndefineOP) 
          continue;
        // If thisBlock only transfer baseGlobal, and there is no operation, skip it.
        auto TargetIt = GloData.find(reg);
        if(TargetIt!=GloData.end() and haveBinaryOperation(&*it) and !isJumpTabType(&*it)){
          auto baseGlobal = TargetIt->second;
          //TODO: offset is or not const
          //if(haveDefOperation(&*it,v))
          //  return;
          int64_t i = isRecordGadgetBlock(baseGlobal,thisBlock);
	  if(i==-1){
	    AssignGadge AG(baseGlobal);
	    assign_gadge.push_back({baseGlobal,AG});
            AllGlobalAddr[baseGlobal] = 1;
	    i = assign_gadge.size()-1;
	  }
          uint32_t tmpOP = UndefineOP;
          // global_I is an instruction to operate global data
          auto tmpBB = assign_gadge[i].second.operation_block;
          auto tmpI = assign_gadge[i].second.global_I;
          tmpOP = assign_gadge[i].second.op;

	  assign_gadge[i].second.operation_block = thisBlock;
          auto itt= dyn_cast<llvm::Instruction>(it); 
          assign_gadge[i].second.global_I = itt;
          assign_gadge[i].second.op = reg;
          if(*ptc.isIndirect or *ptc.isIndirectJmp or getStaticAddrfromDestRegs(&*it)){
            assign_gadge[i].second.static_addr_block = thisBlock;
            assign_gadge[i].second.operation_block = nullptr;
            AllStaticGadget[thisBlock] = 1;
            if(*ptc.isIndirect or *ptc.isIndirectJmp)
              assign_gadge[i].second.indirect = true;  
            harvestCodePointerInDataSegment(i);
            break;
          }else{
            auto result = getGlobalDatafromRegs(&*it,i);
            if(result){
              if(assign_gadge[i].second.static_addr_block){
                assign_gadge[i].second.static_global_I = tmpI;
                assign_gadge[i].second.static_op = tmpOP;
                harvestCodePointerInDataSegment(i,tmpI,tmpOP);
              }
              assign_gadge[i].second.end = false;
              AllGadget[thisBlock] = 1;
            }
            if(!result){
              assign_gadge[i].second.operation_block = tmpBB;
              assign_gadge[i].second.global_I = tmpI;
              assign_gadge[i].second.op = tmpOP;
            }
            break;
          }
        }
      }
    }
  }//?end for? 
}

void JumpTargetManager::handleGlobalStaticAddr(void){
  for(auto base:AllGlobalAddr){
   
      for(unsigned staticIt=0; staticIt<assign_gadge.size(); staticIt++){
        if(assign_gadge[staticIt].second.end){
          auto reserve = base.first;
          if(assign_gadge[staticIt].second.operation_block){
            auto tmpI = assign_gadge[staticIt].second.static_global_I;
            auto tmpOP = assign_gadge[staticIt].second.static_op;
            harvestCodePointerInDataSegment(staticIt,reserve,tmpI,tmpOP);
          }
          harvestCodePointerInDataSegment(staticIt,reserve);
        }
      }
    
  }  
}

bool JumpTargetManager::isGlobalData(uint64_t pc){
  if(DataSegmStartAddr<pc and pc<DataSegmEndAddr)
    return true;
  if(ro_StartAddr<pc and pc<ro_EndAddr)
    return true;
 
  return false;
}

bool JumpTargetManager::haveDefOperation(llvm::Instruction *I, llvm::Value *v){
  BasicBlock::reverse_iterator it(I);
  BasicBlock::reverse_iterator rend = I->getParent()->rend();

  for(; it!=rend; it++){
    if(it->getOpcode()==Instruction::Store){
      auto store = dyn_cast<llvm::StoreInst>(&*it);
      if(store->getPointerOperand()==v)
        return true;
    }
  }
  return false;
}

bool JumpTargetManager::haveDef2OP(llvm::Instruction *I, uint32_t op){
  BasicBlock::iterator it(I);
  BasicBlock::iterator end = I->getParent()->end(); 
 
  /* If op is UndefineOP, the base explicit exist gadget block,
   * we consider no loop deadlock. */
  if(op==UndefineOP)
    return true; 
  for(; it!=end; it++){
    if(it->getOpcode()==Instruction::Store){
      auto store = dyn_cast<llvm::StoreInst>(&*it);
      auto v = store->getPointerOperand();
      if(dyn_cast<Constant>(v)){
        StringRef name = v->getName();
        auto number = StrToInt(name.data());
        auto reg = REGLABLE(number);
        if(reg==op)
          return true;
      } 
    }
  }
  return false;
}

bool JumpTargetManager::isRecordGlobalBase(uint64_t base){
  std::map<uint64_t,uint32_t>::iterator Target = AllGlobalAddr.find(base);
  if(Target != AllGlobalAddr.end())
      return true;
  
  return false;
}

int64_t JumpTargetManager::isRecordGadgetBlock(uint64_t base, llvm::BasicBlock *gadget){
//  std::map<llvm::BasicBlock *,uint32_t>::iterator Target = AllGadget.find(gadget);
//  if(Target!=AllGadget.end())
//      return -2;
  
  for(unsigned i=0; i<assign_gadge.size(); i++){
    if(assign_gadge[i].first==base){
      if(assign_gadge[i].second.operation_block==nullptr)
        return i;
    }
  }

  return -1;
}

bool JumpTargetManager::isJumpTabType(llvm::Instruction *I){
  BasicBlock::iterator it(I);
  BasicBlock::iterator end = I->getParent()->end();
  bool flag = false;
  for(; it!=end; it++){
    if(it->getOpcode()==Instruction::Store){
      auto store = dyn_cast<llvm::StoreInst>(&*it);
      auto v = store->getValueOperand();
      if(dyn_cast<ConstantInt>(v)){
         auto pc = getLimitedValue(v);
         if(pc>codeSeg_StartAddr and (pc<text_StartAddr or pc>=ro_StartAddr))
           flag = true;       
      }
    }
    if(it->getOpcode()==Instruction::Call){
      auto callI = dyn_cast<CallInst>(&*it);
      auto *Callee = callI->getCalledFunction();
      if(Callee != nullptr && Callee->getName() == "newpc")
        break;
    }
  }
  return flag;
}

// If this value have a binary operation and assign to reg/mem.
bool JumpTargetManager::haveBinaryOperation(llvm::Instruction *I){
  BasicBlock::iterator it(I);
  BasicBlock::iterator end = I->getParent()->end();
  BasicBlock::iterator lastInst(I->getParent()->back());

  auto v = dyn_cast<llvm::Value>(I);
  if(I->getOpcode()==Instruction::Store){
    auto store = dyn_cast<llvm::StoreInst>(I);
    v = store->getPointerOperand();
  }

  bool flag = false;
  it++;
  for(; it!=end; it++){ 
    switch(it->getOpcode()){
      case llvm::Instruction::Add:{  
        auto add = dyn_cast<Instruction>(it);
        if((add->getOperand(0) - v) == 0){
          v = dyn_cast<Value>(add);
          if(!flag)
            flag = true; 
        }
        else if((add->getOperand(1) - v) == 0){
          v = dyn_cast<Value>(add);
          if(!flag)
            flag = true;
        }
        break;
      }
      case llvm::Instruction::Call:
        break;
      case llvm::Instruction::Load:{
        auto load = dyn_cast<llvm::LoadInst>(it);
	if((load->getPointerOperand() - v) == 0)
	    v = dyn_cast<Value>(it);
        break;
      }
      case llvm::Instruction::Store:{
        auto store = dyn_cast<llvm::StoreInst>(it);
        if((store->getValueOperand() - v) == 0){
	    v = store->getPointerOperand();
            if(flag){
              if(dyn_cast<Constant>(v))
                return true;
            }
        }
	break;
      }
      default:{
        auto instr = dyn_cast<Instruction>(it);
        for(Use &u : instr->operands()){
            Value *InstV = u.get();
            if((InstV - v) == 0){
	      v = dyn_cast<Value>(instr);
              break; 
            }
        }
      }  
    }
  }//??end for
  return false;
}

void JumpTargetManager::haveGlobalDatainRegs(std::map<uint32_t, uint64_t> &GloData){
  for(int i=0; i<16; i++){
    if(isGlobalData(ptc.regs[i])){
      if(isRecordGlobalBase(ptc.regs[i])){  
        std::map<uint32_t, uint64_t>::iterator Target = GloData.find(i);
        if(Target==GloData.end())
          GloData[i] = ptc.regs[i];
      }
      else{
        AssignGadge AG(ptc.regs[i]);
        assign_gadge.push_back({ptc.regs[i],AG});
        AllGlobalAddr[ptc.regs[i]] = 1;
        GloData[i] = ptc.regs[i];
      }
    }
  }  
}

bool JumpTargetManager::getGlobalDatafromRegs(llvm::Instruction *I, int64_t pre){
  bool result =false; 
  BasicBlock::iterator it(I);
  BasicBlock::iterator end = I->getParent()->end();
  BasicBlock::iterator lastInst(I->getParent()->back());

  auto v = dyn_cast<llvm::Value>(I);
  if(I->getOpcode()==Instruction::Store){
    auto store = dyn_cast<llvm::StoreInst>(I);
    v = store->getPointerOperand();
  }

  it++;
  for(; it!=end; it++){ 
    switch(it->getOpcode()){
      case llvm::Instruction::Call:
        break;
      case llvm::Instruction::Load:{
        auto load = dyn_cast<llvm::LoadInst>(it);
	if((load->getPointerOperand() - v) == 0)
	    v = dyn_cast<Value>(it);
        break;
      }
      case llvm::Instruction::Store:{
        auto store = dyn_cast<llvm::StoreInst>(it);
        if((store->getValueOperand() - v) == 0){
	    v = store->getPointerOperand(); 
            if(dyn_cast<Constant>(v)){
              StringRef name = v->getName();
              auto number = StrToInt(name.data());
              auto op = REGLABLE(number);
              if(op==UndefineOP)
                continue;
              if(isGlobalData(ptc.regs[op])){
                if(!isRecordGlobalBase(ptc.regs[op])){
                  AllGlobalAddr[ptc.regs[op]] = 1;
                } 
	        AssignGadge AG(ptc.regs[op]);
                AG.pre = pre;
                assign_gadge.push_back({ptc.regs[op],AG});
                result = true;
              }
            }
        }
	break;
      }
      default:{
        auto instr = dyn_cast<Instruction>(it);
        for(Use &u : instr->operands()){
            Value *InstV = u.get();
            if((InstV - v) == 0){
	      v = dyn_cast<Value>(instr);
              break; 
            }
        }
      }  
    }
  }//??end for
  if(result)
    return result;
  return false;
}

uint64_t JumpTargetManager::getGlobalDatafromDestRegs(llvm::BasicBlock *thisBlock){
  BasicBlock::iterator it(thisBlock->begin());
  BasicBlock::iterator end(thisBlock->end());
  
  for(;it!=end;it++){
    //Only look for dest register
    if(it->getOpcode()==Instruction::Store){
      auto store = dyn_cast<llvm::StoreInst>(it);
      auto v = store->getPointerOperand();
      if(dyn_cast<Constant>(v)){
       StringRef name = v->getName();
       auto number = StrToInt(name.data());
       auto op = REGLABLE(number); 
       if(op==UndefineOP) 
         continue;
       if(isGlobalData(ptc.regs[op])){
         return ptc.regs[op];
       } 
      }     
    }
  }//?end for?
 
  return 0;
}

//Case1: Load static address from global, and store this value to global
bool JumpTargetManager::isCase1(llvm::Instruction *I, uint64_t global){
  BasicBlock::iterator it(I);
  BasicBlock::iterator end = I->getParent()->end();
  BasicBlock::iterator lastInst(I->getParent()->back());

  Value *v1 = nullptr;
  auto v = dyn_cast<llvm::Value>(I);
  if(I->getOpcode()==Instruction::Store){
    auto store = dyn_cast<llvm::StoreInst>(I);
    v = store->getPointerOperand();
  }

  it++;
  for(; it!=end; it++){ 
    switch(it->getOpcode()){
      case llvm::Instruction::Call:
        break;
      case llvm::Instruction::Load:{
        auto load = dyn_cast<llvm::LoadInst>(it);
	if((load->getPointerOperand() - v) == 0)
	    v = dyn_cast<Value>(it);
        if((load->getPointerOperand() - v1) == 0)
            v1 = dyn_cast<Value>(it);
        break;
      }
      case llvm::Instruction::Store:{
        auto store = dyn_cast<llvm::StoreInst>(it);
        if(dyn_cast<ConstantInt>(store->getValueOperand())){
          auto addr = getLimitedValue(store->getValueOperand()); 
          if(addr==global)
            v1 = store->getPointerOperand();
        }
        if((store->getValueOperand() - v) == 0)
	    v = store->getPointerOperand();
        if((store->getValueOperand() - v1) == 0)
            v1 = store->getPointerOperand();
	break;
      }
      default:{
        auto instr = dyn_cast<Instruction>(it);
        for(Use &u : instr->operands()){
            Value *InstV = u.get();
            if((InstV - v) == 0){
	      v = dyn_cast<Value>(instr);
              break; 
            }
            if((InstV - v1) == 0){
              v1 = dyn_cast<Value>(instr);
              break;
            }
        }
      }  
    }
    if(v==v1)
      return true;
  }//??end for
  return false;
}


//Case2: Load value from global, and add a static addr from a reg to the reg
//          [global] + reg(static) to reg
bool JumpTargetManager::isCase2(llvm::Instruction *I){
  BasicBlock::iterator it(I);
  BasicBlock::iterator end = I->getParent()->end();
  BasicBlock::iterator lastInst(I->getParent()->back());

  Value *v1 = nullptr;
  auto v = dyn_cast<llvm::Value>(I);
  if(I->getOpcode()==Instruction::Store){
    auto store = dyn_cast<llvm::StoreInst>(I);
    v = store->getPointerOperand();
  }

  it++;
  for(; it!=end; it++){ 
    switch(it->getOpcode()){
      case llvm::Instruction::Call:
        return false;
      case llvm::Instruction::Load:{
        auto load = dyn_cast<llvm::LoadInst>(it);
        if((load->getPointerOperand() - v) == 0)
            v = dyn_cast<Value>(it);
        if((load->getPointerOperand() - v1) == 0)
            v1 = dyn_cast<Value>(it);
        auto RegV = load->getPointerOperand();
        if(dyn_cast<Constant>(RegV)){
            StringRef name = RegV->getName();
            auto number = StrToInt(name.data());
            auto op = REGLABLE(number);
            if(op==UndefineOP)
                continue;
            if(isExecutableAddress(ptc.regs[op]))
                v1 = dyn_cast<Value>(it);
        }
        break;
      }
      case llvm::Instruction::Store:{
        auto store = dyn_cast<llvm::StoreInst>(it);
        if((store->getValueOperand() - v) == 0)
	    v = store->getPointerOperand();
        if((store->getValueOperand() - v1) == 0)
            v1 = store->getPointerOperand();
	break;
      }
      case llvm::Instruction::Add:{
        auto instr = dyn_cast<Instruction>(it);
        for(Use &u : instr->operands()){
            Value *InstV = u.get();
            if((InstV - v) == 0)
                v = dyn_cast<Value>(instr);
            if((InstV - v1) == 0)
                v1 = dyn_cast<Value>(instr);
        }
        if(v==v1)
          return true;
      }
      default:{
        auto instr = dyn_cast<Instruction>(it);
        for(Use &u : instr->operands()){
            Value *InstV = u.get();
            if((InstV - v) == 0)
	      v = dyn_cast<Value>(instr);
            if((InstV - v1) == 0)
              v1 = dyn_cast<Value>(instr);
        }
      }  
    }
  }//??end for
  return false;
}

bool JumpTargetManager::getStaticAddrfromDestRegs1(llvm::Instruction *I){
  BasicBlock::iterator it(I);
  BasicBlock::iterator end = I->getParent()->end();
  BasicBlock::iterator lastInst(I->getParent()->back());

  auto v = dyn_cast<llvm::Value>(I);
  if(I->getOpcode()==Instruction::Store){
    auto store = dyn_cast<llvm::StoreInst>(I);
    v = store->getPointerOperand();
  }

  bool flag =false;
  it++;
  for(; it!=end; it++){ 
    switch(it->getOpcode()){
      case llvm::Instruction::Call:
        break;
      case llvm::Instruction::Load:{
        auto load = dyn_cast<llvm::LoadInst>(it);
	if((load->getPointerOperand() - v) == 0)
	    v = dyn_cast<Value>(it);
        break;
      }
      case llvm::Instruction::Store:{
        auto store = dyn_cast<llvm::StoreInst>(it);
        if((store->getValueOperand() - v) == 0){
	    v = store->getPointerOperand();
            if(dyn_cast<Constant>(v)){
              StringRef name = v->getName();
              auto number = StrToInt(name.data());
              auto op = REGLABLE(number);
              if(op==UndefineOP)
                  continue;
              if(ptc.regs[op]==0 and flag)
                return true;
              if(isExecutableAddress(ptc.regs[op]))
                return true;
            }
        }
	break;
      }
      case llvm::Instruction::IntToPtr:{
        auto inttoptrI = dyn_cast<Instruction>(it);
  	if((inttoptrI->getOperand(0) - v) == 0){
  	    flag = true;
            v = dyn_cast<Value>(inttoptrI);
        }
  	break;
      }
      default:{
        auto instr = dyn_cast<Instruction>(it);
        for(Use &u : instr->operands()){
            Value *InstV = u.get();
            if((InstV - v) == 0){
	      v = dyn_cast<Value>(instr);
              break; 
            }
        }
      }  
    }
  }//??end for
  return false;
}


bool JumpTargetManager::getStaticAddrfromDestRegs(llvm::Instruction *I){
  BasicBlock::iterator it(I);
  BasicBlock::iterator end = I->getParent()->end();
  BasicBlock::iterator lastInst(I->getParent()->back());

  auto v = dyn_cast<llvm::Value>(I);
  if(I->getOpcode()==Instruction::Store){
    auto store = dyn_cast<llvm::StoreInst>(I);
    v = store->getPointerOperand();
  }

  it++;
  for(; it!=end; it++){ 
    switch(it->getOpcode()){
      case llvm::Instruction::Call:
        break;
      case llvm::Instruction::Load:{
        auto load = dyn_cast<llvm::LoadInst>(it);
	if((load->getPointerOperand() - v) == 0)
	    v = dyn_cast<Value>(it);
        break;
      }
      case llvm::Instruction::Store:{
        auto store = dyn_cast<llvm::StoreInst>(it);
        if((store->getValueOperand() - v) == 0){
	    v = store->getPointerOperand();
            if(dyn_cast<Constant>(v)){
              StringRef name = v->getName();
              auto number = StrToInt(name.data());
              auto op = REGLABLE(number);
              if(op==UndefineOP)
                  continue;
              if(isExecutableAddress(ptc.regs[op]))
                return true;
            }
        }
	break;
      }
      default:{
        auto instr = dyn_cast<Instruction>(it);
        for(Use &u : instr->operands()){
            Value *InstV = u.get();
            if((InstV - v) == 0){
	      v = dyn_cast<Value>(instr);
              break; 
            }
        }
      }  
    }
  }//??end for
  return false;
}

uint64_t JumpTargetManager::getStaticAddrfromDestRegs(llvm::BasicBlock *thisBlock, uint64_t bound){
  BasicBlock::iterator it(thisBlock->begin());
  BasicBlock::iterator end(thisBlock->end());
  
  revng_assert(bound!=0);
  for(;it!=end;it++){
    //Only look for dest register.
    if(it->getOpcode()==Instruction::Store){
      auto store = dyn_cast<llvm::StoreInst>(it);
      auto v = store->getPointerOperand();
      if(dyn_cast<Constant>(v)){
        StringRef name = v->getName();
        auto number = StrToInt(name.data());
        auto op = REGLABLE(number); 
        if(op==UndefineOP) 
          continue;
        if(isExecutableAddress(ptc.regs[op])){
          auto addr = getInstructionPC(&*it);
          if(addr >= bound)
            return ptc.regs[op]; 
          continue;
        }
      }     
    }
  }
  return 0;
}

bool JumpTargetManager::isIllegalStaticAddr(uint64_t pc){
  if(ro_StartAddr<=pc and pc<ro_EndAddr)
    return true;

  if(IllegalStaticAddrs.empty()){
    return false;
  }
  for(auto addr : IllegalStaticAddrs){
    if(pc >= addr)
      return true;
  }

  return false;
}

void JumpTargetManager::harvestNextAddrofBr(uint64_t blockNext){
//  if(!haveTranslatedPC(blockNext, 0)){
//      if(*ptc.isCall){
//        StaticAddrs[blockNext] = 2;
//      }else{
//        StaticAddrs[blockNext] = true;
//      }
//  }
  if(Statistics and *ptc.isDirectJmp){
      IndirectBlocksMap::iterator it = DirectJmpBlocks.find(*ptc.isDirectJmp);
      if(it == DirectJmpBlocks.end())
          DirectJmpBlocks[*ptc.isDirectJmp] = 1;
  }
}

void JumpTargetManager::harvestRetBlocks(uint64_t thisAddr, uint64_t blockNext){
//  if(!haveTranslatedPC(blockNext, 0))
//    StaticAddrs[blockNext] = true;
  if(Statistics){
    IndirectBlocksMap::iterator it = RetBlocks.find(thisAddr);
    if(it == RetBlocks.end())
        RetBlocks[thisAddr] = 1;
  }
}

void JumpTargetManager::StatisticsLog(void){
  if(!Statistics)
      return;
  outs()<<"---------------------------------------\n";
  outs()<<"Indirect Calls:"<<"                "<<IndirectCallBlocks.size()<<"\n";
  outs()<<"Indirect Jumps:"<<"                "<<IndirectJmpBlocks.size()<<"\n";
  outs()<<"Direct Jumps:"<<"                "<<DirectJmpBlocks.size() +1<<"\n";
  outs()<<"Returns:"<<"                       "<<RetBlocks.size()<<"\n";
  outs()<<"\n";
  outs()<<"Jump Tables of Call:"<<"           "<<CallTable.size()<<"\n";
  outs()<<"Jump Tables of Jmp:"<<"            "<<JmpTable.size()<<"\n";
  outs()<<"\n";
  outs()<<"Call Branches:"<<"                 "<<CallBranches.size()<<"\n";
  outs()<<"Cond. Branches:"<<"                "<<CondBranches.size()<<"\n";
}

bool JumpTargetManager::handleStaticAddr(void){
  if(UnexploreStaticAddr.empty()){
    StaticToUnexplore();
    if(UnexploreStaticAddr.empty())
      return false;
  }
  uint64_t addr = 0;
  uint32_t flag =false;
  while(!UnexploreStaticAddr.empty()){
    auto it = UnexploreStaticAddr.begin();
    addr = it->first;
    flag = it->second;
   if(haveTranslatedPC(addr, 0) or isIllegalStaticAddr(addr)){
     if(flag==2)
       CallNextToStaticAddr(it->first);
     UnexploreStaticAddr.erase(it);
     if(UnexploreStaticAddr.empty()){
       StaticToUnexplore();
     }
   }
   else{
     registerJT(addr,JTReason::GlobalData);
     if(UnexploreStaticAddr.size()==1)
       handleGlobalStaticAddr();
     UnexploreStaticAddr.erase(it);
     break;
   } 
  }

  return flag;
}

void JumpTargetManager::StaticToUnexplore(void){
   for(auto& PC : StaticAddrs){
    BlockMap::iterator TargetIt = JumpTargets.find(PC.first);
    BlockMap::iterator upper;
    upper = JumpTargets.upper_bound(PC.first);
    if(TargetIt == JumpTargets.end() && upper != JumpTargets.end()
      	     && !isIllegalStaticAddr(PC.first)){
      errs()<<format_hex(upper->first,0)<<"  :first\n";
      errs()<<format_hex(PC.first,0)<<" <- static address\n";  
      UnexploreStaticAddr[PC.first] = PC.second;
    }
    if(TargetIt != JumpTargets.end() and PC.second == 2)
      CallNextToStaticAddr(PC.first); 
  } 
  StaticAddrs.clear();
}

void JumpTargetManager::CallNextToStaticAddr(uint32_t PC){
  BasicBlock * Block = obtainJTBB(PC,JTReason::DirectJump);
  BasicBlock::iterator it = Block->begin();
  BasicBlock::iterator end = Block->end();
  uint32_t count = 0;
  if(Block != nullptr){
    for(;it!=end;it++){
      if(it->getOpcode()==llvm::Instruction::Call){
	auto callI = dyn_cast<CallInst>(&*it);
	auto *Callee = callI->getCalledFunction();
	if(Callee != nullptr && Callee->getName() == "newpc"){
	    auto addr = getLimitedValue(callI->getArgOperand(0));
	    count++;
	    if(count>3)
	      return;
	    StaticAddrs[addr] = false;
            //errs()<<format_hex(pc,0)<<" <- No Crash point, to explore next addr.\n";
	}
      }
    }
  }
}

void JumpTargetManager::handleIndirectCall(llvm::BasicBlock *thisBlock, 
		uint64_t thisAddr, bool StaticFlag){
  IndirectBlocksMap::iterator it = IndirectCallBlocks.find(thisAddr);
  if(it != IndirectCallBlocks.end()){
    return;
  }
  IndirectCallBlocks[thisAddr] = 1;

  if(StaticFlag)
    return;
  

  uint32_t userCodeFlag = 0;
  uint32_t &userCodeFlag1 = userCodeFlag;

  // Contains indirect instruction's Block, it must have a store instruction.
  BasicBlock::iterator I = --thisBlock->end();
  if(dyn_cast<BranchInst>(I))
    return;
  errs()<<"indirect call&&&&&\n";
  I--; 
  auto store = dyn_cast<llvm::StoreInst>(--I);
  if(store){
    range = 0;
    // Seeking Value of assign to pc. 
    // eg:store i64 value, i64* @pc 
    NODETYPE nodetmp = nodepCFG;
    std::vector<llvm::Instruction *> DataFlow1;
    std::vector<llvm::Instruction *> &DataFlow = DataFlow1;
    getIllegalValueDFG(store->getValueOperand(),
		       dyn_cast<llvm::Instruction>(store),
		       thisBlock,
		       DataFlow,
		       InterprocessMode,
		       userCodeFlag1);
    errs()<<"Finished analysis indirect Inst access Data Flow!\n";
    nodepCFG = nodetmp;

    std::vector<legalValue> legalSet1;
    std::vector<legalValue> &legalSet = legalSet1;
    analysisLegalValue(DataFlow,legalSet);

    //Log information.
    for(auto set : legalSet){
      for(auto ii : set.I)
        errs()<<*ii<<" -------------";
      errs()<<"\n";
      for(auto vvv : set.value) 
        errs() <<*vvv<<" +++++++++++\n";
      
      errs()<<"\n";
    }
    //To match base+offset mode.
    bool isJmpTable = false;
    for(unsigned i=0; i<legalSet.size(); i++){
      if(legalSet[i].I[0]->getOpcode() == Instruction::Add){
        if(((i+1) < legalSet.size()) and 
	   (legalSet[i+1].I[0]->getOpcode() == Instruction::Shl)){
	    legalSet.back().value[0] = dyn_cast<Value>(legalSet.back().I[0]);
            legalSet.erase(legalSet.begin()+i+2,legalSet.end()-1);
	    isJmpTable = true;
	    break;
	}
      }
      for(unsigned j=0; j<legalSet[i].I.size(); j++){
        if(legalSet[i].I[j]->getOpcode() == Instruction::Add){
          if(((j+1) < legalSet[i].I.size()) and 
	     (legalSet[i].I[j+1]->getOpcode() == Instruction::Shl)){
            isJmpTable = true;
	    //revng_abort("Not implement!\n");
	    return;
	  }
        }       
      }
    }
    if(isJmpTable){ 
      //To assign a legal value
      for(uint64_t n = 0;;n++){
        auto addrConst = foldSet(legalSet,n);
	if(addrConst==nullptr)
	  break;
        auto integer = dyn_cast<ConstantInt>(addrConst);
	auto newaddr = integer->getZExtValue();
	if(newaddr==0)
            continue;
	if(isExecutableAddress(newaddr))
            harvestBTBasicBlock(thisBlock,thisAddr,newaddr);
        else
          break;
      }
      if(Statistics){
        IndirectBlocksMap::iterator it = CallTable.find(thisAddr);
        if(it == CallTable.end())
          CallTable[thisAddr] = 1;
      }
    }

  }
}

bool JumpTargetManager::isAccessMemInst(llvm::Instruction *I){
  BasicBlock::iterator it(I);
  BasicBlock::iterator end = I->getParent()->end();
  BasicBlock::iterator lastInst(I->getParent()->back());
  auto v = dyn_cast<llvm::Value>(I);
  if(I->getOpcode()==Instruction::Store){
    auto store = dyn_cast<llvm::StoreInst>(I);
    v = store->getPointerOperand();
  }

  it++;
  for(; it!=end; it++){ 
    switch(it->getOpcode()){
    case llvm::Instruction::IntToPtr:{
        auto inttoptrI = dyn_cast<Instruction>(it);
	if((inttoptrI->getOperand(0) - v) == 0)
	    return true;
	break;
    }
    case llvm::Instruction::Call:{
	auto callI = dyn_cast<CallInst>(&*it);
	auto *Callee = callI->getCalledFunction();
	if(Callee != nullptr && Callee->getName() == "newpc"){
            it = lastInst;
	    auto pc = getLimitedValue(callI->getArgOperand(0));
            errs()<<format_hex(pc,0)<<" <- No Crash point, to explore next addr.\n";
	}
	if(Callee != nullptr && (
			Callee->getName() == "helper_fldt_ST0"||
			Callee->getName() == "helper_fstt_ST0"||
			Callee->getName() == "helper_divq_EAX"||
			Callee->getName() == "helper_idivl_EAX"))
	    return true;
	break;
    }
    case llvm::Instruction::Load:{
        auto loadI = dyn_cast<llvm::LoadInst>(it);
	if((loadI->getPointerOperand() - v) == 0)
	    v = dyn_cast<Value>(it);
        break;
    }
    case llvm::Instruction::Store:{
        auto storeI = dyn_cast<llvm::StoreInst>(it);
        if((storeI->getValueOperand() - v) == 0)
	    v = storeI->getPointerOperand();
	break;
    }
    default:{
        auto instr = dyn_cast<Instruction>(it);
        for(Use &u : instr->operands()){
            Value *InstV = u.get();
            if((InstV - v) == 0)
	        v = dyn_cast<Value>(instr);
        }
    }
    }
  }
  return false;
}

uint32_t JumpTargetManager::REGLABLE(uint32_t RegOP){
  switch(RegOP){
      case RAX:
            return R_EAX;
          break;
      case RCX:
            return R_ECX;
          break;
      case RDX:
            return R_EDX;
          break;
      case RBX:
            return R_EBX;
          break;
      case RBP:{
	    //memset((void *)(ptc.regs[R_ESP]+8),0,1<12);
            return R_EBP;
          break;
      }
      case RSP:
            return R_ESP;
      case RSI:
            return R_ESI;
          break;
      case RDI:
            return R_EDI;
          break;
      case R8:
            return R_8;
          break;
      case R9:
            return R_9;
          break;
      case R10:
            return R_10;
          break;
      case R11:
            return R_11;
          break;
      case R12:
            return R_12;
          break;
      case R13:
            return R_13;
          break;
      case R14:
            return R_14;
          break;
      case R15:
            return R_15;
          break;
      default:
          return UndefineOP;
  }
}

std::pair<uint32_t,uint64_t> JumpTargetManager::getLastOperandandNextPC(llvm::Instruction *I){
  BasicBlock::iterator it(I);
  BasicBlock::iterator end = I->getParent()->end();
  uint32_t op = UndefineOP;
  uint64_t addr = 0;
  it++;
  for(; it!=end; it++){
    //if(it->getOpcode() == Instruction::Load){
    //  auto load = dyn_cast<LoadInst>(&*it);
    //  load->get
    //}
    if(it->getOpcode() == Instruction::Store){
      auto store = dyn_cast<StoreInst>(&*it);
      auto v = store->getPointerOperand();
      if(dyn_cast<Constant>(v)){
        StringRef name = v->getName();
        auto number = StrToInt(name.data());
        op = REGLABLE(number); 
      }
    }
    if(it->getOpcode() == Instruction::Call){
      auto call = dyn_cast<CallInst>(&*it);
      revng_assert(call);
      auto *Callee = call->getCalledFunction();
      if(Callee != nullptr && Callee->getName() == "newpc"){
        addr = getLimitedValue(call->getArgOperand(0));
        break;
      }
    }
  }
 
  return std::make_pair(op,addr);
}

uint64_t JumpTargetManager::getInstructionPC(llvm::Instruction *I){
  BasicBlock::reverse_iterator it(I);
  BasicBlock::reverse_iterator rend = I->getParent()->rend();

  for(; it!=rend; it++){
    auto callI = dyn_cast<CallInst>(&*it);
    if(callI){
        auto *Callee = callI->getCalledFunction();
	if(Callee != nullptr && Callee->getName() == "newpc"){
	    return getLimitedValue(callI->getArgOperand(0));
            //errs()<<format_hex(pc,0)<<" <- Crash Instruction Address.\n";
	}
        if(Callee != nullptr and Callee->getName() == "helper_raise_exception")
            return 0;
    }
  }
  return 0;
}

BasicBlock * JumpTargetManager::getSplitedBlock(llvm::BranchInst *branch){
  revng_assert(!branch->isConditional());
  auto bb = dyn_cast<BasicBlock>(branch->getOperand(0));
  auto call = dyn_cast<CallInst>(bb->begin());
  auto *Callee = call->getCalledFunction(); 
  if(Callee != nullptr && Callee->getName() == "newpc"){
    auto PC = getLimitedValue(call->getArgOperand(0));
    // This Crash instruction PC is the start address of this block.
    ToPurge.insert(bb);
    Unexplored.push_back(BlockWithAddress(PC, bb));
    return bb;
  }
  return nullptr;
}

uint64_t JumpTargetManager::handleIllegalMemoryAccess(llvm::BasicBlock *thisBlock,
		                                          uint64_t thisAddr,
                                                          size_t ConsumedSize){

  BasicBlock::iterator endInst = thisBlock->end();


  BasicBlock::iterator lastInst = endInst;
  lastInst--;

  auto PC = getInstructionPC(dyn_cast<Instruction>(lastInst));
  if(PC == thisAddr or PC == 0 or *ptc.isSyscall)
    return thisAddr+ConsumedSize;
  return PC;

//  //TODO: modify these
//  uint32_t userCodeFlag = 0;
//  uint32_t &userCodeFlag1 = userCodeFlag;
//  BasicBlock::iterator beginInst = thisBlock->begin();
//  BasicBlock::iterator endInst = thisBlock->end();
//  BasicBlock::iterator I = beginInst; 
//
//  if(*ptc.isIndirect || *ptc.isIndirectJmp || *ptc.isRet) 
//    return nullptr;
//
//  bool islegal = false;
//  uint32_t registerOP = 0; 
//
//  /* Instruction access memory type:
//   * case 1:  [reg + imm + imm] 10+1+1;
//   * case 2:  [reg + imm]       10+1
//   * case 3:  [reg + reg + imm] 10+10+1;
//   * case 4:  [reg + reg]       10+10 */
//  uint32_t accessNUM = 0;
//  BasicBlock::iterator lastInst = endInst;
//  lastInst--;
//  if(!dyn_cast<BranchInst>(lastInst)){
//    auto PC = getInstructionPC(dyn_cast<Instruction>(lastInst));
//    if(PC == thisAddr)
//      return nullptr;
//    return registerJT(PC,JTReason::GlobalData);
//  }
//  if(FAST){
//    return nullptr;
//  }
//
//  I = ++beginInst;
//  for(; I!=endInst; I++){
//    if(I->getOpcode() == Instruction::Load){
//        auto load = dyn_cast<llvm::LoadInst>(I);
//        Value *V = load->getPointerOperand();
//	if(dyn_cast<Constant>(V)){
//            std::tie(islegal,registerOP) = islegalAddr(V);
//	    if(!islegal and registerOP==RSP){
//	      haveBB = 1;
//	      return nullptr;
//	    }
//            if(registerOP != 0 &&
//               isAccessMemInst(dyn_cast<llvm::Instruction>(I)))
//                accessNUM = accessNUM+10;
//	}         
//    }
//    if(I->getOpcode() == Instruction::Store){
//        auto store = dyn_cast<llvm::StoreInst>(I);
//	Value *constV = store->getValueOperand();
//	auto imm = dyn_cast<ConstantInt>(constV); 
//	if(imm){
//	    if(isAccessMemInst(dyn_cast<llvm::Instruction>(I))){
//	      if(isExecutableAddress(imm->getZExtValue()))
//	        accessNUM = accessNUM+10;
//	      else
//		accessNUM = accessNUM+1;
//	    }
//	}
//    }
//    if(I->getOpcode() == Instruction::Call){
//        auto callI = dyn_cast<CallInst>(&*I);
//        auto *Callee = callI->getCalledFunction();
//        if(Callee != nullptr && Callee->getName() == "newpc"){
//          if(accessNUM > 11){
//            auto PC = getLimitedValue(callI->getArgOperand(0));
//	    revng_assert(PC != thisAddr);
//	    return registerJT(PC,JTReason::GlobalData);
//	  }
//          else
//            accessNUM = 0;
//        }
//    }
//  }
//  if(accessNUM > 11){
//    BasicBlock::iterator brI = endInst;
//    brI--;
//    auto branch = dyn_cast<BranchInst>(brI);
//    if(branch){
//      return getSplitedBlock(branch);
//    }
//  }
//
//  revng_assert(accessNUM < 12);
//  I = beginInst;
//  std::vector<llvm::Instruction *> DataFlow1;
//  std::vector<llvm::Instruction *> &DataFlow = DataFlow1; 
//  NODETYPE nodetmp = nodepCFG; 
//  for(;I!=endInst;I++){
//    // case 1: load instruction
//    if(I->getOpcode() == Instruction::Load){
//        errs()<<*I<<"         <-Load \n";
//        auto linst = dyn_cast<llvm::LoadInst>(I);
//        Value *v = linst->getPointerOperand();
//        std::tie(islegal,registerOP) = islegalAddr(v);
//        if(!islegal and isAccessMemInst(dyn_cast<llvm::Instruction>(I))){
//          if(registerOP == RSP){
//	    haveBB = 1;
//	    IllegalStaticAddrs.push_back(thisAddr);
//	    return nullptr;
//	  }
//          getIllegalValueDFG(v,dyn_cast<llvm::Instruction>(I),
//			  thisBlock,DataFlow,CrashMode,userCodeFlag1);
//          errs()<<"Finished analysis illegal access Data Flow!\n";
//          break;
//        }
//    }
//  }
//  nodepCFG = nodetmp;
//
//  // If crash point is not found, choosing one of branches to execute.  
//  if(I==endInst){
//      BasicBlock::iterator brI = endInst;
//      brI--;
//      auto branch = dyn_cast<BranchInst>(brI);
//      if(branch){ 
//	if(!branch->isConditional())
//          return getSplitedBlock(branch);
//	else{
//          auto bb = dyn_cast<BasicBlock>(brI->getOperand(1));
//          auto br = dyn_cast<BranchInst>(--bb->end());
//          while(br){
//            bb = dyn_cast<BasicBlock>(br->getOperand(0));
//            br = dyn_cast<BranchInst>(--bb->end());
//          }
//	  auto PC = getDestBRPCWrite(bb);
//	  revng_assert(PC != 0);
//	  auto block =  registerJT(PC,JTReason::GlobalData);
//	  if(haveBB){
//            //If chosen branch have been executed, setting havveBB=0, 
//	    // to harvest this Block next.
//	    haveBB = 0; 
//	    return nullptr;
//	  }
//	  else
//	    return block;
//	} 
//
//      } 
//  }
//
//  if(I==endInst){////////////////////////////////////////////
//	  errs()<<format_hex(ptc.regs[R_14],0)<<" r14\n";
//	  errs()<<format_hex(ptc.regs[R_15],0)<<" r15\n";
//	  errs()<<*thisBlock<<"\n";}//////////////////////////
//  //revng_assert(I!=endInst);
//
//  std::vector<legalValue> legalSet1;
//  std::vector<legalValue> &legalSet = legalSet1;
//  analysisLegalValue(DataFlow,legalSet);
//  //Log information.
//  for(auto set : legalSet){
//    for(auto ii : set.I)
//      errs()<<*ii<<" -------------";
//    errs()<<"\n";
//    for(auto vvv : set.value) 
//      errs() <<*vvv<<" +++++++++++\n";
//    
//    errs()<<"\n";
//  }
//
//  if(!legalSet.empty()){
//    auto lastSet = legalSet.back();
//    auto v = lastSet.value.front();  
//    auto constv = dyn_cast<ConstantInt>(v);
//    if(constv){
//      auto global = constv->getZExtValue();
//      if(isDataSegmAddr(global))
//          *((uint64_t *) global) = ptc.regs[R_ESP];
//    }
//  }
//
//  if(I!=endInst){
//    auto lable = REGLABLE(registerOP);
//    if(lable == UndefineOP)
//      revng_abort("Unkown register OP!\n");
//    ptc.regs[lable] = ptc.regs[R_ESP];
//  }
//
//  llvm::BasicBlock *Block = nullptr;
//  auto PC = getInstructionPC(dyn_cast<Instruction>(I));
//  revng_assert(isExecutableAddress(PC));
//  if(PC == thisAddr){
//      for(; I!=endInst; I++){
//        if(I->getOpcode() == Instruction::Call){
//          auto callI = dyn_cast<CallInst>(&*I);
//          auto *Callee = callI->getCalledFunction();
//          if(Callee != nullptr && Callee->getName() == "newpc"){
//            auto nextPC = getLimitedValue(callI->getArgOperand(0));
//	    return registerJT(nextPC,JTReason::GlobalData);
//	  }
//        } 
//      }
//      BasicBlock::iterator brI = endInst;
//      brI--;
//      auto branch = dyn_cast<BranchInst>(brI);
//      if(branch){
//        return getSplitedBlock(branch);
//      }
//      revng_assert(I != endInst);
//      // This Crash instruction PC is the start address of this block.
//      //ToPurge.insert(thisBlock);
//      //Unexplored.push_back(BlockWithAddress(thisAddr, thisBlock));
//      //Block = thisBlock;
//  }
//  else
//      Block = registerJT(PC,JTReason::GlobalData);
//
//  return Block;
}

void JumpTargetManager::handleIndirectJmp(llvm::BasicBlock *thisBlock, 
		                                 uint64_t thisAddr,
						 bool StaticFlag){
  uint32_t userCodeFlag = 0;
  uint32_t &userCodeFlag1 = userCodeFlag;
  IndirectJmpBlocks[thisAddr] = 1;

  if(SUPERFAST and StaticFlag)
    return;

  // Contains indirect instruction's Block, it must have a store instruction.
  BasicBlock::iterator I = --thisBlock->end(); 
  I--; 
  auto store = dyn_cast<llvm::StoreInst>(--I);
  if(store){
    range = 0;
    // Seeking Value of assign to pc. 
    // eg:store i64 value, i64* @pc 
    NODETYPE nodetmp = nodepCFG;
    std::vector<llvm::Instruction *> DataFlow1;
    std::vector<llvm::Instruction *> &DataFlow = DataFlow1;
    getIllegalValueDFG(store->getValueOperand(),
		       dyn_cast<llvm::Instruction>(store),
		       thisBlock,
		       DataFlow,
		       JumpTableMode,
		       userCodeFlag1);
    errs()<<"Finished analysis indirect Inst access Data Flow!\n";
    nodepCFG = nodetmp;

    std::vector<legalValue> legalSet1;
    std::vector<legalValue> &legalSet = legalSet1;
    analysisLegalValue(DataFlow,legalSet);

    //Log information.
    for(auto set : legalSet){
      for(auto ii : set.I)
        errs()<<*ii<<" -------------";
      errs()<<"\n";
      for(auto vvv : set.value) 
        errs() <<*vvv<<" +++++++++++\n";
      
      errs()<<"\n";
    }

    //To match base+offset mode.
    bool isJmpTable = false;
    for(unsigned i=0; i<legalSet.size(); i++){
      if(legalSet[i].I[0]->getOpcode() == Instruction::Add){
        if(((i+1) < legalSet.size()) and 
	   (legalSet[i+1].I[0]->getOpcode() == Instruction::Shl)){
	    legalSet.back().value[0] = dyn_cast<Value>(legalSet.back().I[0]);
            legalSet.erase(legalSet.begin()+i+2,legalSet.end()-1);
	    isJmpTable = true;
	    break;
	}
      }
      for(unsigned j=0; j<legalSet[i].I.size(); j++){
        if(legalSet[i].I[j]->getOpcode() == Instruction::Add){
          if(((j+1) < legalSet[i].I.size()) and 
	     (legalSet[i].I[j+1]->getOpcode() == Instruction::Shl)){
            isJmpTable = true;
	    return;
	   // revng_abort("Not implement!\n");
	  }
        }       
      }
    } 
    if(!isJmpTable){
      errs()<<"This indirect jmp is not jmp table type.\n";
      return;
    }
    
    if(Statistics){
      IndirectBlocksMap::iterator it = JmpTable.find(thisAddr);
      if(it == JmpTable.end())
	  JmpTable[thisAddr] = 1;
    }

    range = getLegalValueRange(thisBlock);
    errs()<<range<<" <---range\n";
    if(range == 0){
      //revng_abort("Not implement and 'range == 0'\n");
      for(uint64_t n = 0;;n++){
        auto addrConst = foldSet(legalSet,n);
	if(addrConst==nullptr)
	  break;
        auto integer = dyn_cast<ConstantInt>(addrConst);
	auto newaddr = integer->getZExtValue();
	if(newaddr==0)
            continue;
	if(isExecutableAddress(newaddr))
            harvestBTBasicBlock(thisBlock,thisAddr,newaddr);
        else
          break;
      }
      return;
    }
    // To assign a legal value
    for(uint64_t n = 0; n<=range; n++){
      auto addrConst = foldSet(legalSet,n);
      if(addrConst==nullptr)
        return;
      auto integer = dyn_cast<ConstantInt>(addrConst);
      harvestBTBasicBlock(thisBlock,thisAddr,integer->getZExtValue());
    }
  }
}

// Harvest branch target(destination) address
void JumpTargetManager::harvestBTBasicBlock(llvm::BasicBlock *thisBlock,
		                            uint64_t thisAddr,
					    uint64_t destAddr){
  for(auto item : BranchTargets){
    if(std::get<0>(item) == destAddr)
        return;
  }
  if(!haveTranslatedPC(destAddr, 0)){
      if(!isDataSegmAddr(ptc.regs[R_ESP]) and isDataSegmAddr(ptc.regs[R_EBP]))
          ptc.regs[R_ESP] = ptc.regs[R_EBP];
      //if(isDataSegmAddr(ptc.regs[R_ESP]) and !isDataSegmAddr(ptc.regs[R_EBP]))
      //    ptc.regs[R_EBP] = ptc.regs[R_ESP] + 256;
      if(!isDataSegmAddr(ptc.regs[R_ESP]) and !isDataSegmAddr(ptc.regs[R_EBP])){
          ptc.regs[R_ESP] = *ptc.ElfStartStack - 512;
          ptc.regs[R_EBP] = ptc.regs[R_ESP] + 256;
      }
      auto successed = ptc.storeCPUState();
      if(!successed)
        revng_abort("Store CPU stat failed!\n");
      /* Recording not execute branch destination relationship with current BasicBlock */
     // thisBlock = nullptr; 
      BranchTargets.push_back(std::make_tuple(destAddr,thisBlock,thisAddr)); 
      errs()<<format_hex(destAddr,0)<<" <- Jmp target add\n";
    }
  errs()<<"Branch targets total numbers: "<<BranchTargets.size()<<"\n";  
}

void JumpTargetManager::handleIllegalJumpAddress(llvm::BasicBlock *thisBlock,
		                                 uint64_t thisAddr){
  if(*ptc.isRet || *ptc.isIndirectJmp)
    return;
  if(FAST)
    return;  

  uint32_t userCodeFlag = 0;
  uint32_t &userCodeFlag1 = userCodeFlag;

  //Some bb may be splitted, so tracking to the end bb of splitted.
  auto br = dyn_cast<BranchInst>(--thisBlock->end());
  while(br){
    thisBlock = dyn_cast<BasicBlock>(br->getOperand(0));
    if(thisBlock==nullptr)
      return;
    br = dyn_cast<BranchInst>(--thisBlock->end());
  }
  // Emerge illegal next jump address, current Block must contain a indirect instruction!  
  BasicBlock::iterator I = --thisBlock->end(); 
  I--; 
  auto store = dyn_cast<llvm::StoreInst>(--I);
  if(store){
    range = 0;
    // Seeking Value of assign to pc. 
    // eg:store i64 value, i64* @pc 
    NODETYPE nodetmp = nodepCFG;
    std::vector<llvm::Instruction *> DataFlow1;
    std::vector<llvm::Instruction *> &DataFlow = DataFlow1;
    getIllegalValueDFG(store->getValueOperand(),
		       dyn_cast<llvm::Instruction>(store),
		       thisBlock,DataFlow,FullMode,userCodeFlag1);
    errs()<<"Finished analysis illegal jump Data Flow!\n";
    nodepCFG = nodetmp;

    std::vector<legalValue> legalSet1;
    std::vector<legalValue> &legalSet = legalSet1;
    analysisLegalValue(DataFlow,legalSet);

   // if(*ptc.isIndirectJmp)
   //   range = getLegalValueRange(thisBlock);

    for(auto set : legalSet){
      for(auto ii : set.I)
        errs()<<*ii<<" -------------";
      errs()<<"\n";
      for(auto vvv : set.value) 
        errs() <<*vvv<<" +++++++++++\n";
      
      errs()<<"\n";
    }

    //Determine whether is dead code, eg: call 0
    if(legalSet.size() == 1){  
      errs()<<"This jump address is a dead code.\n";
      return;
    }

    // To assign a legal value
    auto addrConst = foldSet(legalSet,0);
    if(addrConst == nullptr)
        return;
    auto integer = dyn_cast<ConstantInt>(addrConst);
    harvestBTBasicBlock(thisBlock,thisAddr,integer->getZExtValue());
  }
}

uint32_t JumpTargetManager::getLegalValueRange(llvm::BasicBlock *thisBlock){
  llvm::Function::iterator nodeBB(thisBlock);
  llvm::Function::iterator begin(thisBlock->getParent()->begin());

  llvm::BasicBlock *rangeBB = nullptr;
  std::map<llvm::BasicBlock *, llvm::BasicBlock *> DoneOFPath1;
  std::map<llvm::BasicBlock *, llvm::BasicBlock *> &DoneOFPath = DoneOFPath1;
  DoneOFPath[std::get<0>(nodepCFG)] = std::get<1>(nodepCFG); 
  // We set a backtrack window to control the loop. 
  for(int i=0; i<20; i++){
    auto bb = dyn_cast<llvm::BasicBlock>(nodeBB);
    BasicBlock::iterator I = --(bb->end());
    if(auto branch = dyn_cast<BranchInst>(I)){
      if(branch->isConditional()){
	  rangeBB = bb;
          break;
      }
    }

    if((std::get<0>(nodepCFG) - bb) == 0){
      bb = std::get<1>(nodepCFG);
      llvm::Function::iterator it(bb);
      // Handle split Block
      nodeBB = it;
      searchpartCFG(DoneOFPath);  
      while(true){
        auto I = --(bb->end());
	auto branch = dyn_cast<BranchInst>(I);
	if(branch && !branch->isConditional())
	  bb = dyn_cast<BasicBlock>(branch->getOperand(0));
	else
          break;
      }
      auto endI = --(bb->end()); 
      if(auto branch = dyn_cast<BranchInst>(endI)){
        if(branch->isConditional()){
            rangeBB = bb;
	    break;
	}
      }
    }
    nodeBB--;
  }

  if(rangeBB==nullptr)
    return 0;

  BasicBlock::iterator I = --rangeBB->end();
  auto br = dyn_cast<BranchInst>(I);
  auto cmp = dyn_cast<ICmpInst>(br->getCondition());
  revng_assert(cmp,"That should a cmp instruction!");
  CmpInst::Predicate p = cmp->getPredicate(); 
  if(p == CmpInst::ICMP_EQ || p == CmpInst::ICMP_NE){
    *ptc.isIndirectJmp = 0;
    return 0;
  }

  uint32_t userFlag = 1;
  uint32_t &userFlag1 = userFlag;
  std::vector<llvm::Instruction *> DataFlow1;
  std::vector<llvm::Instruction *> &DataFlow = DataFlow1;
  getIllegalValueDFG(br->getCondition(),
		     dyn_cast<llvm::Instruction>(br),
		     rangeBB,
		     DataFlow,
		     RangeMode,userFlag1);

  std::vector<legalValue> legalSet1;
  std::vector<legalValue> &legalSet = legalSet1;
  analysisLegalValue(DataFlow,legalSet);

  //Log information:
  for(auto set : legalSet){
    for(auto ii : set.I)
      errs()<<*ii<<" -------------";
    errs()<<"\n";
    for(auto vvv : set.value) 
      errs() <<*vvv<<" +++++++++++\n";
    
    errs()<<"\n";
  } 

//  //Determine if there have a range.
//  //If all values are constant, there is no range. 
//  for(auto set : legalSet){
//    for(auto value : set.value){
//      auto constant = dyn_cast<ConstantInt>(value);
//      if(constant == nullptr)
//        goto go_on;   
//    }
//  }
//  return 0;

  bool firstConst = true;
  for(auto first : legalSet.front().value){
    auto constant = dyn_cast<ConstantInt>(first);
    if(constant==nullptr)
      firstConst = false;
  }
  if(firstConst){
    if(legalSet.front().value.size() == 1){
      auto constant = dyn_cast<ConstantInt>(legalSet.front().value.front());
      revng_assert(constant,"That should a constant value!\n");
      auto n = constant->getZExtValue();
      return n;
    }
    else{
      revng_abort("To do more implement!\n");
      //foldstack();
      //return n;
    }
  }
  revng_abort("TODO more implement!\n");
  //firstConst ==  false;
  //foldSet(legalSet);
  //return n;
}

void JumpTargetManager::getIllegalValueDFG(llvm::Value *v,
		llvm::Instruction *I,
		llvm::BasicBlock *thisBlock,
		std::vector<llvm::Instruction *> &DataFlow,
		TrackbackMode TrackType,
		uint32_t &userCodeFlag){
  llvm::User *operateUser = nullptr;
  llvm::Value *v1 = nullptr;
  LastAssignmentResult result;
  llvm::Instruction *lastInst = nullptr;
  std::vector<std::tuple<llvm::Value *,llvm::User *,llvm::BasicBlock *,NODETYPE>> vs;
  vs.push_back(std::make_tuple(v,dyn_cast<User>(I),thisBlock,nodepCFG));
  DataFlow.push_back(I);

  uint32_t NUMOFCONST1 = 0;
  uint32_t &NUMOFCONST = NUMOFCONST1;
  uint32_t NextValueNums = 0;
  if(TrackType==CrashMode){
    NextValueNums = 20;
  }
  if(TrackType==JumpTableMode)
    NUMOFCONST = 5;
  if(TrackType==InterprocessMode){
    NextValueNums = 50; // TODO: optimization parameters
    NUMOFCONST = 5;
  }
  if(TrackType==TestMode)
    NUMOFCONST = 30;

  std::map<llvm::BasicBlock *, llvm::BasicBlock *> DoneOFPath1;
  std::map<llvm::BasicBlock *, llvm::BasicBlock *> &DoneOFPath = DoneOFPath1;
  // Get illegal access Value's DFG. 
  while(!vs.empty()){
    llvm::BasicBlock *tmpB = nullptr;
    std::tie(v1,operateUser,tmpB,nodepCFG) = vs.back();
    DoneOFPath.clear();
    DoneOFPath[std::get<0>(nodepCFG)] = std::get<1>(nodepCFG);

    llvm::Function::iterator nodeBB(tmpB);
    llvm::Function::iterator begin(tmpB->getParent()->begin());
    vs.pop_back();

    for(;nodeBB != begin;){  
      auto bb = dyn_cast<llvm::BasicBlock>(nodeBB);
      if(v1->isUsedInBasicBlock(bb)){
	//Determine whether bb belongs to user code section 
	//userCodeFlag = belongToUBlock(bb);
	userCodeFlag = 1;
        std::tie(result,lastInst) = getLastAssignment(v1,operateUser,bb,TrackType,NUMOFCONST);
        switch(result)
        {
          case CurrentBlockValueDef:
          {
              if(lastInst->getOpcode() == Instruction::Select){
                auto select = dyn_cast<llvm::SelectInst>(lastInst);
                v1 = select->getTrueValue();
                vs.push_back(std::make_tuple(select->getFalseValue(),
					dyn_cast<User>(lastInst),bb,nodepCFG));
              }
              else{
                auto nums = lastInst->getNumOperands();
                for(Use &lastU : lastInst->operands()){
                  Value *lastv = lastU.get();
                  vs.push_back(std::make_tuple(lastv,dyn_cast<User>(lastInst),bb,nodepCFG));
                }
                v1 = std::get<0>(vs[vs.size()-nums]);
                vs.erase(vs.begin()+vs.size()-nums);
              }
              DataFlow.push_back(lastInst);
              operateUser = dyn_cast<User>(lastInst);
              nodeBB++;
              break;
          }
          case NextBlockOperating:
          {
              // Judge current BasickBlcok whether reaching partCFG's node
              // if ture, to research partCFG stack and update node 
              if((std::get<0>(nodepCFG) - bb) == 0){
		uint32_t num = 0;
                auto callBB = std::get<1>(nodepCFG);
                auto brJT = dyn_cast<BranchInst>(--(callBB->end()));
		if(brJT){
        	  if(brJT->isConditional() and *ptc.isIndirectJmp){
	            nodeBB = begin;
	            continue;
	          }
		  std::vector<Value *> brNum;
		  brNum.push_back(dyn_cast<Value>(brJT));
		  while(!brNum.empty()){
		    auto br = dyn_cast<BranchInst>(brNum.back());
		    brNum.pop_back();
		    if(br and br->isUnconditional()){
		      // TODO:br->Operands() 
		      auto labelB = dyn_cast<BasicBlock>(br->getOperand(0));
		      brNum.push_back(dyn_cast<Value>(--(labelB->end())));
		      num++;
		    }
		  }
             	}
                llvm::Function::iterator it(std::get<1>(nodepCFG));
                nodeBB = it;
		for(;num>0;num--)
		  nodeBB++;
                searchpartCFG(DoneOFPath);
                continue;
              }
              break;
          }
          case CurrentBlockLastAssign:
          {
              // Only Store instruction can assign a value for Value rather than defined
              auto store = dyn_cast<llvm::StoreInst>(lastInst);
              v1 = store->getValueOperand();
              DataFlow.push_back(lastInst);
              operateUser = dyn_cast<User>(lastInst);
              nodeBB++;
              break;
          }
          case ConstantValueAssign:
              goto NextValue;
          break;
          case UnknowResult:
              revng_abort("Unknow of result!");
          break;
        }
        
      }///?if(v1->isUsedInBasicBlock(bb))?
      else{
        if((std::get<0>(nodepCFG) - bb) == 0){
          uint32_t num = 0;
          auto callBB = std::get<1>(nodepCFG);
          auto brJT = dyn_cast<BranchInst>(--(callBB->end()));
          if(brJT){
	    if(brJT->isConditional() and *ptc.isIndirectJmp){
	      nodeBB = begin;
	      continue;
	    }
            std::vector<Value *> brNum;
            brNum.push_back(dyn_cast<Value>(brJT));
            while(!brNum.empty()){
              auto br = dyn_cast<BranchInst>(brNum.back());
              brNum.pop_back();
              if(br and br->isUnconditional()){
      		// TODO:br->Operands() 
                auto labelB = dyn_cast<BasicBlock>(br->getOperand(0));
                brNum.push_back(dyn_cast<Value>(--(labelB->end())));
                num++;
              }
            }
          }
          llvm::Function::iterator it(std::get<1>(nodepCFG));
          nodeBB = it;
          for(;num>0;num--)
            nodeBB++;
           
	  searchpartCFG(DoneOFPath);
          continue;
         }        
      }
      nodeBB--;
    }///?for(;nodeBB != begin;)?
NextValue:
    errs()<<"Explore next Value of Value of DFG!\n";
    if(TrackType==JumpTableMode)
      NUMOFCONST = 5;
    if(TrackType==InterprocessMode){
      NUMOFCONST = 1;
      NextValueNums--;
      if(NextValueNums==0)
        return;
    }
    if(TrackType==CrashMode){
      //TrackType = RangeMode;
      NextValueNums--;
      if(NextValueNums==0)
        return;
    }
    continue;
  }///?while(!vs.empty())?
}

void JumpTargetManager::analysisLegalValue(std::vector<llvm::Instruction *> &DataFlow,
		std::vector<legalValue> &legalSet){
  if(DataFlow.empty())
    return;

  legalValue *relatedInstPtr = nullptr;
  legalValue *&relatedInstPtr1 = relatedInstPtr;

  llvm::Instruction *next = nullptr;
  for(unsigned i = 0; i < DataFlow.size(); i++){
    if(i == (DataFlow.size() - 1))
        next = nullptr;
    else
        next = DataFlow[i+1]; 
    unsigned Opcode = DataFlow[i]->getOpcode();
    switch(Opcode){
        case Instruction::Load:
	case Instruction::Store:
            handleMemoryAccess(DataFlow[i],next,legalSet,relatedInstPtr1);
        break;
	case Instruction::Select:
	    handleSelectOperation(DataFlow[i],next,legalSet,relatedInstPtr1);
	break;
	case Instruction::Add:
	case Instruction::Sub:
	case Instruction::And:
	case Instruction::Shl:
	case Instruction::AShr:
	case Instruction::LShr:
	case Instruction::Or:
	case Instruction::Xor:
	case Instruction::ICmp:
	case Instruction::Mul:
	    handleBinaryOperation(DataFlow[i],next,legalSet,relatedInstPtr1);
	break;
	//case llvm::Instruction::ICmp:
        case llvm::Instruction::IntToPtr:
	    handleConversionOperations(DataFlow[i],legalSet,relatedInstPtr1);
	break;
        case llvm::Instruction::ZExt:
	case llvm::Instruction::SExt:
	case llvm::Instruction::Trunc:
	case llvm::Instruction::Br:
	case llvm::Instruction::Call:
	break;
	default:
            errs()<<*DataFlow[i];
	    revng_abort("Unknow of instruction!");
	break;
    }
  }
}

llvm::Constant *JumpTargetManager::foldSet(std::vector<legalValue> &legalSet, uint64_t n){
  const DataLayout &DL = TheModule.getDataLayout();
  Constant *base = nullptr;
  //TODO:Fold Set instruction
  for(auto set : make_range(legalSet.rbegin(),legalSet.rend())){
    auto op = set.I[0]->getOpcode();
    if(op==Instruction::Add){
      auto RegConst = dyn_cast<ConstantInt>(set.value[0]);
      if(RegConst == nullptr){
        auto registerOP = StrToInt(set.value[0]->getName().data());
	if(registerOP==RSP)
	  return nullptr;
	auto lable  = REGLABLE(registerOP);
	if(lable==UndefineOP)
	  return nullptr;
        auto first =  ConstantInt::get(Type::getInt64Ty(Context),ptc.regs[lable]);
        set.value[0] = dyn_cast<Value>(first); 
      }
    }
    if(set.I.size()>1 and op!=Instruction::Add)
      return nullptr;
    
    switch(op){
        case Instruction::Load:
	case Instruction::Store:
	{
          auto constant = dyn_cast<ConstantInt>(set.value[0]);
	  if(constant){
	    //uint64_t address = constant->getZExtValue(); 
	    //auto newoperand = ConstantInt::get(set.I[0]->getType(),address);
	    base = dyn_cast<Constant>(set.value[0]);
	  }
	  else
            base = ConstantInt::get(Type::getInt64Ty(Context),n);
	  break;
	}
	case Instruction::Select:
	//TODO:later
	break;
	case Instruction::Mul:
	{
	  //TODO modifying later. x = a*b
	  auto integer1 = dyn_cast<ConstantInt>(set.value[0]);
	  if(integer1 == nullptr)
	    return nullptr;
	  uint64_t a = integer1->getZExtValue();

          auto integer2 = dyn_cast<ConstantInt>(base);
	  uint64_t b = integer2->getZExtValue();
	  uint64_t x = a*b;
	  base = ConstantInt::get(Type::getInt64Ty(Context),x);
	  break;
	}
	case Instruction::And:
	case Instruction::Sub:
	case Instruction::Add:
	case Instruction::LShr:
	case Instruction::AShr:
	case Instruction::Or:
	case Instruction::Shl:
	{
          auto integer = dyn_cast<ConstantInt>(set.value[0]);
	  if(integer == nullptr)
	    return nullptr;
         
	  Constant *op2 = dyn_cast<Constant>(set.value[0]);
          op2 = ConstantExpr::getTruncOrBitCast(op2,set.I[0]->getOperand(1)->getType());
          base = ConstantExpr::getTruncOrBitCast(base,set.I[0]->getOperand(0)->getType());
          base = ConstantFoldBinaryOpOperands(op,base,op2,DL);
	  break;
	}
        case llvm::Instruction::IntToPtr:
	{
	  //auto inttoptr = dyn_cast<IntToPtrInst>(set.I[0]);
	  auto integer = dyn_cast<ConstantInt>(base);
	  uint64_t address = integer->getZExtValue();
	  if(!ptc.isValidExecuteAddr(address)){
	    errs()<<"\nYielding an illegal addrress\n";
	    continue;
	  }
	  uint64_t addr = *((uint64_t *)address);
	  base = ConstantInt::get(base->getType(),addr);
          break;
	}
        default:
	    errs()<<*set.I[0]<<"\n";
            revng_abort("Unknow fold instruction!");
        break;	    
    }/// end switch(...
  }/// end for(auto..
  return base;
}

/* TODO: To assign a value  
 * According binary executing memory and CPU States setting */
llvm::Value *JumpTargetManager::payBinaryValue(llvm::Value *v){
  errs()<<"\n"<<*v<<"\n\n";
  llvm::Type *Int64 = IntegerType::get(TheModule.getContext(),64);
  uint64_t Address = ptc.regs[R_ESP];
  Constant *probableValue = ConstantInt::get(Int64,Address);
  v = dyn_cast<Value>(probableValue);
  errs()<<"\n"<<*v<<"\n\n";  

  return v;
}

// To fold Instruction stack and to assign Value to'global variable'.
void JumpTargetManager::foldStack(legalValue *&relatedInstPtr){
  const DataLayout &DL = TheModule.getDataLayout();

  while(true){
    Value *last = relatedInstPtr->value.back();
    Value *secondlast = *(relatedInstPtr->value.end()-2);

    if(dyn_cast<Constant>(last) and dyn_cast<Constant>(secondlast)){
      if(dyn_cast<ConstantInt>(last) == nullptr){
        last = payBinaryValue(last);
      }
      if(dyn_cast<ConstantInt>(secondlast) == nullptr){
        secondlast = payBinaryValue(secondlast);
      }
      // Fold binary instruction
      Instruction *Inst = relatedInstPtr->I.back();

      if(Inst->getOpcode()==Instruction::Select){ 
        auto base = secondlast;
        errs()<<*base<<" <-encount Select instruction add to base\n";
        relatedInstPtr->value.erase(relatedInstPtr->value.end()-2);
        relatedInstPtr->I.pop_back();
        break;
      }     
      // TODO: To loop base until base equal to 0
      Constant *op1 = dyn_cast<Constant>(last);
      Constant *op2 = dyn_cast<Constant>(secondlast);
      op1 = ConstantExpr::getTruncOrBitCast(op1,Inst->getOperand(0)->getType());
      op2 = ConstantExpr::getTruncOrBitCast(op2,Inst->getOperand(1)->getType());
      Constant *NewOperand = ConstantFoldBinaryOpOperands(Inst->getOpcode(),op1,op2,DL);

      relatedInstPtr->value.erase(relatedInstPtr->value.end()-2,relatedInstPtr->value.end());
      relatedInstPtr->value.push_back(dyn_cast<Value>(NewOperand));
      relatedInstPtr->I.pop_back();
    }
    else 
      break;  
  }  
}

void JumpTargetManager::set2ptr(llvm::Instruction *next,
                                std::vector<legalValue> &legalSet,
                                legalValue *&relatedInstPtr){
  for(unsigned i = 0;i<legalSet.size();i++){
    for(unsigned v = 0; v<legalSet[i].value.size(); v++){
      if(isCorrelationWithNext(legalSet[i].value[v],next)){
        legalSet[i].value.erase(legalSet[i].value.begin()+v);
        relatedInstPtr = &legalSet[i]; 
      }
    }
  }
}

void JumpTargetManager::handleMemoryAccess(llvm::Instruction *current, 
                                           llvm::Instruction *next,
                                           std::vector<legalValue> &legalSet,
                                           legalValue *&relatedInstPtr){
  auto loadI = dyn_cast<llvm::LoadInst>(current);
  auto storeI = dyn_cast<llvm::StoreInst>(current);
  Value *v = nullptr;

  if(loadI)
    v = loadI->getPointerOperand();
  else if(storeI)
    v = storeI->getValueOperand();

  if(!isCorrelationWithNext(v, next)){
    /* Reduct Data flow instructions to Value stack and Instruction stack */
    if(relatedInstPtr){
      relatedInstPtr->value.push_back(v); 
//      auto num = relatedInstPtr->value.size(); 
//      if(num>1){  
//        auto last = dyn_cast<Constant>(relatedInstPtr->value[num-1]);
//        auto secondlast = dyn_cast<Constant>(relatedInstPtr->value[num-2]);
//        if(last and secondlast)
//          foldStack(relatedInstPtr);
//      }
    }
    else 
      legalSet.emplace_back(PushTemple(v),PushTemple(current));

    //Find out value that is related with unrelated Inst.
    set2ptr(next,legalSet,relatedInstPtr);
  }
}

void JumpTargetManager::handleConversionOperations(llvm::Instruction *current,
		                                   std::vector<legalValue> &legalSet,
						   legalValue *&relatedInstPtr){
  if(relatedInstPtr){
    //relatedInstPtr->value.push_back(current->getOperand(0));
    relatedInstPtr->I.push_back(current);
    return;
  }  

  legalSet.emplace_back(PushTemple(current));
}

void JumpTargetManager::handleSelectOperation(llvm::Instruction *current, 
                                              llvm::Instruction *next,
                                              std::vector<legalValue> &legalSet, 
                                              legalValue *&relatedInstPtr){
  auto selectI = dyn_cast<llvm::SelectInst>(current);
  
  if(relatedInstPtr){
    if(dyn_cast<ConstantInt>(selectI->getFalseValue()) == nullptr)
      relatedInstPtr->value.push_back(selectI->getFalseValue());

    relatedInstPtr->I.push_back(current);
    return;
  }

  // Because we have pushed FalseValue, so TrueValue must be correlation.
  revng_assert(!isCorrelationWithNext(selectI->getFalseValue(), next),"That's wrong!");
  legalSet.emplace_back(PushTemple(selectI->getFalseValue()),PushTemple(current));
   //selectI->getTrueValue(); 
}

void JumpTargetManager::handleBinaryOperation(llvm::Instruction *current, 
                                              llvm::Instruction *next,
                                              std::vector<legalValue> &legalSet, 
                                              legalValue *&relatedInstPtr){
  Value *firstOp = current->getOperand(0);
  Value *secondOp = current->getOperand(1);
  bool first = isCorrelationWithNext(firstOp, next);
  bool second = isCorrelationWithNext(secondOp, next);

  if(relatedInstPtr){
    auto v = first ? secondOp:firstOp;
    if(dyn_cast<ConstantInt>(v) == nullptr)
      relatedInstPtr->value.push_back(v);

    relatedInstPtr->I.push_back(current);
    return;
  }

  if(first){
    legalSet.emplace_back(PushTemple(secondOp),PushTemple(current)); 
  }
  else if(second){
    legalSet.emplace_back(PushTemple(firstOp),PushTemple(current));
  }
  else{ 
    errs()<<*next<<"\n";
    revng_abort("Must one Value has correlation!");
  }
  revng_assert(((dyn_cast<Constant>(firstOp)&&dyn_cast<Constant>(secondOp)) != 1),"That's unnormal Inst!");
}

bool JumpTargetManager::isCorrelationWithNext(llvm::Value *preValue, llvm::Instruction *Inst){
  if(Inst==nullptr)
    return 0;

  if(auto storeI = dyn_cast<llvm::StoreInst>(Inst)){
    if((storeI->getPointerOperand() - preValue) == 0)
          return 1;
  }
  else{
    auto v = dyn_cast<llvm::Value>(Inst);
    if((v - preValue) == 0)
          return 1;
  }

  return 0;
}

uint32_t JumpTargetManager::StrToInt(const char *str){
  auto len = strlen(str);
  uint32_t dest = 0;
  if(len==3)
    dest =  str[1]*1000+str[2];
  else if(len==2)
    dest = str[1]*1000;

  return dest;
}

void JumpTargetManager::clearRegs(){
  //reserve function calling conventions regs
  ptc.regs[R_EAX] = 0;
  ptc.regs[R_EBX] = 0;
  ptc.regs[R_10] = 0;
  ptc.regs[R_11] = 0;
  ptc.regs[R_12] = 0;
  ptc.regs[R_13] = 0;
  ptc.regs[R_14] = 0;
  ptc.regs[R_15] = 0;
}

void JumpTargetManager::harvestCallBasicBlock(llvm::BasicBlock *thisBlock,uint64_t thisAddr){
  if(Statistics){
    IndirectBlocksMap::iterator it = CallBranches.find(*ptc.CallNext);
    if(it == CallBranches.end())
      CallBranches[*ptc.CallNext] = 1;
  }
//  if(!haveTranslatedPC(*ptc.CallNext, 0)){
//      StaticAddrs[*ptc.CallNext] = 2;
//      errs()<<format_hex(*ptc.CallNext,0)<<"  147852369\n";
//  }
  for(auto item : BranchTargets){
    if(std::get<0>(item) == *ptc.CallNext)
        return;
  }

  if(!haveTranslatedPC(*ptc.CallNext, 0)){
      /* Construct a state that have executed a call to next instruction of CPU state */
      ptc.regs[R_ESP] = ptc.regs[R_ESP] + 8;
      auto success  = ptc.storeCPUState();
      if(!success)
        revng_abort("Store CPU stat failed!\n");
      
      // Recover stack state
      ptc.regs[R_ESP] = ptc.regs[R_ESP] - 8;

      /* Recording not execute branch destination relationship with current BasicBlock */
      /* If we rewrite a Block that instructions of part have been rewritten, 
       * this Block ends this rewrite and add a br to jump to already existing Block,
       * So,this Block will not contain a call instruction, that has been splited
       * but we still record this relationship, because when we backtracking,
       * we will check splited Block. */ 
      BranchTargets.push_back(std::make_tuple(*ptc.CallNext,thisBlock,thisAddr));
      errs()<<format_hex(*ptc.CallNext,0)<<" <- Call next target add\n";
    }
  errs()<<"Branch targets total numbers: "<<BranchTargets.size()<<"\n";  
}

void JumpTargetManager::harvestbranchBasicBlock(uint64_t nextAddr,
       uint64_t thisAddr,	
       llvm::BasicBlock *thisBlock, 
       uint32_t size, 
       std::map<std::string, llvm::BasicBlock *> &branchlabeledBasicBlock){
  std::map<uint64_t, llvm::BasicBlock *> branchJT;

  // case 1: New block is belong to part of original block, so to split
  //         original block and occure a unconditional branch.
  //     eg:   size  >= 2
  //           label = 1

  // case 2: New block have a conditional branch, and 
  //         contains mutiple label.
  //     eg:   size  >= 2
  //           label >= 2    
  //outs()<<"next  "<<format_hex(nextAddr,0)<<"\n"; 
  BasicBlock::iterator I = --(thisBlock->end());
  
  if(auto branch = dyn_cast<BranchInst>(I)){
    if(branch->isConditional()){
      //outs()<<*I<<"\n";
      revng_assert(size==branchlabeledBasicBlock.size(),
                   "This br block should have many labels!");
      for(auto pair : branchlabeledBasicBlock){
        if(getDestBRPCWrite(pair.second)){
          branchJT[getDestBRPCWrite(pair.second)] = pair.second;
        }   
      } 
    }
  }

  if(!branchJT.empty()){
    //revng_assert(branchJT.size()==2,"There should have tow jump targets!");
    //If there have one jump target, return it. 
    if(branchJT.size() < 2)
      return;
    if(Statistics){
      IndirectBlocksMap::iterator it = CondBranches.find(thisAddr);
      if(it == CondBranches.end())
	  CondBranches[thisAddr] = 1;
    }
    for (auto destAddrSrcBB : branchJT){
      if(!haveTranslatedPC(destAddrSrcBB.first, nextAddr) && 
		      !isIllegalStaticAddr(destAddrSrcBB.first)){
	bool isRecord = false;
	for(auto item : BranchTargets){
	  if(std::get<0>(item) == destAddrSrcBB.first){
            isRecord = true;
	    break;
	  }
	}
	if(!isRecord){
          /* Recording current CPU state */
          if(!isDataSegmAddr(ptc.regs[R_ESP]) and isDataSegmAddr(ptc.regs[R_EBP]))
              ptc.regs[R_ESP] = ptc.regs[R_EBP];
	  //if(isDataSegmAddr(ptc.regs[R_ESP]) and !isDataSegmAddr(ptc.regs[R_EBP]))
	  //    ptc.regs[R_EBP] = ptc.regs[R_ESP] + 256;
          if(!isDataSegmAddr(ptc.regs[R_ESP]) and !isDataSegmAddr(ptc.regs[R_EBP])){
	      ptc.regs[R_ESP] = *ptc.ElfStartStack - 512;
	      ptc.regs[R_EBP] = ptc.regs[R_ESP] + 256;
          }
          auto success = ptc.storeCPUState();
	  if(!success)
	    revng_abort("Store CPU stat failed!\n");
	  
          /* Recording not execute branch destination relationship 
	   * with current BasicBlock and address */ 
          BranchTargets.push_back(std::make_tuple(
				destAddrSrcBB.first,
				//destAddrSrcBB.second,
				thisBlock,
				thisAddr
				)); 
          errs()<<format_hex(destAddrSrcBB.first,0)<<" <- Jmp target add\n";
        }  
      }
    }
    errs()<<"Branch targets total numbers: "<<BranchTargets.size()<<" \n"; 
  }

}

int64_t JumpTargetManager::getDestBRPCWrite(llvm::BasicBlock *block) {
  BasicBlock::iterator current(block->end());
  BasicBlock::iterator Begin(block->begin());
  while(current!=Begin) {
    current--;
    auto Store = dyn_cast<StoreInst>(current);
    if(Store){
      auto constantvalue = dyn_cast<ConstantInt>(Store->getValueOperand());
      if(constantvalue){
        auto pc = constantvalue->getSExtValue();
	if(isExecutableAddress(pc) and isInstructionAligned(pc))  
          return pc;
      }
    }
  }
  return 0;
}


bool JumpTargetManager::haveTranslatedPC(uint64_t pc, uint64_t next){
  if(!isExecutableAddress(pc) || !isInstructionAligned(pc))
    return 1;
  if(pc == next)
    return 1;
  // Do we already have a BasicBlock for this pc?
  BlockMap::iterator TargetIt = JumpTargets.find(pc);
  if(TargetIt != JumpTargets.end()) {
    return 1;
  }

  return 0;
}

using BlockWithAddress = JumpTargetManager::BlockWithAddress;
using JTM = JumpTargetManager;
const BlockWithAddress JTM::NoMoreTargets = BlockWithAddress(0, nullptr);
