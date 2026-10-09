# Real AddressSanitizer for Numba JIT code

An experimental, separately installable package that runs LLVM's **actual
AddressSanitizer pass** on Numba CPU JIT modules. It is not a bounds-checking
emulation and does not require a Numba or llvmlite fork.

Unchecked Numba array accesses can read or write outside allocated memory.
Python's usual memory-safety guarantees do not apply to those native accesses.
The Numba option is spelled `boundscheck`, not `bounds_check`.

**Status:** working proof of concept, not production-ready or comprehensive
memory-safety coverage. Tested on Linux x86-64, CPython 3.12.3, NumPy 2.3.5,
Numba 0.62.1, llvmlite 0.45.1 (embedded LLVM 20.1.8), and external LLVM 20.1.2.
The Python compiler dependencies are deliberately pinned because this package
uses private Numba hooks. Other versions/platforms need validation.

## Quick start

On Ubuntu 24.04 with LLVM 20 packages available:

```sh
sudo apt-get install llvm-20-dev clang-20 libclang-rt-20-dev
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/numba-asan build
.venv/bin/numba-asan run examples/safe.py
.venv/bin/numba-asan run examples/heap_overflow.py
```

The overflow example intentionally terminates with a nonzero exit code:

```text
ERROR: AddressSanitizer: heap-buffer-overflow
WRITE of size 8
... is located 0 bytes after 8192-byte region ...
Shadow bytes around the buggy address:
```

Run your own application or test module:

```sh
.venv/bin/numba-asan run /path/to/application.py argument1
.venv/bin/numba-asan run -- -m unittest discover -s /path/to/tests
# If pytest is installed in this environment:
.venv/bin/numba-asan run -- -m pytest /path/to/tests
```

The default plugin location is relative to the current directory. For use
outside this project, supply the absolute path:

```sh
/path/to/.venv/bin/numba-asan run \
    --plugin /path/to/numba-jit-asan/.build/numba_asan_pass.so \
    /path/to/application.py
```

Enable this at **process startup**, before application code imports Numba.
Merely setting `LD_PRELOAD` to ASan does **not** instrument JIT instructions;
the regression tests demonstrate that distinction.

Do not use this on production workloads. Sanitizer runtimes are debugging
tools, introduce overhead, and are not designed as a security hardening layer.
Do not run intentionally corrupting examples without the sanitizer launcher.

## What is implemented

1. The launcher re-executes Python with the matching Clang ASan shared runtime
   preloaded and `PYTHONMALLOC=malloc` to avoid CPython small-object arenas.
2. A small LLVM pass plugin sets the function-level `sanitize_address`
   attribute and runs LLVM's `AddressSanitizerPass`, including stack/global
   instrumentation, after Numba's normal optimization/linking pipeline.
3. The instrumented module is parsed back into llvmlite and compiled by MCJIT.
   There is no unsafe sharing of C++ LLVM pointers across LLVM installations.
4. Module constructors register ASan globals explicitly. ELF section-based
   global registration is disabled because MCJIT does not provide the linker
   start/stop symbols. Unique exported constructor/destructor names prevent
   collisions between JIT modules.
5. Globals remain registered until their execution engine dies. This matters
   for ufunc wrapper libraries that Numba drops while retaining native code.
6. Pristine optimized IR is used when linking JIT libraries, so instrumentation
   is applied once to the final linked module, rather than repeatedly.
7. All Numba disk caches are disabled for this process. Existing unsanitized
   objects cannot bypass instrumentation, and instrumented objects cannot
   contaminate normal caches.

The plugin uses LLVM's attribute API. In this pinned llvmlite version,
`ValueRef.add_function_attribute("sanitize_address")` attaches it at the return
attribute position and produces invalid IR, so that API is not suitable here.

## Verified coverage

Run the integration tests:

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip check
```

Buggy programs execute only in isolated subprocesses. The tests assert real
ASan error classes, access widths, shadow-byte reports, and nonzero exit
statuses, not merely the presence of runtime symbols.

Covered examples include:

- Heap out-of-bounds reads, writes, and excessively negative indices with
  `boundscheck=False`.
- Use-after-free through an externally allocated array buffer.
- NRT-created allocations when accesses cross the actual malloc boundary.
- Embedded JIT constant globals and stack allocations (the latter via a test
  intrinsic that preserves an addressable stack array).
- Linked `@njit` functions, `@cfunc`, `@vectorize`, and `@guvectorize`.
- `parallel=True` / `prange` on Numba's workqueue backend, including vectorized
  stores. This verifies ASan, not race detection; TBB/OpenMP are not tested.
- Safe execution, IR instrumentation, cache isolation, script/module execution,
  and the installed CLI outside the source directory.

## Important blind spots

**Real ASan is not the same as checking every array's logical bounds.**

- A slice/view can overflow its logical shape while staying within the backing
  allocation. ASan normally cannot detect this, just as it cannot detect every
  C++ subobject/container overflow without extra annotations.
- Numba's NRT packs allocation metadata and array storage into one allocation.
  Aligned allocations also contain padding. A one-past-end array access can
  land in unpoisoned padding; an underflow can hit live metadata. The tests
  explicitly verify the unpoisoned one-past-end boundary rather than claiming
  it is detected.
- NumPy caches small freed data buffers. A cached buffer not actually passed
  to `free` is not marked freed by ASan. Custom allocators, mmap-backed memory,
  and memory owned by other libraries require their own sanitizer integration.
- Python, NumPy, Numba's compiled runtime, BLAS, and other native extensions are
  not rebuilt with ASan here. Allocator interception works for ordinary
  malloc/free, but uninstrumented loads/stores inside those libraries are not
  checked. Instrumented JIT code accessing intercepted allocations **is**
  checked.
- JIT frames currently appear as `<unknown module>` in ASan reports. Frame
  pointers are preserved, but Python source-line symbolization/JIT object
  registration remains future work.
- Only CPU JIT code is hooked. CUDA, AOT, child-process startup strategies,
  and arbitrary third-party codegen subclasses are not covered.
- Like normal ASan, checks apply to surviving optimized accesses, require
  exercised execution paths, and do not guarantee detection of every invalid
  pointer (for example, one landing in another live allocation).

Keep a separate bounds-checking CI job where practical:

```sh
NUMBA_BOUNDSCHECK=1 .venv/bin/python -m unittest discover -s /path/to/tests
```

It complements ASan by checking logical array indices. It does not replace
ASan for pointer lifetime errors or unchecked native intrinsics.

## Other sanitizers

These are separate tools, not one combined ASan mode:

| Sanitizer | Status and requirements |
| --- | --- |
| ASan | Implemented here for CPU JIT IR, with the limitations above. |
| LSan | Available through the ASan runtime. Opt-in, with a deliberate unreachable 64 KiB allocation regression test. CPython/LLVM shutdown leaks make baseline management necessary. |
| TSan | Not implemented. Needs LLVM TSan instrumentation and a separate TSan runtime/process, plus visible synchronization in Numba's runtime, threading backends and native dependencies. Uninstrumented synchronization can cause misleading results. Cannot be combined with ASan. |
| MSan | Not implemented. Needs LLVM MSan instrumentation, shadow/origin propagation across JIT/native call boundaries, and an instrumented Python/NumPy/Numba/native dependency stack or carefully validated boundary adapters. Cannot be combined with ASan. |
| UBSan | Not implemented. Many checks are emitted by Clang's language frontend, not a generic LLVM `ubsan` pass. Numba lowering needs checks consistent with Numba's semantics; C++ signed-overflow rules must not be blindly imposed on NumPy-style arithmetic. |

LSan is disabled by default to keep unrelated interpreter/LLVM leaks out of
ordinary ASan runs. Opt in:

```sh
ASAN_OPTIONS=detect_leaks=1 .venv/bin/numba-asan run /path/to/application.py
```

Use narrow, justified `LSAN_OPTIONS=suppressions=/path/to/file` suppressions
based on a clean baseline. Broad suppressions can hide the bugs you want to
catch. Conservative root scanning can also retain seemingly leaked pointers.
The LSan test checks while Python is alive to avoid shutdown noise and discards
the allocation pointer on a terminated worker thread.

## Path to an upstream/community feature

This repository demonstrates that a separate extension package is viable.
No forks are necessary to reproduce the demonstrated ASan detections.
For maintainable upstream support, the responsibilities divide naturally:

**llvmlite**

- Expose LLVM's sanitizer passes and proper function attribute insertion.
- Manage pass settings/version compatibility without an external `opt` process.
- Provide JIT constructor/destructor and debug-object registration hooks.

**Numba**

- Provide a supported sanitizer compilation option/hook at the final
  optimization/codegen boundary, with propagation to nested functions,
  ufuncs and parallel-generated kernels.
- Include sanitizer configuration/runtime ABI in cache keys instead of
  disabling caches. Never reuse unsanitized objects for a sanitized run.
- Make NRT allocations sanitizer-aware: separate metadata from payload or
  poison allocator-owned padding/redzones, preserving alignment and correctly
  unpoisoning before destructor/debug-fill/realloc/free operations. Never
  poison live metadata or blindly poison shared array-view storage.
- Preserve source debug information and register JIT code with a sanitizer
  symbolization mechanism.
- For UBSan, add semantic checks during lowering rather than relying on Clang
  to recover Python/NumPy semantics from already-generated LLVM IR.

**NumPy and the native dependency stack**

- Use a debug memory policy (NEP 49) with ASan-compatible allocation/free
  behavior, or sanitizer-aware caching, to expose small-buffer lifetimes.
- Rebuild native components with the appropriate sanitizer for end-to-end
  coverage; TSan and especially MSan require their own validated stacks.

The next milestone should be allocator-aware NRT/NumPy coverage and source
symbolization, not advertising "full sanitizers" before those work.

## References

- [Clang AddressSanitizer](https://clang.llvm.org/docs/AddressSanitizer.html)
- [Clang LeakSanitizer](https://clang.llvm.org/docs/LeakSanitizer.html)
- [Clang ThreadSanitizer](https://clang.llvm.org/docs/ThreadSanitizer.html)
- [Clang MemorySanitizer](https://clang.llvm.org/docs/MemorySanitizer.html)
- [Clang UndefinedBehaviorSanitizer](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html)
- [llvmlite 0.45.1 release notes / LLVM 20 transition](https://llvmlite.readthedocs.io/en/v0.45.1/release-notes.html)
- [Numba 0.62.1 codegen](https://github.com/numba/numba/blob/0.62.1/numba/core/codegen.py)
- [Numba 0.62.1 NRT allocator](https://github.com/numba/numba/blob/0.62.1/numba/core/runtime/nrt.cpp)
- [NumPy data memory policies and caches](https://numpy.org/doc/stable/reference/c-api/data_memory.html)
