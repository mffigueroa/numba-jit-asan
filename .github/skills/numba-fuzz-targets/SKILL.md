---
name: numba-fuzz-targets
description: Use when writing, generating, running or triaging Atheris fuzz targets for Python packages with Numba JIT code under numba-asan (SDL fuzz testing). Covers warm-up, sizing, libFuzzer flags, oracles and false-positive pitfalls learned across ~100 packages.
---

# Fuzzing Numba packages with Atheris and numba-asan

The full rationale and evidence is in the README section "Writing effective
fuzz targets for Numba code". This is the checklist.

## Before writing anything

Prefer generating targets from existing tests: record the public-API calls
the test suite makes, then replay and mutate them. Write a target by hand
only for inputs a test-derived target cannot reach, or to add an oracle that
detects wrong results.

## Target checklist

1. File name `fuzz_<pkg>.py`. `<pkg>.py` shadows the package.
2. Import NumPy/SciPy first, then `with atheris.instrument_imports(include=["<pkg>"]):`
   import only the product.
3. **Warm up** before `atheris.Fuzz()`: call each public entry point once per
   type signature the target can generate (dtype × ndim × layout × read-only
   × optional `None`). Few signatures in total.
4. Sizes come from fuzzer integers and cheap fills (tiling, patterns), and
   must be able to cross fixed capacities found in the JIT source.
5. Fuzz unvalidated parameter values too (negative, zero, empty, endpoints).
6. Oracles:
   - a sanitizer report is memory safety;
   - an explicit `raise` in the product, or validation inside a dependency, is handled;
   - an implicit `IndexError`/`ZeroDivisionError`/`KeyError`/`AssertionError`
     from product or JIT code is a robustness finding: record it and keep fuzzing.

   Never catch `Exception` or `TypingError`.

## Run settings

```sh
NUMBA_NUM_THREADS=2 ASAN_OPTIONS=allocator_may_return_null=1:max_allocation_size_mb=2048 \
  numba-asan run --fuzz fuzz_pkg.py corpus/ -max_total_time=600 \
  -rss_limit_mb=4096 -timeout=60 -use_value_profile=1 -len_control=0
```

- Budget startup (imports + warm-up) separately: instrumented compilation can
  take minutes, and up to about 13 min for packages compiling at import.
- Count a run as fuzzing only above about 5 executions/s. Otherwise shrink per-input work.
- Keep `max_allocation_size_mb` below `-rss_limit_mb`: huge single allocations
  then raise `MemoryError` (handled) instead of ending the run as an OOM.
- `build-deps`: ASan-only NumPy/Numba (no `--fuzzer`) unless the dependencies
  themselves are under test. Use `--wheelhouse` to reuse the builds.

## Triage

- Replay every artifact: `numba-asan run --fuzz fuzz_pkg.py crash-<hash>`.
- Timeout or OOM within the first inputs: usually a missing warm-up or a
  limit set too low, not a product bug. Check `slowest_unit_time_sec` and the
  baseline RSS before calling it a finding.
- An exception raised by the product's own validation is not a finding. Fix
  the target.
- Confirm an exception finding outside the sanitizer too. A sanitizer report
  confirms itself: a plain run often silently returns a plausible-looking
  value while reading out of bounds.
- Never publish findings about third-party packages (issues, READMEs, commit
  messages, public CI logs) before their maintainers have fixed them. Report
  privately (GitHub private vulnerability reporting / security policy).

## Learnings

- Shell-script wrappers for LLVM tools break under sanitizer preloads. TSan
  re-executes them, and `opt` then received the script as IR. Use real binaries (rpath-patched).
- Ubuntu's compiler-rt looks for the symbolizer only at
  `/usr/bin/llvm-symbolizer-<N>`. `numba-asan run` passes
  `external_symbolizer_path` and unsets `DEBUGINFOD_URLS`, whose network
  lookups stalled reports.
- Test subprocesses that launch sanitizers must be killed as a process group
  on timeout. An orphaned `opt --version` running with a foreign ASan runtime
  wrote ~100 GB of `DEADLYSIGNAL` lines.
- Do not edit files a running job depends on (requirements, suites,
  inventories). The integration runner fingerprints them and fails the job.
- Guard every side effect of a fuzz target with `if __name__ == "__main__":`.
  Products that use `multiprocessing` (spawn/forkserver, or the Python 3.14
  default) re-import the main script in each worker; an unguarded target
  re-runs its setup there and the pool breaks (`BrokenProcessPool`).
- Pass Numba dispatchers to `spawn`/`forkserver` workers only under
  numba-asan 89824da or later. Before that, the workers ran uninstrumented JIT
  code. Under Atheris `instrument_imports` they also failed with
  `TypingError: Unknown attribute '_trace_branch'`, because a dispatcher not
  yet compiled in the parent was pickled with instrumented bytecode. Products
  that force `set_start_method("spawn")` at import (e.g. in a `utils` module)
  hit this even when the test never asks for parallelism.
- CLI seeds must carry their input files. A recorded argv points into the
  test's `tmp_path`, which is deleted before warm-up, so every seed is
  dropped (`FileNotFoundError`, "input file should exist"). Snapshot the
  named files and their small siblings into the seed, rewrite argv to a fresh
  workspace per replay, and mutate the file contents. File-format
  mutation found a JIT heap overflow that argv replay alone could not reach.
- When seeds come from recorded test calls, record the API the test called,
  not the frame that ran. Tests often reach JIT code through third-party
  code, such as an inherited sklearn `fit_transform`, a product callback
  passed to `pairwise_distances`, or click's `CliRunner.invoke`. Name methods
  through the receiver's class, so subclasses and inherited classmethods
  replay unchanged. Check that every name resolves back to the executed code:
  sklearn's `set_output` wrappers share code, so tell them apart by closure,
  and `singledispatch` implementations are all named `_`.
- Pickle seed arguments with plain pickle first (arrays stay mutable), then
  cloudpickle (closures and classes defined in the test). Pickle product
  singletons by module reference. Never unpickle and re-pickle recorded
  blobs: a class pickled by value can fail to rebuild ("metaclass conflict").
  Resolve references between calls through the original call index instead.
- Snapshot the test's `tmp_path` for each call that names a path inside it,
  and give every replay a fresh copy. Replaying against the recorded
  directory fails on "already exists" checks and reads the recorded outputs.
- Measure before speeding up a slow target: record time per replayed call,
  the cost of an unmutated seed, and Numba compilations after warm-up.
  Across ~20 slow products, only one recompiled per input; the product JITs
  the user's callable on every call. The rest were slow per call under the
  sanitizer, with model loading, file conversion and long fits taking
  1–60 s. Remedies that help: run unmutated setup calls once and restore a
  joint deep copy (copying results separately breaks identity, since
  `fit` returns `self`); give slow products a longer, capped budget
  measured in executions rather than executions per second.
- Hide libFuzzer's arguments from product code: set `sys.argv = [sys.argv[0]]`
  before importing the product and pass the saved list to `atheris.Setup`.
  Packages that parse `sys.argv` at import (argparse, docopt) otherwise exit
  on `-rss_limit_mb=...`.
- libFuzzer's `-timeout` may miss hangs in multithreaded processes (SIGALRM
  reaches a Numba/OpenMP worker). Arm `faulthandler.dump_traceback_later(t,
  exit=True)` per input; it works while JIT code holds the GIL. Search its
  "Timeout (h:mm:ss)!" banner anywhere in the log: product progress bars may
  leave no newline before it.
- Under the `workqueue` threading layer, Numba drops bounds-check errors
  raised in `prange` kernels. The call returns with uninitialized output.
  Cross-check parallel findings with OpenMP (`NUMBA_THREADING_LAYER=omp`).
  Errors escaping parallel kernels surface as
  `SystemError: ... returned a result with an exception set`: treat it as an
  implicit (robustness) error.
- Corroborate ASan findings independently: replay with `NUMBA_BOUNDSCHECK=1`
  (an `IndexError` replaces the report), or replay the call in plain Python
  (a segfault confirms it).
- Atheris 3.x imports only on Python >= 3.11 (it uses `dis.Positions`), and
  3.1.0 has no CPython 3.11 wheels: use 2.3.0 on 3.10 and 3.0.0 on 3.11.
  Source builds of Atheris find libFuzzer through `CLANG_BIN`.
- Products often pin an older Numba/llvmlite in their lock files. Overriding
  the pin with a supported release (uv `--overrides`, also dropping the
  product's own Numba override) and letting the product's test suite validate
  the result unblocks most of them; try both supported Numba releases.
- If the ASan rebuild of NumPy fails (old releases predate Meson), fuzz with
  the stock wheels: the product's JIT code is still instrumented.
- Seeds must reproduce the test's process context: tests that `chdir` into a
  prepared directory call the product with relative paths, and environment
  variables the test set (a GUI toolkit's offscreen platform) must apply at
  replay. Numba dispatchers registered with `functools.singledispatch` are
  reached through standard-library wrapper frames, and a class with its own
  `__new__` must be replayed as the class call (`__new__` alone skips
  `__init__`). Attribute access through `__getattr__` is not an API call:
  replaying it on an unpickled receiver can recurse forever. Also snapshot
  scratch files tests create next to themselves (not only `tmp_path`) and
  directories they put on `sys.path` (model packages imported by name).
- Verify a "clean" automatic run actually exercised the JIT-reaching call:
  a seed that replays only a trivial setup call reports thousands of
  executions per second while testing nothing.
- Guard every fuzzing step with a memory limit below the job's cgroup cap:
  a runaway input then becomes an OOM finding instead of an evidence-less
  job kill.
- For chains with slow setup (loading a model, building a simulation),
  mutate setup calls rarely and restore their results from a cache; a setter
  returning None in the middle should only shorten the cached prefix, not
  disable caching.
- Tests that run a CLI in-process (`runpy.run_module`, `runpy.run_path`) are
  replayable entry points with their argv and input files. A script outside
  the package re-creates its `@njit` functions on every execution, so each
  input recompiles: expect such seeds to be too slow.
- Products that `njit` a fresh but equivalent function per call (closures,
  lambdas, `lambdify` output, callbacks unpickled per input) recompile on
  every input. Reusing the first dispatcher is safe when the code and every
  compile-time constant (closure cells, defaults, referenced globals) are
  equal; compare the code captured at jit time, since numba-jit-asan swaps a
  jitted function's Atheris-instrumented code for the original afterwards.
  Replay findings with reuse disabled.
- More seeds per test help: keep several argument variants of the same call
  chain (parametrized tests), and record public functions the product calls
  internally with the values it passes them.
- Recording must not change the program under test: pickling an argument
  can mutate it (a `__getstate__` that pops attributes from the live object).
  Snapshot and restore argument objects' attributes around pickling, and
  re-check seed counts and statuses on a random sample after recorder changes.
- A constructor behind a decorator (`functools.wraps` around `__init__`) is
  still a class call: record it as one so the instance becomes a reference for
  later calls (`obj(...)`, `obj.method(...)`). Otherwise the receiver is pickled
  by value with all the data it loaded and can exceed size limits, losing the seed.
- Judge "too slow to fuzz" by an execution count capped at the smoke budget
  (rate x min(budget, 120 s)), not rate x budget: longer SDL budgets must never
  downgrade a slow product that was fuzzed enough in a short run.
- A property setter (`obj.attr = x`) shares its qualname with the getter;
  resolving the recorded name gives the getter and replay fails with an
  argument-count TypeError. Record setters as `Class.attr.fset`.
- Numba leaks arrays allocated in a jitted function that raises, so RSS can
  grow across many rejected inputs until libFuzzer reports an OOM that no
  single input reproduces. Restart from the corpus and count executions
  across the runs; do not report it as a finding.
- Tests often configure an object with plain attribute assignments
  (`obj.option = {...}`), which a profile hook never sees. Diff the object's
  `__dict__` against its state after the last recorded call that used it and
  replay the changes as `setattr` calls. Otherwise the replay drops the seed
  with a missing-configuration error.
- An output path the test names inside a scratch directory it created (not
  yet existing) must be snapshotted with that directory. Otherwise replay
  writes into a directory that is already gone.
