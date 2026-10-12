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
NUMBA_NUM_THREADS=2 ASAN_OPTIONS=allocator_may_return_null=1 \
  numba-asan run --fuzz fuzz_pkg.py corpus/ -max_total_time=600 \
  -rss_limit_mb=4096 -timeout=60 -use_value_profile=1 -len_control=0
```

- Budget startup (imports + warm-up) separately: instrumented compilation can
  take minutes, and up to about 13 min for packages compiling at import.
- Count a run as fuzzing only above about 5 executions/s. Otherwise shrink per-input work.
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
