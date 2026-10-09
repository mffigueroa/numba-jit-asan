"""Atheris integration: coverage-guided fuzzing of Numba JIT code.

Fuzzer runs are bounded (-runs / -max_total_time, -rss_limit_mb, subprocess
timeouts) and their output is captured through files and read with a size
limit, so a misbehaving runtime cannot exhaust memory.
"""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from helpers import ROOT, SanitizerTestCase, source_line
from numba_jit_asan.runner import DEFAULT_PLUGIN, instrumented_dependencies

CASES = ROOT / "tests" / "atheris_cases.py"
MODULE = ROOT / "tests" / "atheris_module.py"
OUTPUT_LIMIT = 2 * 1024 * 1024
FUZZ_LIMITS = ["-rss_limit_mb=2048", "-seed=1", "-print_final_stats=0"]


@unittest.skipUnless(importlib.util.find_spec("atheris"), "Atheris is not installed")
class AtherisTests(SanitizerTestCase):
    def fuzz(self, case, *flags, launcher_args=("--fuzz",), env=None, timeout=240):
        directory = Path(tempfile.mkdtemp(prefix="atheris-test-", dir=self.work.name))
        environment = os.environ.copy()
        environment.update({
            "FUZZ_CASE": case, "NUMBA_NUM_THREADS": "2",
            "NUMBA_THREADING_LAYER": "workqueue",
            "NUMBA_ASAN_JIT_DIR": self.jit_directory.name,
            "PYTHONPATH": str(ROOT / "tests"),
        })
        environment.update(env or {})
        command = [sys.executable, "-m", "numba_jit_asan.runner", "run",
                   "--plugin", str(self.plugin), *launcher_args, str(CASES),
                   *FUZZ_LIMITS, f"-artifact_prefix={directory}/", *flags]
        log = directory / "output.log"
        with log.open("wb") as output:
            result = subprocess.run(command, cwd=directory, env=environment,
                                    stdout=output, stderr=subprocess.STDOUT,
                                    timeout=timeout)
        with log.open("rb") as output:
            text = output.read(OUTPUT_LIMIT).decode(errors="replace")
        self.assertNotIn("DEADLYSIGNAL", text)
        return result.returncode, text, directory

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.work = tempfile.TemporaryDirectory(prefix="atheris-tests-")

    @classmethod
    def tearDownClass(cls):
        cls.work.cleanup()
        super().tearDownClass()

    def artifacts(self, directory, kind):
        return sorted(directory.glob(f"{kind}-*"))

    def assert_overflow(self, code, output, function, path, marker):
        self.assertNotEqual(code, 0, output)
        self.assertIn("ERROR: AddressSanitizer: heap-buffer-overflow", output)
        line = source_line(path, marker)
        self.assertRegex(output, rf"#0 0x[0-9a-f]+ in {function}\([^\n]*\) {path}:{line}:")

    def test_finds_comparison_guarded_jit_bug_and_reproduces_it(self):
        # A 4-byte magic compared inside JIT code: found only through JIT
        # coverage and comparison tracing (2**32 guesses otherwise).
        code, output, directory = self.fuzz("parse", "-runs=2000000", "-max_total_time=150")
        self.assert_overflow(code, output, "__main__::parse", CASES, "# atheris: magic overflow")
        self.assertRegex(output, r"INITED cov: \d+")
        crashes = self.artifacts(directory, "crash")
        self.assertEqual(len(crashes), 1, output)
        self.assertTrue(crashes[0].read_bytes().startswith(b"NBA!"))

        code, output, _ = self.fuzz("parse", str(crashes[0]))
        self.assert_overflow(code, output, "__main__::parse", CASES, "# atheris: magic overflow")
        self.assertIn(f"Running: {crashes[0]}", output)

    def test_bytecode_instrumentation_keeps_jit_functions_compilable(self):
        code, output, directory = self.fuzz("instrument_imports", "-runs=1000000",
                                            "-max_total_time=120")
        self.assertNotIn("TypingError", output)
        self.assert_overflow(code, output, "atheris_module::checksum", MODULE,
                             "# atheris module: overflow")
        code, output, _ = self.fuzz("instrument_all", "-runs=3000")
        self.assertNotIn("TypingError", output)
        self.assertNotIn("Uncaught Python exception", output)
        self.assertEqual(code, 0, output)

    def test_fork_mode_children_are_sanitized(self):
        code, output, directory = self.fuzz("fork", "-fork=1", "-max_total_time=150")
        self.assertIn("-fork=1: fuzzing in separate process", output)
        self.assertNotIn("Permission denied", output)
        self.assertRegex(output, r"cov: [1-9]\d* ft: \d+ corp: [1-9]")
        self.assert_overflow(code, output, "__main__::parse", CASES, "# atheris: magic overflow")
        self.assertEqual(len(self.artifacts(directory, "crash")), 1, output)

    def test_fork_mode_warns_without_compiled_jit_code(self):
        code, output, _ = self.fuzz("fork_cold", "-fork=1", "-max_total_time=3",
                                    "-ignore_crashes=1")
        self.assertIn("WARNING: numba-asan: -fork is used, but no Numba JIT code", output)

    def test_timeout_reports_hanging_jit_line(self):
        code, output, directory = self.fuzz("hang", "-timeout=2", "-max_total_time=90")
        self.assertIn("ERROR: libFuzzer: timeout", output)
        self.assertEqual(len(self.artifacts(directory, "timeout")), 1, output)
        line = source_line(CASES, "# atheris: hang")
        self.assertRegex(output, rf"in __main__::spin\([^\n]*\) {CASES}:{line}:")

    def test_undefined_behavior_checks(self):
        code, output, directory = self.fuzz("shift", "-max_total_time=90",
                                            launcher_args=("--fuzz", "--undefined"))
        line = source_line(CASES, "# atheris: shift")
        self.assertRegex(output, rf"{CASES}:{line}:\d+: runtime error: shift exponent \d+ "
                                 r"is too large for 64-bit type 'int64'")
        self.assertEqual(len(self.artifacts(directory, "crash")), 1, output)

    def test_run_without_fuzz_explains_coverage(self):
        code, output, _ = self.fuzz("parse", "-runs=10", launcher_args=())
        self.assertEqual(code, 0, output)
        self.assertIn("Use `numba-asan run --fuzz` both to fuzz and to reproduce", output)

    def test_rejects_manually_preloaded_atheris_runtime(self):
        import atheris

        runtime = Path(atheris.path()) / "asan_with_fuzzer.so"
        code, output, _ = self.fuzz("parse", "-runs=10", env={"LD_PRELOAD": str(runtime)})
        self.assertEqual(code, 2, output)
        self.assertIn("use --fuzz instead of preloading", output)


class FuzzerRuntimeTests(unittest.TestCase):
    def test_runtime_matches_jit_instrumentation(self):
        runtime = DEFAULT_PLUGIN.with_name("asan_with_fuzzer.so")
        symbols = subprocess.run(["llvm-nm-20", "-D", "--defined-only", str(runtime)],
                                 capture_output=True, text=True, check=True).stdout
        for symbol in ("LLVMFuzzerRunDriver", "__asan_version_mismatch_check_v8",
                       "__sanitizer_cov_8bit_counters_init", "__sanitizer_cov_pcs_init",
                       "__ubsan_handle_shift_out_of_bounds"):
            self.assertRegex(symbols, rf" T {symbol}\n")

    @unittest.skipUnless(os.environ.get("EXPECT_FUZZER_DEPENDENCIES") == "1",
                         "requires numba-asan build-deps --fuzzer")
    def test_dependencies_have_native_fuzzer_coverage(self):
        coverage = instrumented_dependencies("__sanitizer_cov_8bit_counters_init")
        self.assertEqual(set(coverage.values()), {True}, coverage)


if __name__ == "__main__":
    unittest.main()
