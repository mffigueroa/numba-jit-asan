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

from helpers import ROOT, SanitizerTestCase, run_bounded, source_line
from numba_jit_asan.runner import DEFAULT_PLUGIN, instrumented_dependencies
from numba_jit_asan.toolchain import tool_name

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
            result = run_bounded(command, cwd=directory, env=environment,
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
        # Value profiling makes discovery robust to the seed and to how many
        # other modules (instrumented NumPy/Numba) share the coverage map.
        code, output, directory = self.fuzz("parse", "-runs=2000000", "-max_total_time=150",
                                            "-use_value_profile=1")
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
        # Instrumenting every loaded function takes minutes on some Pythons.
        code, output, _ = self.fuzz("instrument_all", "-runs=3000", timeout=900)
        self.assertNotIn("TypingError", output)
        self.assertNotIn("Uncaught Python exception", output)
        self.assertEqual(code, 0, output)

    def test_spawned_workers_compile_original_bytecode_and_are_sanitized(self):
        code, output, _ = self.fuzz("spawn", "-runs=1")
        self.assertNotIn("TypingError", output)
        self.assertNotIn("instrumented by Atheris before numba-asan", output)
        self.assertIn("ERROR: AddressSanitizer: heap-buffer-overflow", output)
        line = source_line(MODULE, "# atheris module: overflow")
        self.assertRegex(output, rf"#0 0x[0-9a-f]+ in atheris_module::checksum\([^\n]*\) "
                                 rf"{MODULE}:{line}:")
        self.assertNotEqual(code, 0, output)

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
        # The preloaded runtime also sanitizes the launcher's interpreter; its
        # exit-time leak report (CPython-build dependent) would change the exit code.
        environment = os.environ.copy()
        # handle_segv=0: Atheris's older runtime sometimes crashes during the
        # interpreter's own startup on current kernels; it must then die at once
        # instead of looping on DEADLYSIGNAL.
        environment.update({"LD_PRELOAD": str(runtime),
                            "ASAN_OPTIONS": "detect_leaks=0:handle_segv=0",
                            "FUZZ_CASE": "parse", "PYTHONPATH": str(ROOT / "tests")})
        outputs = []
        for _ in range(5):
            # The installed command, as users run it.
            result = run_bounded(
                [str(Path(sys.executable).with_name("numba-asan")), "run",
                 "--plugin", str(self.plugin), "--fuzz", str(CASES), "-runs=10"],
                env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=120,
            )
            output = result.stdout.decode(errors="replace")[:OUTPUT_LIMIT]
            self.assertNotIn("DEADLYSIGNAL", output)
            if result.returncode == 2:
                self.assertIn("use --fuzz instead of preloading", output)
                return
            # Anything else must be the foreign runtime crashing before numba-asan ran.
            self.assertLess(result.returncode, 0, output)
            outputs.append(result.returncode)
        self.skipTest(f"Atheris's prebuilt runtime crashed at startup in every attempt: {outputs}")


class FuzzerRuntimeTests(unittest.TestCase):
    def test_runtime_matches_jit_instrumentation(self):
        runtime = DEFAULT_PLUGIN.with_name("asan_with_fuzzer.so")
        symbols = subprocess.run([tool_name("llvm-nm"), "-D", "--defined-only", str(runtime)],
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
