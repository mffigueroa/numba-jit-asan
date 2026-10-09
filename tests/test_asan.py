import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from numba_jit_asan.runner import DEFAULT_PLUGIN, _runtime


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "tests" / "cases.py"


class AddressSanitizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plugin = (ROOT / DEFAULT_PLUGIN).resolve()
        if not cls.plugin.is_file():
            raise RuntimeError("Build the plugin with numba-asan build before testing")
        cls.runtime = _runtime()

    def run_case(self, case, *, cache=None, sanitized=True, leaks=False):
        env = os.environ.copy()
        env["NUMBA_NUM_THREADS"] = "2"
        env["NUMBA_THREADING_LAYER"] = "workqueue"
        env["ASAN_OPTIONS"] = f"detect_leaks={int(leaks)}:halt_on_error=1"
        if leaks:
            env["ASAN_OPTIONS"] += ":malloc_context_size=4"
            env["LSAN_OPTIONS"] = "max_leaks=20"
        if cache is not None:
            env["NUMBA_CACHE_DIR"] = str(cache)
        if sanitized:
            command = [
                sys.executable, "-m", "numba_jit_asan.runner", "run",
                "--plugin", str(self.plugin), str(CASES), case,
            ]
        else:
            command = [sys.executable, str(CASES), case]
            env["LD_PRELOAD"] = str(self.runtime)
            env["PYTHONMALLOC"] = "malloc"
            env["ASAN_TEST_UNINSTRUMENTED"] = "1"
        return subprocess.run(
            command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
        )

    def assert_asan(self, result, error, operation, *, vectorized=False):
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"ERROR: AddressSanitizer: {error}", result.stderr)
        if vectorized:
            self.assertRegex(result.stderr, rf"{operation} of size (8|16|32|64)\b")
        else:
            self.assertIn(f"{operation} of size 8", result.stderr)
        self.assertIn("Shadow bytes around the buggy address", result.stderr)

    def test_real_heap_errors(self):
        cases = [
            ("write", "WRITE"), ("read", "READ"), ("underflow", "READ"),
            ("linked", "WRITE"), ("nrt_overflow", "WRITE"),
            ("cfunc", "READ"),
        ]
        for case, operation in cases:
            with self.subTest(case=case):
                self.assert_asan(self.run_case(case), "heap-buffer-overflow", operation)

    def test_parallel_vectorized_store(self):
        self.assert_asan(
            self.run_case("parallel"), "heap-buffer-overflow", "WRITE",
            vectorized=True,
        )

    def test_use_after_free(self):
        self.assert_asan(self.run_case("uaf"), "heap-use-after-free", "READ")

    def test_global_registration(self):
        self.assert_asan(self.run_case("global"), "global-buffer-overflow", "READ")

    def test_stack_redzones(self):
        self.assert_asan(self.run_case("stack"), "stack-buffer-overflow", "READ")

    def test_ufunc_wrappers(self):
        self.assert_asan(self.run_case("vectorize"), "global-buffer-overflow", "READ")
        self.assert_asan(self.run_case("guvectorize"), "heap-buffer-overflow", "READ")

    def test_lsan_opt_in(self):
        result = self.run_case("leak", leaks=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ERROR: LeakSanitizer: detected memory leaks", result.stderr)
        self.assertIn("65536 byte(s) in 1 object(s)", result.stderr)

    def test_safe_behavior(self):
        result = self.run_case("safe")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("safe cases passed", result.stdout)

    def test_real_ir_instrumentation(self):
        result = self.run_case("ir")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("real ASan load instrumentation", result.stdout)

    def test_preload_alone_does_not_instrument_jit(self):
        result = self.run_case("write", sanitized=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("ERROR: AddressSanitizer", result.stderr)

    def test_unsanitized_cache_cannot_bypass_instrumentation(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            baseline = self.run_case("safe", cache=cache, sanitized=False)
            self.assertEqual(baseline.returncode, 0, baseline.stderr)
            before = {path: path.read_bytes() for path in cache.rglob("*") if path.is_file()}
            self.assertTrue(any(path.suffix == ".nbc" for path in before))
            self.assert_asan(
                self.run_case("write", cache=cache), "heap-buffer-overflow", "WRITE",
            )
            after = {path: path.read_bytes() for path in cache.rglob("*") if path.is_file()}
            self.assertEqual(before, after, "ASan must not modify ordinary Numba caches")

    def test_documented_allocation_blind_spots(self):
        for case in ("nrt_padding", "view"):
            with self.subTest(case=case):
                result = self.run_case(case)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_module_entry_point(self):
        result = subprocess.run(
            [
                sys.executable, "-m", "numba_jit_asan.runner", "run",
                "--plugin", str(self.plugin), "--", "-m", "numba_jit_asan.runner",
                "--help",
            ],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Run real LLVM AddressSanitizer", result.stdout)

    def test_missing_plugin_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable, "-m", "numba_jit_asan.runner", "run",
                    "--plugin", str(Path(directory) / "missing.so"), str(CASES), "safe",
                ],
                cwd=ROOT, capture_output=True, text=True, timeout=120,
            )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("plugin is missing", result.stderr)

    def test_installed_launcher_outside_workspace(self):
        launcher = Path(sys.executable).with_name("numba-asan")
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    str(launcher), "run", "--plugin", str(self.plugin),
                    str(ROOT / "examples" / "safe.py"),
                ],
                cwd=directory, capture_output=True, text=True, timeout=120,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("safe code passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
