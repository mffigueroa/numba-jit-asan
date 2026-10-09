from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from helpers import CASES, ROOT, SanitizerTestCase


class AddressSanitizerTests(SanitizerTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.native_directory = tempfile.TemporaryDirectory(prefix="asan-test-native-")
        cls.native_libraries = {}
        for instrumented in (False, True):
            path = Path(cls.native_directory.name) / f"native_{instrumented}.so"
            command = [
                "clang++-20", "-shared", "-fPIC", "-g", "-O1",
                str(ROOT / "tests" / "native_access.cpp"), "-o", str(path),
            ]
            if instrumented:
                command += ["-fsanitize=address", "-fno-omit-frame-pointer"]
            subprocess.run(command, check=True)
            cls.native_libraries[instrumented] = path

    @classmethod
    def tearDownClass(cls):
        cls.native_directory.cleanup()
        super().tearDownClass()

    def run_case(self, case, *, native=None, **options):
        if native is not None:
            options["env"] = {"ASAN_TEST_NATIVE_LIBRARY": str(self.native_libraries[native])}
        return super().run_case(case, **options)

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

    def test_nrt_exact_payload_boundaries(self):
        for case in ("nrt_one_past", "nrt_underflow", "nrt_metadata"):
            with self.subTest(case=case):
                self.assert_asan(self.run_case(case), "heap-buffer-overflow", "WRITE")

    def test_nrt_api_and_destructor_semantics(self):
        for case in ("nrt_api", "nrt_destructor"):
            with self.subTest(case=case):
                result = self.run_case(case)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_custom_allocator_guards(self):
        for case in (
            "external_overflow", "external_underflow",
            "external_mmap_overflow", "external_mmap_underflow",
        ):
            with self.subTest(case=case):
                result = self.run_case(case)
                self.assert_asan(result, "use-after-poison", "WRITE")
        for case in ("external_safe", "external_mmap_safe"):
            result = self.run_case(case)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("freed the original base", result.stdout)

    def test_numpy_small_buffer_lifetime(self):
        self.assert_asan(self.run_case("numpy_small_uaf"), "heap-use-after-free", "READ")

    def test_numpy_policy_threads_and_explicit_contexts(self):
        for case in ("numpy_thread", "numpy_context"):
            with self.subTest(case=case):
                result = self.run_case(case)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_jit_source_diagnostics(self):
        result = self.run_case("write")
        self.assert_asan(result, "heap-buffer-overflow", "WRITE")
        # Frame #0 is rewritten in place instead of "(<unknown module>)".
        self.assertRegex(result.stderr, r"#0 0x[0-9a-f]+ in __main__::write\(")
        self.assert_jit_frame(result, "__main__::write", CASES, "values[index] = 123")
        self.assertNotRegex(result.stderr, r"#0 0x[0-9a-f]+ +\(<unknown module>\)")
        self.assertRegex(result.stderr, rf"SUMMARY: AddressSanitizer: heap-buffer-overflow "
                                         rf"{CASES}:\d+:\d+ in __main__::write")

    def test_debug_info_does_not_change_boundscheck(self):
        self.assert_asan(self.run_case("default_unchecked"), "heap-buffer-overflow", "READ")
        result = self.run_case("checked")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("explicit bounds checking preserved", result.stdout)

    def test_native_access_requires_native_instrumentation(self):
        uninstrumented = self.run_case("native_store", native=False)
        self.assertEqual(uninstrumented.returncode, 0, uninstrumented.stderr)
        self.assertNotIn("ERROR: AddressSanitizer", uninstrumented.stderr)
        instrumented = self.run_case("native_store", native=True)
        self.assert_asan(instrumented, "heap-buffer-overflow", "WRITE")
        self.assertIn("unchecked_native_store", instrumented.stderr)
        self.assertIn("native_access.cpp:5", instrumented.stderr)

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
        self.assertIn("is missing; run numba-asan build", result.stderr)

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
