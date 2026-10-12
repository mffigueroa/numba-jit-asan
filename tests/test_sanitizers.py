import os
import unittest

from helpers import SANITIZER_CASES as CASES, SanitizerTestCase
from numba_jit_asan.runner import instrumented_dependencies

# ASan, TSan and MSan runtimes are mutually exclusive: modules rebuilt with
# ASan by build-deps cannot load in a TSan or MSan process.
ASAN_DEPENDENCIES = any(instrumented_dependencies().values())
needs_stock_dependencies = unittest.skipIf(
    ASAN_DEPENDENCIES, "TSan/MSan need stock (non-ASan) NumPy/Numba builds")


class UndefinedBehaviorTests(SanitizerTestCase):
    def run_ub(self, case, *args):
        return self.run_case(case, script=CASES, args=("--undefined", *args))

    def assert_ub(self, result, message, function, marker, location=None):
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertRegex(result.stderr, rf"{location or CASES}:\d+:\d+: runtime error: {message}")
        self.assert_jit_frame(result, function, CASES, marker)

    def test_shift_out_of_bounds(self):
        self.assert_ub(self.run_ub("ub_shift"),
                       "shift exponent 70 is too large for 64-bit type 'int64'",
                       "__main__::shift", "# ubsan: shift")
        self.assert_ub(self.run_ub("ub_negative_shift"),
                       "shift exponent -1 is negative", "__main__::shift", "# ubsan: shift")

    def test_float_cast_overflow(self):
        self.assert_ub(self.run_ub("ub_cast"),
                       r"1e\+300 is outside the range of representable values of type 'int32'",
                       "__main__::to_int32", "# ubsan: float to int32")
        # Numba 0.68 compiles int() as a separate @overload: the cast is reported
        # in numba/cpython/builtins.py, with the caller's line in the next frame.
        self.assert_ub(self.run_ub("ub_nan"),
                       "nan is outside the range of representable values of type 'int64'",
                       "__main__::to_int64", "# ubsan: float to int64",
                       location=rf"(?:{CASES}|\S+/numba/cpython/builtins\.py)")

    def test_in_range_operations_are_clean(self):
        result = self.run_ub("ub_safe")
        self.assert_clean(result)
        self.assertIn("in-range shifts and casts passed", result.stdout)

    def test_checks_are_opt_in(self):
        result = self.run_case("ub_shift", script=CASES)
        self.assert_clean(result)

    @needs_stock_dependencies
    def test_combined_with_thread_sanitizer(self):
        self.assert_ub(self.run_ub("ub_shift", "--sanitizer", "thread"),
                       "shift exponent 70 is too large", "__main__::shift", "# ubsan: shift")


@needs_stock_dependencies
class ThreadSanitizerTests(SanitizerTestCase):
    def run_tsan(self, case):
        return self.run_case(case, script=CASES, args=("--sanitizer", "thread"))

    def assert_race(self, result, function, marker):
        self.assertEqual(result.returncode, 66, result.stdout + result.stderr)
        self.assertIn("WARNING: ThreadSanitizer: data race", result.stderr)
        self.assert_jit_frame(result, function, CASES, marker)
        self.assertRegex(result.stderr, r"SUMMARY: ThreadSanitizer: data race ")

    def test_nogil_threads_race(self):
        result = self.run_tsan("race_threads")
        self.assert_race(result, "__main__::accumulate", "# tsan: shared counter")

    def test_prange_shared_element_race(self):
        result = self.run_tsan("race_prange")
        self.assertEqual(result.returncode, 66, result.stdout + result.stderr)
        self.assertIn("WARNING: ThreadSanitizer: data race", result.stderr)
        # Parfor kernels are generated from the loop body; the line is preserved.
        self.assertRegex(result.stderr, r"__numba_parfor_gufunc_\w+\([^\n]*:\d+:\d+ \(numba-jit\+")

    def test_synchronized_code_is_clean(self):
        locked = self.run_tsan("race_locked")
        self.assert_clean(locked)
        self.assertIn("counter 40000000", locked.stdout)
        reduction = self.run_tsan("race_reduction")
        self.assert_clean(reduction)
        self.assertIn("prange reduction passed", reduction.stdout)


@needs_stock_dependencies
class MemorySanitizerTests(SanitizerTestCase):
    def run_msan(self, case):
        return self.run_case(case, script=CASES, args=("--sanitizer", "memory"))

    def assert_uninitialized(self, result):
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("WARNING: MemorySanitizer: use-of-uninitialized-value", result.stderr)

    def test_uninitialized_branch(self):
        result = self.run_msan("uninit_branch")
        self.assert_uninitialized(result)
        self.assert_jit_frame(result, "__main__::unwritten_branch", CASES, "# msan: branch")

    def test_uninitialized_value_escapes_to_python(self):
        result = self.run_msan("uninit_escape")
        self.assert_uninitialized(result)
        self.assertIn("Uninitialized value was stored to memory at", result.stderr)
        self.assert_jit_frame(result, "__main__::total", CASES, "# msan: escape")

    def test_initialized_and_external_data_is_clean(self):
        result = self.run_msan("uninit_safe")
        self.assert_clean(result)
        self.assertIn("initialized data passed", result.stdout)


class LeakSanitizerTests(SanitizerTestCase):
    def run_lsan(self, case):
        return self.run_case(case, script=CASES, args=("--leaks",))

    def test_jit_leak_reported_with_source(self):
        result = self.run_lsan("leak_jit")
        self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
        self.assertIn("ERROR: LeakSanitizer: detected memory leaks", result.stderr)
        self.assertRegex(result.stderr, r"leak of 4096 byte\(s\) in 1 object\(s\)")
        self.assert_jit_frame(result, "__main__::leaky", CASES, "# lsan: leak")

    def test_live_python_state_is_not_reported(self):
        result = self.run_lsan("leak_none")
        self.assert_clean(result)
        self.assertIn("no leaks", result.stdout)

    def test_scoped_check_api(self):
        result = self.run_lsan("leak_scoped")
        self.assertIn("scoped leak check raised", result.stdout)
        # The leak still exists at exit, so the automatic final check fails too.
        self.assertEqual(result.returncode, 23, result.stdout + result.stderr)

    def test_leak_checks_are_opt_in(self):
        self.assert_clean(self.run_case("leak_jit", script=CASES))


class NativeDependencyTests(SanitizerTestCase):
    """Native loads/stores are checked only in ASan-rebuilt dependencies.

    Both outcomes are asserted: stock wheels silently read freed memory, while
    ``numba-asan build-deps`` makes the same native access a real ASan report.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.instrumented = instrumented_dependencies()

    def test_expected_configuration(self):
        # CI sets this so an instrumented job cannot silently test stock wheels.
        expected = os.environ.get("EXPECT_INSTRUMENTED_DEPENDENCIES")
        if expected is None:
            self.skipTest("EXPECT_INSTRUMENTED_DEPENDENCIES not set")
        self.assertEqual(set(self.instrumented.values()), {expected == "1"})

    def check(self, case, dependency, function, source):
        result = self.run_case(case, script=CASES)
        if self.instrumented[dependency]:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("ERROR: AddressSanitizer: heap-use-after-free", result.stderr)
            self.assertRegex(result.stderr, rf"#0 0x[0-9a-f]+ in {function} [^\n]*{source}")
        else:
            self.assert_clean(result)

    def test_numpy_native_loop(self):
        self.check("native_numpy", "numpy", "DOUBLE_pairwise_sum", "loops_utils.h.src")

    def test_numba_native_runtime(self):
        self.check("native_numba", "numba", "NRT_MemInfo_data", "nrt.cpp")

    @unittest.skipUnless(ASAN_DEPENDENCIES, "requires numba-asan build-deps")
    def test_incompatible_sanitizer_is_rejected(self):
        for sanitizer in ("thread", "memory"):
            with self.subTest(sanitizer=sanitizer):
                result = self.run_case("race_reduction", script=CASES,
                                       args=("--sanitizer", sanitizer))
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("rebuilt with ASan by build-deps", result.stderr)


if __name__ == "__main__":
    unittest.main()
