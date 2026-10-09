"""Shared launcher harness: buggy programs only ever run in subprocesses."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from numba_jit_asan.runner import DEFAULT_PLUGIN, _runtime


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "tests" / "cases.py"
SANITIZER_CASES = ROOT / "tests" / "sanitizer_cases.py"


def source_line(path: Path, text: str) -> int:
    """1-based line number of the unique line containing *text*."""
    matches = [number for number, line in enumerate(path.read_text().splitlines(), 1)
               if text in line]
    if len(matches) != 1:
        raise AssertionError(f"{text!r} matches lines {matches} in {path}")
    return matches[0]


class SanitizerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plugin = DEFAULT_PLUGIN.resolve()
        if not cls.plugin.is_file():
            raise RuntimeError("Build the plugin with numba-asan build before testing")
        cls.runtime = _runtime()
        cls.jit_directory = tempfile.TemporaryDirectory(prefix="sanitizer-test-jit-")

    @classmethod
    def tearDownClass(cls):
        cls.jit_directory.cleanup()

    def run_case(self, case, *, script=CASES, args=(), cache=None, sanitized=True,
                 env=None, timeout=300):
        environment = os.environ.copy()
        environment["NUMBA_NUM_THREADS"] = "2"
        environment["NUMBA_THREADING_LAYER"] = "workqueue"
        environment["NUMBA_ASAN_JIT_DIR"] = self.jit_directory.name
        environment["ASAN_OPTIONS"] = "halt_on_error=1"
        if cache is not None:
            environment["NUMBA_CACHE_DIR"] = str(cache)
        environment.update(env or {})
        if sanitized:
            command = [
                sys.executable, "-m", "numba_jit_asan.runner", "run",
                "--plugin", str(self.plugin), *args, str(script), case,
            ]
        else:
            command = [sys.executable, str(script), case]
            environment["LD_PRELOAD"] = str(self.runtime)
            environment["PYTHONMALLOC"] = "malloc"
            environment["ASAN_TEST_UNINSTRUMENTED"] = "1"
        return subprocess.run(
            command, cwd=ROOT, env=environment, capture_output=True, text=True,
            timeout=timeout,
        )

    def assert_clean(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotRegex(result.stderr, r"(Address|Thread|Memory|Leak)Sanitizer|runtime error:")

    def assert_jit_frame(self, result, function, script, text):
        """A JIT frame was rewritten in place to its Numba function and source line."""
        line = source_line(script, text)
        self.assertRegex(
            result.stderr,
            rf"#\d+ 0x[0-9a-f]+ in [^\n]*{function}\([^\n]*\) {script}:{line}:\d+ \(numba-jit\+",
        )
