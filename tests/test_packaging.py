"""Packaging metadata: the fuzz extra installs a working Atheris on every supported Python."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]

try:
    from packaging.requirements import Requirement
except ImportError:  # pragma: no cover - packaging ships with pip/setuptools environments
    Requirement = None


def _fuzz_extra():
    text = (ROOT / "pyproject.toml").read_text()
    block = re.search(r"^fuzz = \[(.*?)\]", text, re.S | re.M).group(1)
    return [Requirement(item) for item in re.findall(r"'([^']+)'|\"([^\"]+)\"", block)
            for item in item if item]


@unittest.skipIf(Requirement is None, "packaging is not installed")
class FuzzExtraTests(unittest.TestCase):
    def test_exactly_one_atheris_per_supported_python(self):
        requirements = _fuzz_extra()
        for minor in range(10, 15):
            environment = {"python_version": f"3.{minor}", "python_full_version": f"3.{minor}.0"}
            selected = [str(req.specifier) for req in requirements
                        if req.name == "atheris" and
                        (req.marker is None or req.marker.evaluate(environment))]
            self.assertEqual(len(selected), 1, (minor, selected))
            # Atheris 3.x uses dis.Positions, which Python 3.10 lacks.
            if minor == 10:
                self.assertTrue(selected[0].startswith("==2."), selected)


class BuildJobsTests(unittest.TestCase):
    """build-deps sizes compile parallelism to the memory it may use."""

    def setUp(self):
        import os
        from numba_jit_asan import runner
        self.runner = runner
        self.saved = os.environ.pop("NUMBA_ASAN_BUILD_JOBS", None)

    def tearDown(self):
        import os
        os.environ.pop("NUMBA_ASAN_BUILD_JOBS", None)
        if self.saved is not None:
            os.environ["NUMBA_ASAN_BUILD_JOBS"] = self.saved

    def test_jobs_follow_memory_budget_and_override(self):
        import os
        from unittest import mock
        with mock.patch.object(self.runner, "_memory_budget", return_value=3 << 30), \
                mock.patch.object(os, "cpu_count", return_value=16):
            self.assertEqual(self.runner.build_jobs(), 3)
        with mock.patch.object(self.runner, "_memory_budget", return_value=100 << 20):
            self.assertEqual(self.runner.build_jobs(), 1)
        os.environ["NUMBA_ASAN_BUILD_JOBS"] = "5"
        self.assertEqual(self.runner.build_jobs(), 5)
        self.assertIn("-Ccompile-args=-j5", self.runner._build_options("numpy", 5))
        self.assertNotIn("-Ccompile-args=-j5", self.runner._build_options("numba", 5))


if __name__ == "__main__":
    unittest.main()
