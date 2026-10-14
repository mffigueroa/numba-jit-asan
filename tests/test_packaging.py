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


if __name__ == "__main__":
    unittest.main()
