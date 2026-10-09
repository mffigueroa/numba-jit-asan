import os
from pathlib import Path
import runpy
import sys

from .instrumentation import enable


def main() -> None:
    enable(
        opt=os.environ["NUMBA_ASAN_OPT"],
        plugin=Path(os.environ["NUMBA_ASAN_PLUGIN"]),
    )
    args = sys.argv[1:]
    if args[0] == "-m":
        sys.argv = args[1:]
        runpy.run_module(args[1], run_name="__main__", alter_sys=True)
    else:
        script = Path(args[0]).resolve()
        sys.argv = [str(script), *args[1:]]
        sys.path.insert(0, str(script.parent))
        runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
