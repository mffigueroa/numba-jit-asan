import os
from pathlib import Path
import runpy
import shlex
import sys

from .instrumentation import enable


def _run(args) -> None:
    if args[0] == "-m":
        sys.argv = args[1:]
        runpy.run_module(args[1], run_name="__main__", alter_sys=True)
    else:
        script = Path(args[0]).resolve()
        sys.argv = [str(script), *args[1:]]
        sys.path.insert(0, str(script.parent))
        runpy.run_path(str(script), run_name="__main__")


def _relauncher(args):
    """An executable that re-enters numba-asan for libFuzzer child processes."""
    path = Path(os.environ["NUMBA_SANITIZER_JIT_DIR"]) / "fuzz-target"
    command = " ".join(shlex.quote(part) for part in
                       (sys.executable, "-m", "numba_jit_asan._bootstrap", *args))
    script = f"#!/bin/sh\nexec {command} \"$@\"\n"
    if not path.exists():
        temporary = path.with_name(f".fuzz-target.{os.getpid()}")
        temporary.write_text(script)
        temporary.chmod(0o755)
        temporary.replace(path)
    return str(path)


_CHILD_PREAMBLE = "import numba_jit_asan._bootstrap as _numba_asan; _numba_asan.child(); "


def _enable_from_environment() -> None:
    enable(
        opt=os.environ["NUMBA_ASAN_OPT"],
        plugin=Path(os.environ["NUMBA_ASAN_PLUGIN"]),
        sanitizer=os.environ.get("NUMBA_SANITIZER", "address"),
        undefined=os.environ.get("NUMBA_SANITIZER_UNDEFINED") == "1",
        coverage=os.environ.get("NUMBA_SANITIZER_COVERAGE") == "1",
    )
    _sanitize_multiprocessing_children()


def _sanitize_multiprocessing_children() -> None:
    """Enable the sanitizer in multiprocessing spawn and forkserver children.

    Those start a fresh interpreter (``python -c "from multiprocessing...``)
    that inherits the sanitizer runtime but would otherwise compile JIT code
    uninstrumented. Fork children inherit the instrumented process as is.
    """
    from multiprocessing import util

    spawnv_passfds = util.spawnv_passfds
    if getattr(spawnv_passfds, "_numba_asan", False):
        return

    def sanitized_spawnv_passfds(path, args, passfds):
        args = list(args)
        for index, arg in enumerate(args[:-1]):
            if arg == "-c" and args[index + 1].startswith("from multiprocessing"):
                args[index + 1] = _CHILD_PREAMBLE + args[index + 1]
                break
        return spawnv_passfds(path, args, passfds)

    sanitized_spawnv_passfds._numba_asan = True
    util.spawnv_passfds = sanitized_spawnv_passfds


def child() -> None:
    """Entry point of a multiprocessing spawn/forkserver child process."""
    _enable_from_environment()


def main() -> None:
    args = sys.argv[1:]
    target = args[:2] if args[0] == "-m" else [str(Path(args[0]).resolve())]
    if os.environ.get("NUMBA_SANITIZER_COVERAGE") == "1":
        os.environ["NUMBA_ASAN_RELAUNCH"] = _relauncher(target)
    _enable_from_environment()
    if os.environ.get("NUMBA_SANITIZER_LEAKS") != "1":
        _run(sys.argv[1:])
        return
    code = 0
    try:
        _run(sys.argv[1:])
    except SystemExit as exc:
        code = exc.code
    from .leaks import check

    # Before interpreter finalization: live Python objects are still reachable.
    if check(raise_on_leak=False):
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(23)  # LeakSanitizer's conventional exit code.
    raise SystemExit(code)


if __name__ == "__main__":
    main()
