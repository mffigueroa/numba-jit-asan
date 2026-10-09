"""Compatibility with Atheris's Python bytecode instrumentation.

``atheris.instrument_imports()`` and ``atheris.instrument_all()`` rewrite
Python bytecode to insert coverage callbacks. Numba compiles ``@njit``
functions from their bytecode and cannot type those callbacks, so every
instrumented JIT function would fail with a ``TypingError``.

This module records the original code object for every code object Atheris
instruments, and Numba compiles from the original. JIT code gets native
LLVM SanitizerCoverage instead (``numba-asan run --fuzz``), so no coverage is
lost; Python code keeps Atheris's bytecode coverage.
"""
import importlib.abc
import importlib.machinery
import sys
import types

_MARKER = "__ATHERIS_INSTRUMENTED__"
_originals: dict = {}
_relaunch = None  # executable re-entering numba-asan for child processes
_coverage_modules = 0
_fork_mode = False


def note_coverage_module() -> None:
    """Called when a JIT module registers its coverage counters."""
    global _coverage_modules
    _coverage_modules += 1


def _instrumented(code: types.CodeType) -> bool:
    return any(isinstance(const, str) and const == _MARKER for const in code.co_consts)


def original_code(code: types.CodeType) -> types.CodeType:
    """The uninstrumented code object for *code* (or *code* itself)."""
    if not _instrumented(code):
        return code
    try:
        return _originals[code]
    except KeyError:
        raise RuntimeError(
            f"{code.co_name} ({code.co_filename}:{code.co_firstlineno}) was "
            "instrumented by Atheris before numba-asan could record its original "
            "bytecode; run the harness with numba-asan run"
        ) from None


def restore(function) -> None:
    """Give a function about to be compiled by Numba its original bytecode."""
    code = getattr(function, "__code__", None)
    if isinstance(code, types.CodeType) and _instrumented(code):
        function.__code__ = original_code(code)


def _wrap_patch_code(module) -> None:
    patch_code = module.patch_code
    if getattr(patch_code, "_numba_asan", False):
        return

    def recording_patch_code(code, *args, **kwargs):
        # Atheris recurses through this module global for nested functions,
        # so their originals are recorded too.
        result = patch_code(code, *args, **kwargs)
        if result is not code:
            _originals[result] = code
        return result

    recording_patch_code._numba_asan = True
    module.patch_code = recording_patch_code


def _wrap_setup(module) -> None:
    """libFuzzer -fork/-jobs/-merge re-run argv[0]; keep them sanitized.

    libFuzzer treats args[0] as the program to run (usually the harness
    script from sys.argv). Children must re-enter numba-asan, or their JIT
    code would be uninstrumented. The harness's own sys.argv is unchanged.
    """
    setup = module.Setup
    if getattr(setup, "_numba_asan", False):
        return

    def relaunching_setup(args, *rest, **options):
        global _fork_mode
        args = list(args)
        if _relaunch is None:
            sys.stderr.write(
                "INFO: numba-asan: Atheris harness run without --fuzz: no JIT "
                "coverage. Use `numba-asan run --fuzz` both to fuzz and to "
                "reproduce crash inputs.\n")
        elif args:
            args[0] = _relaunch
        _fork_mode = any(str(arg).startswith("-fork=") and str(arg) != "-fork=0"
                         for arg in args[1:])
        return setup(args, *rest, **options)

    fuzz = module.Fuzz

    def checking_fuzz(*args, **options):
        if _fork_mode and not _coverage_modules:
            sys.stderr.write(
                "WARNING: numba-asan: -fork is used, but no Numba JIT code is compiled "
                "yet. The fork-mode parent cannot track coverage of JIT code compiled "
                "only in child processes, so no corpus accumulates. Call your fuzz "
                "target once before atheris.Fuzz(), e.g. TestOneInput(b\"\"), or use "
                "-jobs/-workers instead of -fork.\n")
        return fuzz(*args, **options)

    relaunching_setup._numba_asan = True
    module.Setup = relaunching_setup
    module.Fuzz = checking_fuzz


_PATCHES = {"atheris": _wrap_setup, "atheris.instrument_bytecode": _wrap_patch_code}


class _Loader(importlib.abc.Loader):
    def __init__(self, loader, patch):
        self.loader = loader
        self.patch = patch

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        # Before other modules bind the names with "from ... import".
        self.patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        patch = _PATCHES.get(name)
        if patch is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader, patch)
        return spec


def install(relaunch=None) -> None:
    """Hook Atheris (when imported) and Numba's bytecode access.

    *relaunch* is an executable that libFuzzer child processes run instead
    of the harness script.
    """
    global _relaunch
    from numba.core import bytecode

    _relaunch = relaunch
    if "atheris" in sys.modules:
        raise RuntimeError("Atheris must not be imported before numba-asan starts")
    if not any(isinstance(finder, _Finder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _Finder())

    identity = bytecode.FunctionIdentity
    from_function = identity.from_function.__func__

    def restoring_from_function(cls, pyfunc):
        restore(bytecode.get_function_object(pyfunc))
        return from_function(cls, pyfunc)

    identity.from_function = classmethod(restoring_from_function)
