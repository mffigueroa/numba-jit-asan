import ctypes
from pathlib import Path
import subprocess
import sys
import uuid
import warnings
import weakref


def enable(*, opt: str, plugin: Path) -> None:
    """Install process-wide CPU instrumentation before importing Numba."""
    if "numba" in sys.modules:
        raise RuntimeError("ASan must be enabled before importing Numba")
    process = ctypes.CDLL(None)
    try:
        process.__asan_version_mismatch_check_v8
    except AttributeError as exc:
        raise RuntimeError(
            "ASan must be preloaded at process startup; use numba-asan run"
        ) from exc

    import llvmlite
    from llvmlite import binding as llvm
    import numba
    from numba.core import caching, codegen

    if (numba.__version__, llvmlite.__version__) != ("0.62.1", "0.45.1"):
        raise RuntimeError("This prototype requires Numba 0.62.1 / llvmlite 0.45.1")
    if llvm.llvm_version_info[0] != 20:
        raise RuntimeError("This prototype requires llvmlite built with LLVM 20")
    if not plugin.is_file():
        raise RuntimeError(f"Missing ASan pass plugin: {plugin}")

    library_class = codegen.JITCodeLibrary
    original_finalize = library_class._finalize_final_module
    original_linking = library_class._get_module_for_linking

    def finalize(library):
        library._asan_pristine_module = library._final_module.clone()
        result = subprocess.run(
            [
                opt, f"-load-pass-plugin={plugin}", "-passes=numba-asan",
                "-S", "-o", "-",
            ],
            input=str(library._final_module),
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                f"LLVM ASan instrumentation failed for {library.name}:\n"
                f"{result.stderr}"
            )
        module = llvm.parse_assembly(result.stdout)
        module.name = library._final_module.name
        token = uuid.uuid4().hex
        callbacks = {}
        for function in module.functions:
            if function.name in ("asan.module_ctor", "asan.module_dtor"):
                kind = function.name.rsplit("_", 1)[-1]
                function.name = f"numba_asan_{kind}_{token}"
                function.linkage = "external"
                callbacks[kind] = function.name
        module.verify()
        library._final_module = module
        original_finalize(library)

        engine = library._codegen._engine
        for kind in ("ctor", "dtor"):
            if kind not in callbacks:
                continue
            address = engine.get_function_address(callbacks[kind])
            if not address:
                raise RuntimeError(f"Cannot resolve ASan module {kind}")
            callback = ctypes.CFUNCTYPE(None)(address)
            if kind == "ctor":
                callback()
            else:
                # Numba retains MCJIT modules even after wrapper libraries die.
                # Redzones must live as long as the engine, not the library.
                weakref.finalize(engine, callback)

    def get_module_for_linking(library):
        library._ensure_finalized()
        sanitized = library._final_module
        library._final_module = library._asan_pristine_module
        try:
            return original_linking(library)
        finally:
            library._final_module = sanitized

    def disable_cache(cache):
        cache.disable()

    library_class._finalize_final_module = finalize
    library_class._get_module_for_linking = get_module_for_linking
    caching.Cache.enable = disable_cache
    warnings.warn(
        "Experimental Numba ASan enabled; all Numba disk caches are disabled. "
        "Native extension code and logical array/view bounds are not covered.",
        RuntimeWarning,
        stacklevel=2,
    )
