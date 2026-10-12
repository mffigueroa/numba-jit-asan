import ctypes
import os
from pathlib import Path
import subprocess
import sys
import uuid
import warnings
import weakref

from .toolchain import clean_env


SANITIZERS = {
    # name: (runtime probe symbol, LLVM pipeline, constructor prefix)
    "address": ("__asan_version_mismatch_check_v8", "numba-asan", "asan"),
    "thread": ("__tsan_init", "numba-tsan", "tsan"),
    "memory": ("__msan_init", "numba-msan", "msan"),
}


def _run_opt(opt, plugin, pipeline, module_text, name):
    result = subprocess.run(
        [opt, f"-load-pass-plugin={plugin}", f"-passes={pipeline}", "-S", "-o", "-"],
        input=module_text, text=True, capture_output=True, check=False,
        # This runs inside the sanitized process: opt needs no sanitizer runtime.
        env=clean_env(),
    )
    if result.returncode:
        raise RuntimeError(f"LLVM {pipeline} instrumentation failed for {name}:\n"
                           f"{result.stderr}")
    return result.stdout


def enable(*, opt: str, plugin: Path, sanitizer: str = "address",
           undefined: bool = False, coverage: bool = False) -> None:
    """Install process-wide CPU instrumentation before importing Numba."""
    if "numba" in sys.modules:
        raise RuntimeError("Sanitizers must be enabled before importing Numba")
    probe, pipeline, prefix = SANITIZERS[sanitizer]
    if coverage:
        # libFuzzer-compatible SanitizerCoverage for coverage-guided fuzzing.
        pipeline = f"numba-sancov,{pipeline}"
    try:
        getattr(ctypes.CDLL(None), probe)
    except AttributeError as exc:
        raise RuntimeError(
            f"The {sanitizer} sanitizer runtime must be loaded at process startup; "
            "use numba-asan run"
        ) from exc

    import llvmlite
    from llvmlite import binding as llvm
    import numba
    from numba.core import caching, codegen

    from .toolchain import llvm_major

    major = llvm_major(numba.__version__, llvmlite.__version__)
    if llvm.llvm_version_info[0] != major:
        raise RuntimeError(f"This prototype requires llvmlite built with LLVM {major}")
    if not plugin.is_file():
        raise RuntimeError(f"Missing sanitizer pass plugin: {plugin}")

    from . import atheris_compat
    from .diagnostics import Diagnostics
    from numba.core.cpu import CPUTargetOptions
    from numba.np.ufunc.ufuncbuilder import UFuncTargetOptions

    keepalive = []
    if sanitizer == "address":
        from .allocators import install

        keepalive.append(install(plugin.with_name("numba_asan_native.so")))
    elif sanitizer == "memory":
        from .allocators import install_msan

        keepalive.append(install_msan(plugin.with_name("numba_msan_native.so")))
    diagnostics = Diagnostics()
    atheris_compat.install(os.environ.get("NUMBA_ASAN_RELAUNCH"))
    for options_class in (CPUTargetOptions, UFuncTargetOptions):
        original_options = options_class.finalize

        def finalize_options(self, flags, options, original=original_options):
            original(self, flags, options)
            # Resolve boundscheck defaults before enabling debug info.
            flags.debuginfo = True

        options_class.finalize = finalize_options

    cpu_library = codegen.CPUCodeLibrary
    original_add_module = cpu_library.add_llvm_module
    if undefined:
        def add_llvm_module(library, ll_module):
            # Before optimization: LLVM may otherwise exploit the poison values.
            name = ll_module.name
            text = _run_opt(opt, plugin, "numba-ubsan", str(ll_module), name)
            ll_module = llvm.parse_assembly(text)
            ll_module.name = name
            ll_module.verify()
            return original_add_module(library, ll_module)

        cpu_library.add_llvm_module = add_llvm_module

    library_class = codegen.JITCodeLibrary
    original_finalize = library_class._finalize_final_module
    original_linking = library_class._get_module_for_linking
    callback_names = {
        f"{prefix}.module_ctor": "ctor", f"{prefix}.module_dtor": "dtor",
        "sancov.module_ctor_8bit_counters": "ctor",
    }

    def finalize(library):
        library._sanitizer_keepalive = keepalive
        if not library._object_caching_enabled:
            library.enable_object_caching()
        library._sanitizer_pristine_module = library._final_module.clone()
        text = _run_opt(opt, plugin, pipeline, str(library._final_module), library.name)
        module = llvm.parse_assembly(text)
        module.name = library._final_module.name
        token = uuid.uuid4().hex
        callbacks = []
        for function in module.functions:
            kind = callback_names.get(function.name)
            if kind:
                # Unique external names: MCJIT never runs llvm.global_ctors.
                origin = function.name.split(".")[0]
                function.name = f"numba_{origin}_{kind}_{token}"
                function.linkage = "external"
                callbacks.append((kind, function.name))
        module.verify()
        library._final_module = module
        original_finalize(library)
        diagnostics.register(library, token)

        engine = library._codegen._engine
        for kind, name in callbacks:
            address = engine.get_function_address(name)
            if not address:
                raise RuntimeError(f"Cannot resolve {prefix} module {kind}")
            callback = ctypes.CFUNCTYPE(None)(address)
            if kind == "ctor":
                callback()
                if name.startswith("numba_sancov_"):
                    atheris_compat.note_coverage_module()
            else:
                # Numba retains MCJIT modules even after wrapper libraries die.
                # Registrations must live as long as the engine, not the library.
                weakref.finalize(engine, callback)

    def get_module_for_linking(library):
        library._ensure_finalized()
        sanitized = library._final_module
        library._final_module = library._sanitizer_pristine_module
        try:
            return original_linking(library)
        finally:
            library._final_module = sanitized

    def disable_cache(cache):
        cache.disable()

    library_class._finalize_final_module = finalize
    library_class._get_module_for_linking = get_module_for_linking
    caching.Cache.enable = disable_cache
    checks = sanitizer + (" + undefined-behavior checks" if undefined else "") + (
        " + fuzzer coverage" if coverage else "")
    warnings.warn(
        f"Experimental Numba sanitizer enabled ({checks}); all Numba disk caches "
        "are disabled. Unrebuilt native extensions and logical array/view bounds "
        "remain uncovered.",
        RuntimeWarning,
        stacklevel=2,
    )
