import ctypes
from pathlib import Path
import threading


_native = None


def install_numpy_policy() -> None:
    """Install the ASan policy in this thread/context after enabling the launcher."""
    if _native is None:
        raise RuntimeError("NumPy ASan policy requires the numba-asan launcher")
    if _native.numba_asan_install_numpy() != 0:
        raise RuntimeError("Failed to install NumPy ASan memory policy")


def install(native_path: Path) -> ctypes.PyDLL:
    global _native
    from numba.core.runtime import _nrt_python

    native = ctypes.PyDLL(str(native_path))
    native.numba_asan_init_nrt.argtypes = [ctypes.c_void_p]
    native.numba_asan_init_nrt.restype = None
    native.numba_asan_init_nrt(
        _nrt_python.c_helpers["MemInfo_alloc_dtor"],
    )
    get_api = ctypes.CFUNCTYPE(ctypes.c_void_p)(_nrt_python.c_helpers["get_api"])
    native.numba_asan_init_api.argtypes = [ctypes.c_void_p]
    native.numba_asan_init_api.restype = None
    native.numba_asan_init_api(get_api())
    native.numba_asan_install_numpy.restype = ctypes.c_int
    _native = native
    install_numpy_policy()
    original_run = threading.Thread.run

    def run_with_policy(thread):
        install_numpy_policy()
        return original_run(thread)

    threading.Thread.run = run_with_policy
    replacements = {
        "get_api": "get_api",
        "MemInfo_alloc": "alloc",
        "MemInfo_alloc_safe": "alloc_safe",
        "MemInfo_alloc_aligned": "alloc_aligned",
        "MemInfo_alloc_safe_aligned": "alloc_safe_aligned",
        "MemInfo_alloc_dtor": "alloc_dtor",
        "MemInfo_alloc_dtor_safe": "alloc_dtor_safe",
        "MemInfo_alloc_safe_aligned_external": "alloc_safe_aligned_external",
    }
    for helper, suffix in replacements.items():
        function = getattr(native, f"numba_asan_{suffix}")
        _nrt_python.c_helpers[helper] = ctypes.cast(function, ctypes.c_void_p).value
    return native
