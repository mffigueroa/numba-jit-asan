"""Scoped LeakSanitizer checks for Python processes.

Leak checks at interpreter exit report CPython's intentionally retained
allocations. Checking while Python is still alive reports only allocations
that are unreachable from any live memory, such as leaked NRT payloads.
"""
import ctypes
import gc
import sys


class LeakError(AssertionError):
    pass


# CPython 3.11+ ``PyFrameObject``: PyObject_HEAD, f_back, then f_frame.
_F_FRAME_OFFSET = 3 * ctypes.sizeof(ctypes.c_void_p)


def _runtime():
    runtime = ctypes.CDLL(None)
    try:
        check = runtime.__lsan_do_recoverable_leak_check
    except AttributeError as exc:
        raise RuntimeError("LeakSanitizer requires numba-asan run --leaks") from exc
    check.restype = ctypes.c_int
    for name in ("__lsan_register_root_region", "__lsan_unregister_root_region"):
        getattr(runtime, name).argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    locate = runtime.__asan_locate_address
    locate.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
                       ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)]
    locate.restype = ctypes.c_char_p
    return runtime, check


def _on_heap(runtime, address):
    start, size = ctypes.c_void_p(), ctypes.c_size_t()
    kind = runtime.__asan_locate_address(address, None, 0, ctypes.byref(start),
                                         ctypes.byref(size))
    return kind == b"heap"


def _frame_stack_regions(runtime):
    """Mappings holding active interpreter frames.

    CPython 3.11+ keeps active frames (their locals and frame objects) in
    mmap-backed data-stack chunks that LeakSanitizer never scans; objects
    referenced only from running frames would otherwise be reported.
    """
    if sys.version_info < (3, 11) or sys.implementation.name != "cpython":
        return []
    addresses = set()
    for frame in list(sys._current_frames().values()):
        while frame is not None:
            pointer = ctypes.c_void_p.from_address(id(frame) + _F_FRAME_OFFSET).value
            # Completed frames live inside heap frame objects: already scanned.
            if pointer and not _on_heap(runtime, pointer):
                addresses.add(pointer)
            frame = frame.f_back
    regions = set()
    with open("/proc/self/maps") as maps:
        for line in maps:
            fields = line.split()
            start, end = (int(value, 16) for value in fields[0].split("-"))
            # Anonymous, writable mappings only.
            if len(fields) == 5 and fields[1].startswith("rw") and any(
                start <= address < end for address in addresses
            ):
                regions.add((start, end - start))
    return sorted(regions)


def check(raise_on_leak: bool = True) -> bool:
    """Run a leak check now; report details to the sanitizer log."""
    runtime, leak_check = _runtime()
    gc.collect()
    regions = _frame_stack_regions(runtime)
    for start, size in regions:
        runtime.__lsan_register_root_region(start, size)
    try:
        leaked = bool(leak_check())
    finally:
        for start, size in regions:
            runtime.__lsan_unregister_root_region(start, size)
    if leaked and raise_on_leak:
        raise LeakError("LeakSanitizer detected leaks; see the sanitizer report")
    return leaked
