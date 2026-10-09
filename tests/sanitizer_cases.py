"""Isolated cases for UBSan, TSan, MSan, LSan and native dependency coverage.

Buggy cases must never execute in the test runner process.
"""
import ctypes
import sys
import threading

import numpy as np
from numba import njit, prange, types
from numba.extending import intrinsic


# --- UndefinedBehaviorSanitizer -------------------------------------------
@njit
def shift(value, amount):
    return value << amount  # ubsan: shift


@njit
def to_int32(value):
    return np.int32(value)  # ubsan: float to int32


@njit
def to_int64(value):
    return int(value)  # ubsan: float to int64


def ub_safe():
    assert shift(1, 3) == 8 and shift(-1, 63) == -(2**63)
    assert to_int32(5.9) == 5 and to_int32(-2147483648.9) == -(2**31)
    assert to_int32(2147483647.9) == 2**31 - 1
    assert to_int64(-1.5) == -1
    print("in-range shifts and casts passed")


# --- ThreadSanitizer --------------------------------------------------------
@njit(nogil=True)
def accumulate(counter, values):
    for index in range(values.size):
        counter[0] += values[index]  # tsan: shared counter


@njit(parallel=True)
def racy_sum(values):
    total = np.zeros(1)
    for index in prange(values.size):
        total[0] += values[index]  # tsan: prange shared element
    return total[0]


@njit(parallel=True)
def reduction_sum(values):
    total = 0.0
    for index in prange(values.size):
        total += values[index]
    return total


def _threads(locked):
    counter = np.zeros(1, dtype=np.int64)
    values = np.ones(2_000_000, dtype=np.int64)
    accumulate(counter, values[:1])
    counter[0] = 0
    lock = threading.Lock()

    def work():
        if locked:
            with lock:
                accumulate(counter, values)
        else:
            accumulate(counter, values)

    workers = [threading.Thread(target=work) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    print("counter", counter[0])


def race_threads():
    _threads(locked=False)


def race_locked():
    _threads(locked=True)
    print("locked threads passed")


def race_prange():
    print(racy_sum(np.ones(100_000)))


def race_reduction():
    assert reduction_sum(np.ones(100_000)) == 100_000
    print("prange reduction passed")


# --- MemorySanitizer --------------------------------------------------------
@njit
def total(count, initialize):
    values = np.zeros(count) if initialize else np.empty(count)
    result = 0.0
    for index in range(count):
        result += values[index]  # msan: accumulate
    return result  # msan: escape


@njit
def unwritten_branch(count):
    values = np.empty(count, dtype=np.int64)
    values[0] = 1
    if values[count - 1] > 0:  # msan: branch
        return 1
    return 0


@njit
def positive_sum(values):
    result = 0.0
    for value in values:
        if value > 0.5:
            result += value
    return result


def uninit_safe():
    assert total(16, True) == 0.0
    # Data written by uninstrumented NumPy/CPython must not be reported.
    assert positive_sum(np.arange(10.0)) == 45.0
    assert positive_sum(np.full(4, 2.0)) == 8.0
    data = np.empty(8)
    data[:] = 3.0
    assert positive_sum(data) == 24.0
    print("initialized data passed")


def uninit_escape():
    print(total(16, False))


def uninit_branch():
    print(unwritten_branch(16))


# --- LeakSanitizer ----------------------------------------------------------
@intrinsic
def leak_meminfo(typing_context, size):
    def codegen(context, builder, signature, args):
        context.nrt.meminfo_alloc(builder, args[0])  # never released
        return context.get_dummy_value()

    return types.none(size), codegen


@njit
def leaky(size):
    leak_meminfo(size)  # lsan: leak


@njit
def balanced(size):
    return np.ones(size).sum()


def leak_none():
    # Objects referenced only by active Python frames must not be reported.
    frames = [sys._getframe(), sys._getframe(1)]
    assert balanced(4096) == 4096 and frames
    print("no leaks")


def leak_jit():
    leaky(4096)
    print("leaked")


def leak_scoped():
    from numba_jit_asan.leaks import LeakError, check

    assert check() is False
    leaky(4096)
    try:
        check()
    except LeakError:
        print("scoped leak check raised")
        sys.stdout.flush()
        raise SystemExit(0)
    raise AssertionError("leak not detected")


# --- Native dependency instrumentation -------------------------------------
def _libc():
    libc = ctypes.CDLL(None)
    libc.malloc.restype = ctypes.c_void_p
    libc.malloc.argtypes = [ctypes.c_size_t]
    libc.free.argtypes = [ctypes.c_void_p]
    return libc


def native_numpy():
    libc = _libc()
    pointer = libc.malloc(8 * 1024)
    values = np.ctypeslib.as_array((ctypes.c_double * 1024).from_address(pointer))
    values[:] = 1.0
    libc.free(pointer)
    print("numpy sum", np.sum(values))  # NumPy's C reduction reads freed memory


def native_numba():
    from numba.core.runtime import _nrt_python

    balanced(1)

    def helper(name, *signature):
        return ctypes.CFUNCTYPE(*signature)(_nrt_python.c_helpers[name])

    allocate = helper("MemInfo_alloc", ctypes.c_void_p, ctypes.c_size_t)
    release = helper("MemInfo_release", None, ctypes.c_void_p)
    data = helper("MemInfo_data", ctypes.c_void_p, ctypes.c_void_p)
    meminfo = allocate(64)
    release(meminfo)  # reference count 1 -> 0 frees the MemInfo
    print("stale data pointer", data(meminfo))  # Numba's C runtime reads freed memory


CASES = {name: function for name, function in globals().items()
         if callable(function) and name.split("_")[0] in
         ("ub", "race", "uninit", "leak", "native")}


if __name__ == "__main__":
    case = sys.argv[1]
    if case == "ub_shift":
        print(shift(1, 70))
    elif case == "ub_negative_shift":
        print(shift(1, -1))
    elif case == "ub_cast":
        print(to_int32(1e300))
    elif case == "ub_nan":
        print(to_int64(float("nan")))
    else:
        CASES[case]()
