"""Isolated cases: buggy code must never execute in the test runner process."""
import ctypes
import os
import sys
import threading

import numpy as np
from llvmlite import ir
from numba import cfunc, guvectorize, njit, prange, types, vectorize
from numba.extending import intrinsic


@njit(boundscheck=False, cache=True)
def write(values, index):
    values[index] = 123


@njit(boundscheck=False)
def read(values, index):
    return values[index]


@njit(boundscheck=False, inline="never")
def caller(values, index):
    write(values, index)


@njit(boundscheck=False)
def allocate():
    return np.ones(1024, dtype=np.int64)


@njit(boundscheck=False, parallel=True)
def parallel_write(values, count):
    for index in prange(count):
        values[index] = index


GLOBAL_VALUES = np.arange(1024, dtype=np.int64)


@njit(boundscheck=False)
def global_read(index):
    return GLOBAL_VALUES[index]


@intrinsic
def stack_read(typing_context, index, seed):
    signature = types.int64(index, seed)

    def codegen(context, builder, signature, args):
        integer = ir.IntType(64)
        buffer = builder.alloca(ir.ArrayType(integer, 4))
        zero = ir.Constant(integer, 0)
        for slot in range(4):
            offset = ir.Constant(integer, slot)
            pointer = builder.gep(buffer, [zero, offset])
            builder.store(builder.add(args[1], offset), pointer)
        return builder.load(builder.gep(buffer, [zero, args[0]]))

    return signature, codegen


@njit(boundscheck=False)
def read_stack(index, seed):
    return stack_read(index, seed)


def main():
    case = sys.argv[1]
    values = np.ones(1024, dtype=np.int64)
    if case == "write":
        write(values, values.size)
    elif case == "read":
        print(read(values, values.size))
    elif case == "underflow":
        # Numba wraps -1, but an index less than -size remains out of bounds.
        print(read(values, -values.size - 1))
    elif case == "linked":
        caller(values, values.size)
    elif case == "nrt_overflow":
        values = allocate()
        write(values, values.size + 16)
    elif case == "nrt_padding":
        values = allocate()
        runtime = ctypes.CDLL(None)
        query = runtime.__asan_address_is_poisoned
        query.argtypes = [ctypes.c_void_p]
        query.restype = ctypes.c_int
        assert query(values.ctypes.data + values.nbytes) == 0
        print("NRT one-past-end padding is unpoisoned")
    elif case == "view":
        view = values[:8]
        assert read(view, 8) == 1
        print("logical view overflow is inside backing allocation")
    elif case == "uaf":
        libc = ctypes.CDLL(None)
        libc.malloc.argtypes = [ctypes.c_size_t]
        libc.malloc.restype = ctypes.c_void_p
        libc.free.argtypes = [ctypes.c_void_p]
        pointer = libc.malloc(values.nbytes)
        if not pointer:
            raise MemoryError("malloc failed")
        raw = (ctypes.c_int64 * values.size).from_address(pointer)
        dangling = np.ctypeslib.as_array(raw)
        read(dangling, 0)
        libc.free(pointer)
        print(read(dangling, 0))
    elif case == "global":
        print(global_read(GLOBAL_VALUES.size))
    elif case == "stack":
        print(read_stack(4, 5))
    elif case == "cfunc":
        @cfunc(types.int64(types.CPointer(types.int64), types.int64))
        def native_read(pointer, index):
            return pointer[index]

        pointer = values.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        print(native_read.ctypes(pointer, values.size))
    elif case == "parallel":
        parallel_write(values, values.size + 1)
    elif case == "vectorize":
        @vectorize([types.int64(types.int64)], nopython=True)
        def vector_read(index):
            return GLOBAL_VALUES[index]

        print(vector_read(np.array([values.size], dtype=np.int64)))
    elif case == "guvectorize":
        @guvectorize([(types.int64[:], types.int64[:])], "(n)->()", nopython=True)
        def gufunc_read(source, output):
            output[0] = source[source.size]

        print(gufunc_read(values))
    elif case == "leak":
        read(values, 0)
        libc = ctypes.CDLL(None)
        libc.malloc.argtypes = [ctypes.c_size_t]
        libc.malloc.restype = ctypes.c_void_p
        worker = threading.Thread(target=lambda: libc.malloc(65536))
        worker.start()
        worker.join()
        runtime = ctypes.CDLL(None)
        check = runtime.__lsan_do_recoverable_leak_check
        check.restype = ctypes.c_int
        result = check()
        os._exit(1 if result else 0)
    elif case == "safe":
        write(values, 0)
        assert read(values, -1) == 1
        assert read(values, 0) == 123
        caller(values, 1)
        assert read(values, 1) == 123
        allocated = allocate()
        assert np.all(allocated == 1)
        parallel_write(values, values.size)
        assert np.array_equal(values, np.arange(values.size))
        assert global_read(1) == 1
        assert read_stack(1, 5) == 6
        print("safe cases passed")
    elif case == "ir":
        read(values, 0)
        ir = read.inspect_llvm(read.signatures[0])
        assert "call void @__asan_report_load8" in ir
        assert "sanitize_address" in ir
        print("real ASan load instrumentation is present")
    else:
        raise ValueError(f"Unknown case: {case}")
    if os.environ.get("ASAN_TEST_UNINSTRUMENTED"):
        os._exit(0)


if __name__ == "__main__":
    main()
