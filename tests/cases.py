"""Isolated cases: buggy code must never execute in the test runner process."""
import ctypes
import os
import sys
import threading
from pathlib import Path

import numpy as np
from llvmlite import ir
from numba import cfunc, guvectorize, njit, prange, types, vectorize
from numba.extending import intrinsic


def handler_name():
    """NumPy's active memory handler (``numpy._core`` in 2.x, ``numpy.core`` in 1.x)."""
    try:
        multiarray = np._core.multiarray
    except AttributeError:  # NumPy 1.x: numpy._core is a stub without multiarray
        multiarray = np.core.multiarray
    return multiarray.get_handler_name()


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
        assert query(values.ctypes.data + values.nbytes) == 1
        assert query(values.ctypes.data - 1) == 1
        print("NRT payload boundaries are poisoned")
    elif case == "nrt_one_past":
        values = allocate()
        write(values, values.size)
    elif case == "nrt_underflow":
        values = allocate()
        write(values, -values.size - 1)
    elif case == "nrt_metadata":
        values = allocate()
        # The old layout placed MemInfo and alignment padding before the data.
        write(values, -values.size - 8)
    elif case == "numpy_small_uaf":
        values = np.ones(8, dtype=np.int64)
        pointer = values.ctypes.data
        raw = (ctypes.c_int64 * values.size).from_address(pointer)
        dangling = np.ctypeslib.as_array(raw)
        read(dangling, 0)
        del values
        print(read(dangling, 0))
    elif case == "native_store":
        native = ctypes.CDLL(os.environ["ASAN_TEST_NATIVE_LIBRARY"])
        store = native.unchecked_native_store
        store.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t]
        store.restype = None
        store(values.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), values.size)
        os._exit(0)
    elif case == "numpy_thread":
        state = {}

        def worker():
            state["policy"] = handler_name()

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        assert state["policy"] == "numba_asan_malloc"
        print(state["policy"])
    elif case == "numpy_context":
        import contextvars
        from numba_jit_asan.allocators import install_numpy_policy

        def isolated():
            install_numpy_policy()
            assert handler_name() == "numba_asan_malloc"

        contextvars.Context().run(isolated)
        print("isolated context policy installed")
    elif case == "nrt_destructor":
        from numba.core.runtime import _nrt_python

        native = ctypes.CDLL(str(Path(os.environ["NUMBA_ASAN_PLUGIN"]).with_name(
            "numba_asan_native.so",
        )))
        destructor_type = ctypes.CFUNCTYPE(
            None, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
        )
        calls = []

        @destructor_type
        def destructor(data, size, info):
            calls.append((ctypes.c_uint8.from_address(data).value, size, info))

        alloc = native.numba_asan_alloc_dtor_safe
        alloc.argtypes = [ctypes.c_size_t, destructor_type]
        alloc.restype = ctypes.c_void_p
        metadata = alloc(17, destructor)
        if not metadata:
            raise MemoryError("safe destructor allocation failed")
        release = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(
            _nrt_python.c_helpers["MemInfo_release"],
        )
        release(metadata)
        assert calls == [(0xCB, 17, None)]
        print("safe fill and destructor semantics preserved")
    elif case == "nrt_api":
        from numba.core.runtime import _nrt_python

        class API(ctypes.Structure):
            _fields_ = [(name, ctypes.c_void_p) for name in (
                "allocate", "allocate_external", "manage_memory",
                "acquire", "release", "get_data",
            )]

        get_api = ctypes.CFUNCTYPE(ctypes.POINTER(API))(
            _nrt_python.c_helpers["get_api"],
        )
        api = get_api().contents
        alloc = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t)(api.allocate)
        get_data = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)(api.get_data)
        release = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(api.release)
        metadata = alloc(64)
        if not metadata:
            raise MemoryError("NRT API allocation failed")
        pointer = get_data(metadata)
        raw = (ctypes.c_int64 * 8).from_address(pointer)
        wrapped = np.ctypeslib.as_array(raw)
        write(wrapped, 0)
        assert read(wrapped, 0) == 123
        runtime = ctypes.CDLL(None)
        query = runtime.__asan_address_is_poisoned
        query.argtypes = [ctypes.c_void_p]
        query.restype = ctypes.c_int
        assert query(pointer - 1)
        assert query(pointer + 64)
        release(metadata)
        print("NRT API payload guards preserved")
    elif case.startswith("external_"):
        native = ctypes.CDLL(str(Path(os.environ["NUMBA_ASAN_PLUGIN"]).with_name(
            "numba_asan_native.so",
        )))
        libc = ctypes.CDLL(None)
        libc.malloc.argtypes = [ctypes.c_size_t]
        libc.malloc.restype = ctypes.c_void_p
        libc.free.argtypes = [ctypes.c_void_p]
        libc.mmap.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_size_t,
        ]
        libc.mmap.restype = ctypes.c_void_p
        libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        libc.munmap.restype = ctypes.c_int
        malloc_type = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
        free_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)
        state = {"allocated": None, "freed": None}

        @malloc_type
        def custom_malloc(size, opaque):
            if "mmap" in case:
                assert size < 4096 - 64
                base = libc.mmap(None, 4096, 3, 0x22, -1, 0)
                if base == ctypes.c_void_p(-1).value:
                    return None
                ctypes.c_uint64.from_address(base).value = 0x12345678
                state["mapping"] = base
                pointer = base + 64
            else:
                pointer = libc.malloc(size)
            state["allocated"] = pointer
            return pointer

        @free_type
        def custom_free(pointer, opaque):
            state["freed"] = pointer
            if "mmap" in case:
                assert ctypes.c_uint64.from_address(state["mapping"]).value == 0x12345678
                # Model a pool writing free-list metadata inside its block.
                ctypes.c_uint64.from_address(pointer).value = 0
                assert libc.munmap(state["mapping"], 4096) == 0
            else:
                libc.free(pointer)

        class ExternalAllocator(ctypes.Structure):
            _fields_ = [
                ("malloc", malloc_type), ("realloc", ctypes.c_void_p),
                ("free", free_type), ("opaque", ctypes.c_void_p),
            ]

        class MemInfo(ctypes.Structure):
            _fields_ = [
                ("refcount", ctypes.c_size_t), ("destructor", ctypes.c_void_p),
                ("info", ctypes.c_void_p), ("data", ctypes.c_void_p),
                ("size", ctypes.c_size_t), ("external", ctypes.c_void_p),
            ]

        allocator = ExternalAllocator(custom_malloc, None, custom_free, None)
        alloc = native.numba_asan_alloc_safe_aligned_external
        alloc.argtypes = [ctypes.c_size_t, ctypes.c_uint, ctypes.POINTER(ExternalAllocator)]
        alloc.restype = ctypes.POINTER(MemInfo)
        mi = alloc(64, 64, ctypes.byref(allocator))
        if not mi:
            raise MemoryError("external allocation failed")
        pointer = mi.contents.data
        assert pointer % 64 == 0
        assert pointer != ctypes.addressof(mi.contents)
        raw = (ctypes.c_int64 * 8).from_address(pointer)
        external_values = np.ctypeslib.as_array(raw)
        if case in ("external_overflow", "external_mmap_overflow"):
            write(external_values, external_values.size)
        elif case in ("external_underflow", "external_mmap_underflow"):
            write(external_values, -external_values.size - 1)
        elif case in ("external_safe", "external_mmap_safe"):
            write(external_values, 0)
            assert read(external_values, 0) == 123
            from numba.core.runtime import _nrt_python

            release = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(
                _nrt_python.c_helpers["MemInfo_release"],
            )
            release(mi)
            assert state["freed"] == state["allocated"]
            print("external allocator freed the original base")
        else:
            raise ValueError(case)
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
    elif case == "safe":
        write(values, 0)
        assert read(values, -1) == 1
        assert read(values, 0) == 123
        caller(values, 1)
        assert read(values, 1) == 123
        allocated = allocate()
        assert np.all(allocated == 1)
        @njit
        def varied_allocations():
            from_size = np.empty(0, dtype=np.int64)
            small = np.ones(3, dtype=np.uint8)
            return from_size.size, small.sum()

        assert varied_allocations() == (0, 3)
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
    elif case == "default_unchecked":
        @njit
        def default_read(source, index):
            return source[index]

        print(default_read(values, values.size))
    elif case == "checked":
        @njit(boundscheck=True)
        def checked_read(source, index):
            return source[index]

        try:
            checked_read(values, values.size)
        except IndexError:
            print("explicit bounds checking preserved")
        else:
            raise AssertionError("boundscheck=True was overridden")
    else:
        raise ValueError(f"Unknown case: {case}")
    if os.environ.get("ASAN_TEST_UNINSTRUMENTED"):
        os._exit(0)


if __name__ == "__main__":
    main()
