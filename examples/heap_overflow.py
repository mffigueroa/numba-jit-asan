"""Run with numba-asan run examples/heap_overflow.py."""
import numpy as np
from numba import njit


@njit(boundscheck=False)
def write(values, index):
    values[index] = 123


values = np.ones(1024, dtype=np.int64)
write(values, values.size)
