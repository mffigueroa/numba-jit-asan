"""Library code fuzzed through Atheris's Python bytecode instrumentation."""
import numpy as np
from numba import njit


@njit(boundscheck=False)
def checksum(data):
    if data.size > 2 and data[0] == 0x7F and data[1] == 0x45:
        lookup = np.zeros(8, dtype=np.int64)
        return lookup[data[2]]  # atheris module: overflow
    return 0


def entry(raw):
    if not raw:
        return -1
    return checksum(np.frombuffer(raw, dtype=np.uint8))
