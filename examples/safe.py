import numpy as np
from numba import njit


@njit(boundscheck=False)
def sum_values(values):
    total = 0
    for value in values:
        total += value
    return total


assert sum_values(np.arange(1024, dtype=np.int64)) == 523776
print("safe code passed")
