"""Atheris harness for Numba JIT code.

Run with: numba-asan run --fuzz examples/fuzz_parser.py -max_total_time=120 -rss_limit_mb=2048
"""
import sys

import atheris
import numpy as np
from numba import njit


@njit(boundscheck=False)
def parse(data):
    if data.size >= 6 and data[0] == 0x4E and data[1] == 0x42 and data[2] == 0x41 and data[3] == 0x21:
        table = np.zeros(4, dtype=np.uint8)
        table[data[4]] = data[5]  # unchecked index from the input
    return 0


def TestOneInput(raw):
    parse(np.frombuffer(raw, dtype=np.uint8))


atheris.Setup(sys.argv, TestOneInput)
atheris.Fuzz()
