"""Atheris harnesses for the integration tests (selected by $FUZZ_CASE).

The case is chosen through the environment because libFuzzer interprets
positional command-line arguments as corpus directories or inputs.
"""
import os
import sys

import atheris
import numpy as np
from numba import njit

CASE = os.environ["FUZZ_CASE"]

if CASE == "instrument_imports":
    with atheris.instrument_imports():
        import atheris_module as module
else:
    import atheris_module as module


@njit(boundscheck=False)
def parse(data):
    if data.size >= 6 and data[0] == 0x4E and data[1] == 0x42 and data[2] == 0x41 and data[3] == 0x21:
        table = np.zeros(4, dtype=np.uint8)
        table[data[4]] = data[5]  # atheris: magic overflow
    return 0


@njit(boundscheck=False)
def spin(data):
    total = 0
    if data.size > 0 and data[0] == 0x41:
        for index in range(1 << 62):
            total += index ^ data[index % data.size]  # atheris: hang
    return total


@njit
def shift(data):
    if data.size > 1 and data[0] == 0x43:
        return np.int64(1) << np.int64(data[1])  # atheris: shift
    return 0


def TestOneInput(raw):
    data = np.frombuffer(raw, dtype=np.uint8)
    if CASE in ("parse", "fork", "fork_cold"):
        parse(data)
    elif CASE in ("instrument_imports", "instrument_all"):
        module.entry(raw)
    elif CASE == "hang":
        spin(data)
    elif CASE == "shift":
        shift(data)


if CASE == "instrument_all":
    atheris.instrument_all()
if CASE == "fork":
    TestOneInput(b"")  # compile JIT code before fuzzing, as recommended for -fork
atheris.Setup(sys.argv, TestOneInput)
atheris.Fuzz()
