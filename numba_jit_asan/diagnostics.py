import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


class Diagnostics:
    def __init__(self, native):
        self.native = native
        self.symbolizer = shutil.which("llvm-symbolizer-20")
        self.nm = shutil.which("llvm-nm-20")
        if not self.symbolizer or not self.nm:
            raise RuntimeError("JIT diagnostics require llvm-symbolizer-20 and llvm-nm-20")
        directory = os.environ.get("NUMBA_ASAN_JIT_DIR")
        if directory:
            self.directory = Path(directory).resolve()
            self.directory.mkdir(parents=True, exist_ok=True)
        else:
            self.temporary = tempfile.TemporaryDirectory(prefix="numba-asan-jit-")
            self.directory = Path(self.temporary.name)
        self.native.numba_asan_register_frame.argtypes = [
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
        ]
        self.native.numba_asan_register_frame.restype = None
        self.native.numba_asan_install_report_callback.restype = None
        self.native.numba_asan_install_report_callback()

    def register(self, library, token):
        object_path = self.directory / f"{token}.o"
        object_path.write_bytes(library._get_compiled_object())
        result = subprocess.run(
            [self.nm, "--defined-only", "--print-size", str(object_path)],
            capture_output=True, text=True, check=True,
        )
        engine = library._codegen._engine
        for line in result.stdout.splitlines():
            fields = line.split(maxsplit=3)
            if len(fields) != 4 or fields[2] not in ("t", "T", "w", "W"):
                continue
            offset, size, _, name = fields
            size = int(size, 16)
            if not size:
                continue
            address = engine.get_function_address(name)
            if address:
                self.native.numba_asan_register_frame(
                    address, size, int(offset, 16), name.encode(),
                    os.fsencode(object_path), os.fsencode(self.symbolizer),
                )
