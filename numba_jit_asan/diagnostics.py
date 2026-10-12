import os
from pathlib import Path
import subprocess
import sys

from . import toolchain


class Diagnostics:
    """Records JIT object code and address ranges for report symbolization."""

    def __init__(self):
        self.nm = toolchain.which("llvm-nm")
        if not self.nm:
            raise RuntimeError(f"JIT diagnostics require {toolchain.tool_name('llvm-nm')}")
        self.directory = Path(os.environ["NUMBA_SANITIZER_JIT_DIR"]).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.frames = self.directory / f"frames.{os.getpid()}.tsv"

    def register(self, library, token):
        if self.frames.name != f"frames.{os.getpid()}.tsv":
            # Forked child: parent ranges stay valid; record new ones separately.
            self.frames = self.directory / f"frames.{os.getpid()}.tsv"
        object_path = self.directory / f"{token}.o"
        try:
            compiled = library._get_compiled_object()
        except RuntimeError:
            # MCJIT compiled this module together with another library's
            # finalize_object(), so no object was captured for it. Its frames
            # stay unsymbolized; instrumentation is unaffected.
            print(f"numba-asan: no object code captured for {library.name}; its JIT "
                  "frames will not be symbolized", file=sys.stderr)
            return
        object_path.write_bytes(compiled)
        result = subprocess.run(
            [self.nm, "--defined-only", "--print-size", str(object_path)],
            capture_output=True, text=True, check=True, timeout=120,
            env=toolchain.clean_env(), stdin=subprocess.DEVNULL,
        )
        engine = library._codegen._engine
        rows = []
        for line in result.stdout.splitlines():
            fields = line.split(maxsplit=3)
            if len(fields) != 4 or fields[2] not in ("t", "T", "w", "W"):
                continue
            offset, size, _, name = fields
            size = int(size, 16)
            address = engine.get_function_address(name) if size else 0
            if address:
                rows.append(f"{address:x}\t{size:x}\t{offset}\t{name}\t{object_path}\n")
        with self.frames.open("a") as frames:
            frames.writelines(rows)
