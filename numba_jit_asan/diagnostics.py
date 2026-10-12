import os
from pathlib import Path
import subprocess

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
        object_path.write_bytes(library._get_compiled_object())
        result = subprocess.run(
            [self.nm, "--defined-only", "--print-size", str(object_path)],
            capture_output=True, text=True, check=True,
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
