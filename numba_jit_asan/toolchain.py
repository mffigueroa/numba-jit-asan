"""Validated Numba/llvmlite releases and the LLVM toolchain matching each.

The sanitizer pass plugin and runtimes must come from the same LLVM major
version that llvmlite embeds, and the private Numba hooks are only validated
for these exact releases.
"""
from importlib import metadata
import os
import re
import shutil
import subprocess


# (numba, llvmlite): LLVM major version embedded in that llvmlite release.
SUPPORTED = {
    ("0.62.1", "0.45.1"): 20,
    ("0.68.0", "0.50.0"): 22,
}


def _describe() -> str:
    return ", ".join(f"Numba {numba} / llvmlite {llvmlite} (LLVM {major})"
                     for (numba, llvmlite), major in SUPPORTED.items())


def llvm_major(numba_version: str | None = None, llvmlite_version: str | None = None) -> int:
    """LLVM major version for the installed (or given) Numba/llvmlite pair.

    Uses package metadata so the caller need not import Numba or llvmlite.
    """
    try:
        key = (numba_version or metadata.version("numba"),
               llvmlite_version or metadata.version("llvmlite"))
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"{exc.name} is not installed in this environment") from exc
    if key not in SUPPORTED:
        raise RuntimeError(f"Unsupported Numba {key[0]} / llvmlite {key[1]}; "
                           f"validated releases: {_describe()}")
    return SUPPORTED[key]


def clean_env(base=None) -> dict:
    """Environment for LLVM tools: never with a sanitizer runtime preloaded.

    Tools are uninstrumented. A foreign sanitizer runtime (for example
    Atheris's prebuilt one) preloaded into them can loop endlessly printing
    DEADLYSIGNAL, and a matching one only adds overhead.
    """
    env = dict(os.environ if base is None else base)
    env.pop("LD_PRELOAD", None)
    return env


def tool_name(name: str) -> str:
    return f"{name}-{llvm_major()}"


def which(name: str) -> str | None:
    return shutil.which(tool_name(name))


def tool(name: str) -> str:
    """Path to the versioned LLVM tool, verified to report the expected version."""
    major = llvm_major()
    versioned = f"{name}-{major}"
    path = shutil.which(versioned)
    if path is None:
        raise RuntimeError(f"Missing {versioned}; install the LLVM {major} toolchain")
    version = subprocess.run(
        [path, "--version"], env=clean_env(), stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=60, check=True,
    ).stdout
    if not re.search(rf"\b(?:version )?{major}\.", version):
        raise RuntimeError(f"Expected LLVM {major}: {path}\n{version}")
    return path
