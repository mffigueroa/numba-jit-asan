import argparse
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig


DEFAULT_PLUGIN = Path(".build/numba_asan_pass.so")


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"Missing {name}; install the LLVM 20 toolchain")
    return path


def _llvm20(tool: str) -> None:
    version = subprocess.check_output([tool, "--version"], text=True)
    if not re.search(r"\b(?:version )?20\.", version):
        raise RuntimeError(f"Expected LLVM 20: {tool}\n{version}")


def build_plugin(output: Path) -> None:
    config = _tool("llvm-config-20")
    compiler = _tool("clang++-20")
    _llvm20(config)
    _llvm20(compiler)
    flags = shlex.split(subprocess.check_output(
        [config, "--cxxflags", "--ldflags", "--system-libs", "--libs",
         "passes", "instrumentation"], text=True,
    ))
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [compiler, "-shared", "-fPIC", str(Path(__file__).with_name("asan_pass.cpp")),
         *flags, "-o", str(output)],
        check=True,
    )
    import numpy

    native = output.with_name("numba_asan_native.so")
    subprocess.run(
        [
            compiler, "-shared", "-fPIC", "-g", "-O1", "-std=c++17",
            "-fsanitize=address", "-fno-omit-frame-pointer",
            f"-I{numpy.get_include()}", f"-I{sysconfig.get_path('include')}",
            str(Path(__file__).with_name("native.cpp")), "-o", str(native),
        ],
        check=True,
    )
    print(f"Built {output.resolve()}")
    print(f"Built {native.resolve()}")


def _runtime() -> Path:
    compiler = _tool("clang-20")
    _llvm20(compiler)
    path = Path(subprocess.check_output(
        [compiler, "-print-file-name=libclang_rt.asan-x86_64.so"], text=True,
    ).strip())
    if not path.is_file():
        raise RuntimeError("Missing LLVM 20 ASan runtime; install libclang-rt-20-dev")
    return path.resolve()


def launch(plugin: Path, target: list[str]) -> None:
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise RuntimeError("This prototype currently supports Linux x86-64 only")
    if not plugin.is_file():
        raise RuntimeError("ASan plugin is missing; run numba-asan build first")
    if not plugin.with_name("numba_asan_native.so").is_file():
        raise RuntimeError("ASan native bridge is missing; run numba-asan build again")
    opt = _tool("opt-20")
    _llvm20(opt)
    runtime = _runtime()
    env = os.environ.copy()
    if "libasan" in env.get("LD_PRELOAD", "") or "libclang_rt.asan" in env.get("LD_PRELOAD", ""):
        raise RuntimeError("Remove an existing ASan LD_PRELOAD before using this launcher")
    env["LD_PRELOAD"] = str(runtime) + (
        ":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else ""
    )
    env["PYTHONMALLOC"] = "malloc"
    # CPython and uninstrumented extensions have process-exit leaks. Opt in to LSan.
    env["ASAN_OPTIONS"] = "detect_leaks=0:" + env.get("ASAN_OPTIONS", "")
    env["NUMBA_ASAN_OPT"] = opt
    env["NUMBA_ASAN_PLUGIN"] = str(plugin.resolve())
    os.execve(
        sys.executable,
        [sys.executable, "-m", "numba_jit_asan._bootstrap", *target],
        env,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run real LLVM AddressSanitizer on Numba CPU JIT code",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build the LLVM 20 ASan pass plugin")
    build.add_argument("--output", type=Path, default=DEFAULT_PLUGIN)
    run = commands.add_parser("run", help="run a Python script or -m module under ASan")
    run.add_argument("--plugin", type=Path, default=DEFAULT_PLUGIN)
    run.add_argument("target", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        if args.command == "build":
            build_plugin(args.output)
        else:
            target = args.target
            if target and target[0] == "--":
                target = target[1:]
            if not target or (target[0] == "-m" and len(target) < 2):
                parser.error("run requires a script path or -- -m module")
            launch(args.plugin, target)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"numba-asan: {exc}\n")


if __name__ == "__main__":
    main()
