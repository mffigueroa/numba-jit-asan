import argparse
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import signal
import sys
import sysconfig
import tempfile

from .symbolize import FrameMap, rewrite
from .toolchain import clean_env, llvm_major, tool as _tool, which as which_tool


# Per environment: artifacts must match this environment's Numba/LLVM, and
# `numba-asan run` then works from any directory.
DEFAULT_PLUGIN = (Path(os.environ["NUMBA_ASAN_BUILD_DIR"]) if os.environ.get("NUMBA_ASAN_BUILD_DIR")
                  else Path(sys.prefix) / "share" / "numba-jit-asan") / "numba_asan_pass.so"


def _numpy_include() -> Path:
    """NumPy's C headers, located without importing NumPy.

    After build-deps, NumPy's extensions only import under the ASan runtime.
    """
    import importlib.util

    spec = importlib.util.find_spec("numpy")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("NumPy is not installed in this environment")
    for location in spec.submodule_search_locations:
        for candidate in (Path(location) / "_core" / "include",
                          Path(location) / "core" / "include"):
            if (candidate / "numpy" / "arrayobject.h").is_file():
                return candidate
    raise RuntimeError("Cannot find NumPy's C headers")


def build_plugin(output: Path) -> None:
    config = _tool("llvm-config")
    compiler = _tool("clang++")
    flags = shlex.split(subprocess.check_output(
        [config, "--cxxflags", "--ldflags", "--system-libs", "--libs",
         "passes", "instrumentation"], text=True, env=clean_env(),
    ))
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [compiler, "-shared", "-fPIC", str(Path(__file__).with_name("asan_pass.cpp")),
         *flags, "-o", str(output)],
        check=True, env=clean_env(),
    )
    native = output.with_name("numba_asan_native.so")
    subprocess.run(
        [
            compiler, "-shared", "-fPIC", "-g", "-O1", "-std=c++17",
            "-fsanitize=address", "-fno-omit-frame-pointer",
            f"-I{_numpy_include()}", f"-I{sysconfig.get_path('include')}",
            str(Path(__file__).with_name("native.cpp")), "-o", str(native),
        ],
        check=True, env=clean_env(),
    )
    print(f"Built {output.resolve()}")
    print(f"Built {native.resolve()}")
    _build_fuzzer_runtime(compiler, output.parent)
    _build_msan(compiler, output.parent)


FUZZER_RUNTIME = "asan_with_fuzzer.so"


def _build_fuzzer_runtime(compiler: str, directory: Path) -> None:
    """LLVM ASan runtime with libFuzzer linked in, for Atheris.

    Mirrors Atheris's setup_utils/merge_libfuzzer_sanitizer.sh, but uses the
    same LLVM release as llvmlite and the JIT instrumentation. Atheris's
    prebuilt runtime is older; on current kernels it can loop endlessly
    printing DEADLYSIGNAL.
    """
    linux = Path(subprocess.check_output(
        [compiler, "-print-file-name=libclang_rt.asan-x86_64.a"], text=True, env=clean_env(),
    ).strip()).parent
    asan = linux / "libclang_rt.asan-x86_64.a"
    fuzzer = linux / "libclang_rt.fuzzer_no_main-x86_64.a"
    if not asan.is_file() or not fuzzer.is_file():
        print(f"Skipped Atheris runtime: install libclang-rt-{llvm_major()}-dev")
        return
    output = directory / FUZZER_RUNTIME
    with tempfile.TemporaryDirectory() as scratch:
        archive = Path(scratch) / "asan.a"
        shutil.copyfile(asan, archive)
        # The preinit object only works in executables, not shared libraries.
        subprocess.run(["ar", "d", str(archive), "asan_preinit.cpp.o"], check=True,
                       env=clean_env())
        subprocess.run(
            [compiler, "-shared", "-Wl,--whole-archive", str(fuzzer), str(archive),
             "-Wl,--no-whole-archive", "-lpthread", "-ldl", "-o", str(output)],
            check=True, env=clean_env(),
        )
    print(f"Built {output.resolve()}")


def _build_msan(compiler: str, directory: Path) -> None:
    """MSan needs its runtime in the executable: build a CPython entry point."""
    library = sysconfig.get_config_var("LDLIBRARY") or ""
    libdir = sysconfig.get_config_var("LIBDIR") or ""
    if not library.endswith(".so") or not (Path(libdir) / library).is_file():
        print("Skipped MemorySanitizer: this Python has no shared libpython")
        return
    # Origin tracking must match the JIT instrumentation (TrackOrigins=2).
    common = ["-g", "-O1", "-fsanitize=memory", "-fsanitize-memory-track-origins=2",
              "-fno-omit-frame-pointer"]
    native = directory / "numba_msan_native.so"
    subprocess.run(
        [compiler, "-shared", "-fPIC", "-std=c++17", *common,
         str(Path(__file__).with_name("native.cpp")), "-o", str(native)],
        check=True, env=clean_env(),
    )
    executable = directory / "numba-msan-python"
    name = library.removeprefix("lib").removesuffix(".so")
    subprocess.run(
        [compiler.replace("clang++", "clang"), *common,
         f"-I{sysconfig.get_path('include')}",
         str(Path(__file__).with_name("msan_python.c")),
         f"-L{libdir}", f"-Wl,-rpath,{libdir}", f"-l{name}", "-o", str(executable)],
        check=True, env=clean_env(),
    )
    print(f"Built {native.resolve()}")
    print(f"Built {executable.resolve()}")


INSTRUMENTED_DEPENDENCIES = ("numpy", "numba")


def _installer() -> list[str]:
    """pip for this interpreter; uv-created environments have no pip module."""
    import importlib.util

    if importlib.util.find_spec("pip") is not None:
        return [sys.executable, "-m", "pip", "install", "--no-cache-dir"]
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("build-deps needs pip or uv for this environment")
    return [uv, "pip", "install", "--no-cache", "--python", sys.executable]


def _wheel_builder() -> list[str]:
    """``pip wheel`` for this interpreter (pip is added to uv-created environments)."""
    import importlib.util

    if importlib.util.find_spec("pip") is None:
        subprocess.run([*_installer(), "pip"], check=True)
    return [sys.executable, "-m", "pip", "wheel", "--no-cache-dir", "--no-deps"]


# Source-build options per dependency.
_BUILD_OPTIONS = {
    "numpy": ["-Csetup-args=-Db_sanitize=address",
              # No external BLAS: NumPy's own (instrumented) fallback loops are used.
              "-Csetup-args=-Dallow-noblas=true",
              "-Csetup-args=-Dblas=none", "-Csetup-args=-Dlapack=none"],
    "numba": [],
}


# Peak memory of one ASan + debug-info compile of NumPy's larger C files.
_MEMORY_PER_BUILD_JOB = 1 << 30


def _memory_budget() -> int | None:
    """Bytes this build may use: MemAvailable, capped by a cgroup v2 memory.max."""
    budget = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                budget = int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        relative = Path("/proc/self/cgroup").read_text().splitlines()[0].split("::", 1)[1]
        group = Path("/sys/fs/cgroup") / relative.lstrip("/")
        limit = (group / "memory.max").read_text().strip()
        if limit.isdigit():
            used = int((group / "memory.current").read_text().strip())
            room = max(0, int(limit) - used)
            budget = room if budget is None else min(budget, room)
    except (OSError, ValueError, IndexError):
        pass
    return budget


def build_jobs() -> int:
    """Parallel compile jobs for build-deps: ``NUMBA_ASAN_BUILD_JOBS``, else the
    CPU count limited to one job per GiB of available memory (sanitized
    debug builds of NumPy otherwise exhaust memory-capped jobs)."""
    configured = os.environ.get("NUMBA_ASAN_BUILD_JOBS")
    if configured:
        return max(1, int(configured))
    jobs = os.cpu_count() or 1
    budget = _memory_budget()
    if budget is not None:
        jobs = min(jobs, max(1, budget // _MEMORY_PER_BUILD_JOB))
    return jobs


def _build_options(name: str, jobs: int) -> list[str]:
    options = list(_BUILD_OPTIONS[name])
    if name == "numpy":
        options.append(f"-Ccompile-args=-j{jobs}")  # meson-python's ninja parallelism
    return options


def _find_wheel(directory: Path, name: str, version: str) -> Path | None:
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    matches = sorted(directory.glob(f"{name}-{version}-{tag}-*.whl"))
    return matches[0] if matches else None


def build_dependencies(fuzzer: bool = False, wheelhouse: Path | None = None) -> None:
    """Rebuild the installed NumPy and Numba releases with ASan in this environment.

    With *fuzzer*, also add libFuzzer coverage (``-fsanitize=fuzzer-no-link``),
    as Atheris requires for native extensions. With *wheelhouse*, the
    instrumented wheels are cached there per LLVM version and sanitizer flavor
    and reused by later environments with the same versions and Python.
    """
    from importlib import metadata

    compiler = _tool("clang")
    compiler_cxx = _tool("clang++")
    versions = {name: metadata.version(name) for name in INSTRUMENTED_DEPENDENCIES}
    sanitizers = "address,fuzzer-no-link" if fuzzer else "address"
    flags = f"-fsanitize={sanitizers} -fno-omit-frame-pointer -g"
    env = os.environ.copy()
    env.update(
        CC=compiler, CXX=compiler_cxx, CFLAGS=flags, CXXFLAGS=flags,
        LDFLAGS=f"-fsanitize={sanitizers}",
        # Optional threading backends need separately sanitized OpenMP/TBB.
        NUMBA_DISABLE_OPENMP="1", NUMBA_DISABLE_TBB="1",
    )
    # Build tools and Python code run during the build; never sanitize the pip process.
    env.pop("LD_PRELOAD", None)
    jobs = build_jobs()
    env.setdefault("MAX_JOBS", str(jobs))
    env.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", str(jobs))
    print(f"Building with {jobs} parallel compile job(s)")
    flavor = f"llvm{llvm_major()}-{sanitizers.replace(',', '-')}"
    with tempfile.TemporaryDirectory(prefix="numba-asan-wheels-") as scratch:
        cache = Path(wheelhouse).resolve() / flavor if wheelhouse else Path(scratch)
        cache.mkdir(parents=True, exist_ok=True)
        wheels = []
        for name in INSTRUMENTED_DEPENDENCIES:
            wheel = _find_wheel(cache, name, versions[name])
            if wheel is not None:
                print(f"Using cached instrumented wheel {wheel}")
            else:
                staging = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=cache))
                try:
                    subprocess.run(
                        [*_wheel_builder(), "--no-binary", name, "-w", str(staging),
                         f"{name}=={versions[name]}", *_build_options(name, jobs)],
                        env=env, check=True,
                    )
                    built = _find_wheel(staging, name, versions[name])
                    if built is None:
                        raise RuntimeError(f"Building {name}=={versions[name]} produced no wheel")
                    wheel = cache / built.name
                    # Atomic within the cache: concurrent builders never see partial wheels.
                    os.replace(built, wheel)
                finally:
                    shutil.rmtree(staging, ignore_errors=True)
            wheels.append(wheel)
        subprocess.run([*_installer(), "--force-reinstall", "--no-deps", *map(str, wheels)],
                       env=env, check=True)
    checks = {"ASan": "__asan_init"}
    if fuzzer:
        checks["fuzzer coverage"] = "__sanitizer_cov_8bit_counters_init"
    for label, symbol in checks.items():
        missing = [name for name, ok in instrumented_dependencies(symbol).items() if not ok]
        if missing:
            raise RuntimeError(f"Rebuilt dependencies lack {label} instrumentation: {missing}")
    print(f"NumPy and Numba native extensions are instrumented ({', '.join(checks)})")


def instrumented_dependencies(symbol: str = "__asan_init") -> dict[str, bool]:
    """Report whether every native extension of each dependency imports *symbol*.

    ``__asan_init`` indicates ASan; ``__sanitizer_cov_8bit_counters_init``
    indicates libFuzzer coverage (``-fsanitize=fuzzer-no-link``).
    """
    import importlib.util

    nm = _tool("llvm-nm")
    result = {}
    for name in INSTRUMENTED_DEPENDENCIES:
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            result[name] = False
            continue
        root = Path(spec.origin).parent
        extensions = list(root.rglob("*.so"))
        result[name] = bool(extensions) and all(
            symbol in subprocess.run(
                [nm, "-D", "--undefined-only", str(path)],
                capture_output=True, text=True, check=False, env=clean_env(),
                stdin=subprocess.DEVNULL, timeout=120,
            ).stdout
            for path in extensions
        )
    return result


def _runtime(sanitizer: str = "address") -> Path:
    compiler = _tool("clang")
    name = {"address": "asan", "thread": "tsan"}[sanitizer]
    path = Path(subprocess.check_output(
        [compiler, f"-print-file-name=libclang_rt.{name}-x86_64.so"], text=True,
        env=clean_env(),
    ).strip())
    if not path.is_file():
        major = llvm_major()
        raise RuntimeError(f"Missing LLVM {major} {name} runtime; install libclang-rt-{major}-dev")
    return path.resolve()


_OPTIONS = {"address": "ASAN_OPTIONS", "thread": "TSAN_OPTIONS",
            "memory": "MSAN_OPTIONS"}


def _prepend(env: dict, name: str, options: str) -> None:
    # Later options win: user-provided values override these defaults.
    env[name] = options + (":" + env[name] if env.get(name) else "")


def _fuzzer_runtime(plugin: Path) -> Path:
    """The matching LLVM ASan+libFuzzer runtime for Atheris, built by numba-asan build."""
    result = subprocess.run(
        [sys.executable, "-c", "import atheris"], capture_output=True, check=False,
        env=clean_env(), stdin=subprocess.DEVNULL, timeout=120,
    )
    if result.returncode:
        raise RuntimeError("--fuzz requires Atheris in this environment: pip install atheris")
    path = plugin.with_name(FUZZER_RUNTIME)
    if not path.is_file():
        raise RuntimeError(f"Missing {path}; rerun numba-asan build")
    return path


REPORT_LIMIT = 4 * 1024 * 1024


def _read_report(path: Path) -> str:
    """Bounded read: a crashing runtime can write reports in an endless loop."""
    with path.open("rb") as report:
        data = report.read(REPORT_LIMIT + 1)
    text = data[:REPORT_LIMIT].decode(errors="replace")
    if len(data) > REPORT_LIMIT:
        text += f"\n[numba-asan: report truncated at {REPORT_LIMIT} bytes: {path.name}]\n"
    return text


def launch(plugin: Path, target: list[str], *, sanitizer: str = "address",
           undefined: bool = False, leaks: bool = False, fuzz: bool = False) -> int:
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise RuntimeError("This prototype currently supports Linux x86-64 only")
    # Before running any tool: a foreign preloaded runtime is inherited by it.
    if re.search(r"lib(clang_rt\.)?[atm]san|(asan|ubsan|ubsan_cxx)_with_fuzzer",
                 os.environ.get("LD_PRELOAD", "")):
        raise RuntimeError(
            "Remove the sanitizer from LD_PRELOAD: numba-asan loads the matching "
            "runtime itself (for Atheris, use --fuzz instead of preloading "
            "Atheris's asan_with_fuzzer.so)")
    if not plugin.is_file():
        raise RuntimeError(f"Sanitizer plugin {plugin} is missing; run numba-asan build "
                           "first (or pass --plugin)")
    if leaks and sanitizer != "address":
        raise RuntimeError("--leaks requires the address sanitizer (LSan)")
    if undefined and sanitizer == "memory":
        raise RuntimeError("--undefined is supported with address or thread")
    if fuzz and (sanitizer != "address" or leaks):
        raise RuntimeError("--fuzz uses Atheris's ASan runtime; use libFuzzer's "
                           "-detect_leaks instead of --leaks")
    opt = _tool("opt")
    if sanitizer != "address" and any(instrumented_dependencies().values()):
        raise RuntimeError(
            f"NumPy/Numba in this environment were rebuilt with ASan by build-deps; "
            f"the {sanitizer} sanitizer cannot load ASan-instrumented modules. "
            "Use a separate environment with stock wheels for TSan/MSan."
        )
    env = os.environ.copy()
    # llvm-symbolizer would otherwise fetch debug info for system libraries
    # from debuginfod servers: network access that can stall reports.
    for environment in (os.environ, env):
        environment.pop("DEBUGINFOD_URLS", None)
    executable = sys.executable
    if sanitizer == "memory":
        executable = str(plugin.with_name("numba-msan-python"))
        if not Path(executable).is_file():
            raise RuntimeError("MemorySanitizer launcher is missing; run numba-asan build")
        # The launcher embeds libpython; reuse this interpreter's import paths.
        env["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)
        # Memory written by uninstrumented CPython/NumPy has no MSan shadow
        # updates, so only allocations the bridge marks are uninitialized.
        _prepend(env, "MSAN_OPTIONS", "poison_in_malloc=0:poison_in_free=0")
    else:
        if sanitizer == "address" and not plugin.with_name("numba_asan_native.so").is_file():
            raise RuntimeError("ASan native bridge is missing; run numba-asan build again")
        runtime = _fuzzer_runtime(plugin) if fuzz else _runtime(sanitizer)
        env["LD_PRELOAD"] = str(runtime) + (
            ":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else ""
        )
    env["PYTHONMALLOC"] = "malloc"
    if sanitizer == "address":
        # Exit-time checks report CPython's retained memory; --leaks checks
        # reachability while the interpreter is alive instead.
        _prepend(env, "ASAN_OPTIONS",
                 f"detect_leaks={int(leaks)}:leak_check_at_exit=0")
        if leaks:
            suppressions = Path(__file__).with_name("lsan.supp")
            _prepend(env, "LSAN_OPTIONS",
                     f"suppressions={suppressions}:print_suppressions=0")
    if sanitizer == "thread":
        # A race is reported only if TSan can restore the other access's stack
        # from its per-thread trace. Tight JIT loops fill the default trace
        # after ~100k instrumented accesses and such races are dropped silently;
        # the largest history roughly doubles that window.
        _prepend(env, "TSAN_OPTIONS", "history_size=7")
    if undefined:
        _prepend(env, "UBSAN_OPTIONS", "print_stacktrace=1:halt_on_error=1")

    owned = tempfile.TemporaryDirectory(prefix="numba-sanitizer-")
    retained = env.get("NUMBA_ASAN_JIT_DIR")
    if retained:
        Path(retained).mkdir(parents=True, exist_ok=True)
    # Per-launch directory: frame maps from other runs may reuse addresses.
    jit_directory = (Path(tempfile.mkdtemp(prefix="run-", dir=retained)) if retained
                     else Path(owned.name) / "jit").resolve()
    jit_directory.mkdir(parents=True, exist_ok=True)
    env["NUMBA_SANITIZER_JIT_DIR"] = str(jit_directory)
    options = _OPTIONS[sanitizer]
    symbolizer = which_tool("llvm-symbolizer")
    if symbolizer:
        # Some distributions' runtimes look only for a fixed /usr/bin path;
        # use the symbolizer matching this LLVM toolchain unless one is set.
        for name in dict.fromkeys((options, "UBSAN_OPTIONS")):
            if "external_symbolizer_path=" not in env.get(name, ""):
                _prepend(env, name, f"external_symbolizer_path={symbolizer}")
    capture = not any("log_path=" in env.get(name, "") for name in
                      (options, "UBSAN_OPTIONS", "LSAN_OPTIONS"))
    reports = Path(owned.name) / "report"
    if capture:
        _prepend(env, options, f"log_path={reports}")
    env["NUMBA_ASAN_OPT"] = opt
    env["NUMBA_ASAN_PLUGIN"] = str(plugin.resolve())
    env["NUMBA_SANITIZER"] = sanitizer
    env["NUMBA_SANITIZER_UNDEFINED"] = str(int(undefined))
    env["NUMBA_SANITIZER_LEAKS"] = str(int(leaks))
    env["NUMBA_SANITIZER_COVERAGE"] = str(int(fuzz))
    child = subprocess.Popen(
        [executable, "-m", "numba_jit_asan._bootstrap", *target], env=env,
    )
    forwarded = {}
    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        forwarded[number] = signal.signal(
            number, lambda received, frame: child.send_signal(received),
        )
    try:
        code = child.wait()
    finally:
        for number, handler in forwarded.items():
            signal.signal(number, handler)
    try:
        if capture:
            for path in sorted(reports.parent.glob(f"{reports.name}.*")):
                # log_path appends the reporting process's pid (fork/-jobs
                # children have their own JIT address ranges).
                pid = path.suffix[1:]
                frames = FrameMap(jit_directory, pid if pid.isdigit() else None)
                sys.stderr.write(rewrite(_read_report(path), frames))
            sys.stderr.flush()
    finally:
        owned.cleanup()
    return 128 - code if code < 0 else code


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run real LLVM AddressSanitizer on Numba CPU JIT code (plus TSan, MSan, UBSan and LSan modes)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build the sanitizer pass plugin with llvmlite's LLVM version")
    build.add_argument("--output", type=Path, default=DEFAULT_PLUGIN)
    deps = commands.add_parser(
        "build-deps",
        help="rebuild NumPy and Numba from source with ASan in this environment",
    )
    deps.add_argument("--fuzzer", action="store_true",
                      help="also add libFuzzer coverage (-fsanitize=fuzzer-no-link) for Atheris")
    deps.add_argument("--wheelhouse", type=Path,
                      help="cache the instrumented wheels here and reuse them when present")
    run = commands.add_parser("run", help="run a Python script or -m module under a sanitizer")
    run.add_argument("--plugin", type=Path, default=DEFAULT_PLUGIN)
    run.add_argument("--sanitizer", choices=("address", "thread", "memory"),
                     default="address", help="LLVM sanitizer for JIT code")
    run.add_argument("--undefined", action="store_true",
                     help="add UBSan checks for shift and float-to-int overflows")
    run.add_argument("--leaks", action="store_true",
                     help="enable LeakSanitizer checks while Python is alive")
    run.add_argument("--fuzz", action="store_true",
                     help="fuzz with Atheris: its ASan+libFuzzer runtime and JIT coverage")
    run.add_argument("target", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        if args.command == "build":
            build_plugin(args.output)
        elif args.command == "build-deps":
            build_dependencies(fuzzer=args.fuzzer, wheelhouse=args.wheelhouse)
        else:
            target = args.target
            if target and target[0] == "--":
                target = target[1:]
            if not target or (target[0] == "-m" and len(target) < 2):
                parser.error("run requires a script path or -- -m module")
            sys.exit(launch(args.plugin, target, sanitizer=args.sanitizer,
                            undefined=args.undefined, leaks=args.leaks,
                            fuzz=args.fuzz))
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"numba-asan: {exc}\n")


if __name__ == "__main__":
    main()
