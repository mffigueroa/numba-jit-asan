"""Rewrite sanitizer reports so JIT frames show Numba functions and source lines."""
import bisect
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess

from . import toolchain

_FRAME = re.compile(r"^(?P<indent>\s*)#(?P<index>\d+) (?P<body>.*)$")
_ADDRESS = re.compile(r"0x[0-9a-fA-F]+")
_ABI_TAG = re.compile(r"\[abi:[^\]]*\]")
_UNKNOWN = re.compile(r"\(<unknown module>\)")


@dataclass(frozen=True)
class Frame:
    start: int
    size: int
    offset: int
    name: str
    object: str


class FrameMap:
    """JIT address ranges; *pid* restricts them to one process's frames."""

    def __init__(self, directory: Path, pid=None):
        frames = []
        pattern = f"frames.{pid}.tsv" if pid is not None else "frames.*.tsv"
        if not any(directory.glob(pattern)):
            # e.g. os.fork() children: JIT code was compiled by an ancestor.
            pattern = "frames.*.tsv"
        for path in sorted(directory.glob(pattern)):
            for line in path.read_text().splitlines():
                start, size, offset, name, obj = line.split("\t")
                frames.append(Frame(int(start, 16), int(size, 16), int(offset, 16), name, obj))
        frames.sort(key=lambda frame: frame.start)
        self.frames = frames
        self.starts = [frame.start for frame in frames]
        self.symbolizer = toolchain.which("llvm-symbolizer")
        self.demangler = toolchain.which("llvm-cxxfilt")
        self._cache = {}

    def find(self, pc: int):
        index = bisect.bisect_right(self.starts, pc) - 1
        if index >= 0:
            frame = self.frames[index]
            if pc < frame.start + frame.size:
                return frame
        return None

    def _name(self, mangled: str) -> str:
        if not self.demangler:
            return mangled
        result = subprocess.run([self.demangler, mangled], capture_output=True,
                                text=True, check=False)
        name = result.stdout.strip() or mangled
        return _ABI_TAG.sub("", name)

    def describe(self, pc: int, return_address: bool):
        frame = self.find(pc)
        if frame is None:
            return None
        # Frames above the first hold return addresses; attribute to the call.
        offset = frame.offset + pc - frame.start - (1 if return_address else 0)
        key = (frame.object, offset, frame.name)
        if key not in self._cache:
            location = "??"
            if self.symbolizer:
                result = subprocess.run(
                    [self.symbolizer, "--obj", frame.object, hex(offset)],
                    capture_output=True, text=True, check=False,
                )
                lines = [line for line in result.stdout.splitlines() if line]
                if len(lines) >= 2 and not lines[1].startswith("??"):
                    location = lines[1]
            self._cache[key] = (self._name(frame.name), location, offset)
        return self._cache[key]


def rewrite(report: str, frames: FrameMap) -> str:
    output = []
    first_location = None
    for line in report.splitlines():
        match = _FRAME.match(line)
        if match:
            for token in _ADDRESS.findall(match.group("body")):
                description = frames.describe(int(token, 16), int(match.group("index")) > 0)
                if description:
                    name, location, offset = description
                    line = (f"{match.group('indent')}#{match.group('index')} {token} "
                            f"in {name} {location} (numba-jit+{offset:#x})")
                    if first_location is None:
                        first_location = f"{location} in {name}"
                    break
        elif line.startswith("SUMMARY:") and first_location:
            line = _UNKNOWN.sub(first_location, line)
        output.append(line)
    return "\n".join(output) + ("\n" if report.endswith("\n") else "")
