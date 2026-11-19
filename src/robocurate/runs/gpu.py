"""GPU etiquette helpers (CLAUDE.md): the shared-GPU busy check and a peak-memory poller.

Busy rule: refuse to start if any process NOT owned by the current user uses > 2 GB, or if GPU utilization
is > 10% while any foreign process is present. Parsing is separated from querying so it is CPU-testable.
"""

from __future__ import annotations

import getpass
import shutil
import subprocess
import threading
from dataclasses import dataclass

FOREIGN_MEM_MB = 2048
FOREIGN_UTIL_PCT = 10


class GPUBusy(RuntimeError):
    pass


class GPUUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class GPUProc:
    pid: int
    used_mb: float
    user: str


def parse_compute_apps(text: str) -> list[tuple[int, float]]:
    """Parse `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits`."""
    out = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        pid, mem = (x.strip() for x in line.split(",")[:2])
        out.append((int(pid), float(mem) if mem not in ("[N/A]", "") else 0.0))
    return out


def parse_utilization(text: str) -> list[float]:
    """Parse `nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits` (one line per GPU)."""
    return [float(x.strip()) for x in text.strip().splitlines() if x.strip()]


def decide_busy(procs: list[GPUProc], util: list[float], me: str) -> str | None:
    """Reason string if the GPU is busy by the CLAUDE.md rule, else None."""
    foreign = [p for p in procs if p.user != me]
    for p in foreign:
        if p.used_mb > FOREIGN_MEM_MB:
            return f"pid {p.pid} ({p.user}) uses {p.used_mb:.0f} MB > {FOREIGN_MEM_MB} MB"
    if foreign and any(u > FOREIGN_UTIL_PCT for u in util):
        return f"GPU utilization {max(util):.0f}% > {FOREIGN_UTIL_PCT}% with foreign processes present"
    return None


def _smi(*args: str) -> str:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        raise GPUUnavailable("nvidia-smi not found")
    return subprocess.run([exe, *args], capture_output=True, text=True, check=True, timeout=30).stdout


def _owner(pid: int) -> str:
    r = subprocess.run(["ps", "-o", "user=", "-p", str(pid)], capture_output=True, text=True)
    return r.stdout.strip() or "?"


def query() -> tuple[list[GPUProc], list[float]]:
    apps = parse_compute_apps(_smi("--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"))
    util = parse_utilization(_smi("--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"))
    return [GPUProc(pid, mb, _owner(pid)) for pid, mb in apps], util


def check_free() -> None:
    procs, util = query()
    reason = decide_busy(procs, util, getpass.getuser())
    if reason:
        raise GPUBusy(reason)


class PeakMemPoller:
    """Polls nvidia-smi for one pid's used memory; records the max (MB). Use as a context manager."""

    def __init__(self, pid: int, interval_s: float = 5.0):
        self.pid, self.interval_s = pid, interval_s
        self.peak_mb: float | None = None
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                for pid, mb in parse_compute_apps(
                    _smi("--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits")
                ):
                    if pid == self.pid:
                        self.peak_mb = mb if self.peak_mb is None else max(self.peak_mb, mb)
            except (GPUUnavailable, subprocess.SubprocessError):
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self) -> PeakMemPoller:
        self._t.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._t.join(timeout=self.interval_s + 5)
