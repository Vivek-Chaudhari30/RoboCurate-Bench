"""Idempotent, restart-safe run executor with pluggable backends.

- A spec whose valid results JSON exists is skipped (idempotent; a third invocation is a no-op).
- Each run holds a lock file `<results_dir>/.locks/<run_id>.lock` (O_CREAT|O_EXCL) recording host and pid.
  A lock whose pid is dead on this host is stale (e.g. after SIGKILL) and is reclaimed on restart.
- Results are written atomically (results.write_atomic), so a kill never leaves a half-written JSON.
- `gpu_check` runs before every run. The fake backend's check is a no-op.
"""

from __future__ import annotations

import json
import os
import socket
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from robocurate import results
from robocurate.runs.spec import RunSpec, run_id


class Backend(Protocol):
    name: str
    kind: str  # "fake" | "calibration" | "result"

    def config_hashes(self) -> dict[str, str]: ...

    def gpu_check(self) -> None: ...

    def run(self, spec: RunSpec, rid: str) -> dict[str, Any]: ...


@dataclass
class ExecReport:
    done: list[str] = field(default_factory=list)
    skipped_existing: list[str] = field(default_factory=list)
    skipped_locked: list[str] = field(default_factory=list)
    reclaimed_stale: list[str] = field(default_factory=list)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class RunLock:
    def __init__(self, lock_dir: Path, rid: str):
        self.path = lock_dir / f"{rid}.lock"
        self.held = False

    def _is_stale(self) -> bool:
        try:
            info = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return True  # unreadable/partial lock -> treat as stale
        if info.get("host") != socket.gethostname():
            return False  # cannot judge a remote pid; leave it
        return not _pid_alive(int(info.get("pid", -1)))

    def acquire(self) -> str:
        """Returns 'acquired', 'reclaimed' or 'locked'."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        status = "acquired"
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                if self._is_stale():
                    self.path.unlink(missing_ok=True)
                    status = "reclaimed"
                    continue
                return "locked"
            with os.fdopen(fd, "w") as f:
                json.dump({"host": socket.gethostname(), "pid": os.getpid()}, f)
            self.held = True
            return status
        return "locked"

    def release(self) -> None:
        if self.held:
            self.path.unlink(missing_ok=True)
            self.held = False


def result_path(results_dir: Path, rid: str) -> Path:
    return results_dir / f"{rid}.json"


def execute(
    specs: list[RunSpec],
    backend: Backend,
    results_dir: str | Path,
    *,
    log: Callable[[str], None] = print,
) -> ExecReport:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    lock_dir = results_dir / ".locks"
    hashes = backend.config_hashes()
    rep = ExecReport()
    ids = [run_id(s, hashes) for s in specs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate run_ids in plan (identical specs)")
    for spec, rid in zip(specs, ids, strict=True):
        out = result_path(results_dir, rid)
        if results.is_valid_file(out):
            rep.skipped_existing.append(rid)
            continue
        lock = RunLock(lock_dir, rid)
        status = lock.acquire()
        if status == "locked":
            rep.skipped_locked.append(rid)
            log(f"[locked]  {rid}  {spec.condition.label} seed={spec.train_seed}")
            continue
        if status == "reclaimed":
            rep.reclaimed_stale.append(rid)
        try:
            if results.is_valid_file(out):  # finished by another worker between check and lock
                rep.skipped_existing.append(rid)
                continue
            backend.gpu_check()
            record = backend.run(spec, rid)
            if record["run_id"] != rid or record["kind"] != backend.kind:
                raise RuntimeError(f"backend returned run_id/kind mismatch for {rid}")
            results.write_atomic(out, record)
            rep.done.append(rid)
            log(f"[done]    {rid}  {spec.condition.label} b={spec.budget} seed={spec.train_seed}")
        finally:
            lock.release()
    return rep
