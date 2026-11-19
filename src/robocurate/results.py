"""Results JSON schema, provenance capture, atomic write, load and validate.

One file per training run (unit of analysis = training run, D-002). CLAUDE.md rule 8 fields are
mandatory. `kind="result"` additionally requires a clean git tree and >= 200 eval rollouts.
"""

from __future__ import annotations

import json
import math
import os
import socket
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
KINDS = ("calibration", "result", "fake")


class ResultsValidationError(ValueError):
    pass


@dataclass
class GitInfo:
    hash: str
    dirty: bool


@dataclass
class ConfigRef:
    path: str
    sha256: str


@dataclass
class RunResult:
    run_id: str
    spec: dict[str, Any]
    kind: str
    git: GitInfo
    configs: list[ConfigRef]
    lerobot_version: str
    command_lines: list[list[str]]
    seeds: dict[str, Any]
    per_episode_success: list[bool]
    n_episodes: int
    success_rate: float
    wilson_95ci: list[float]
    checkpoint_step: int | None
    wall_clock_s: float
    peak_gpu_mem_mb: float | None
    hostname: str
    created_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    extra: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------- provenance


def git_info(repo: Path | None = None) -> GitInfo:
    from robocurate.paths import repo_root

    repo = repo or repo_root()

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip()

    try:
        h = run("rev-parse", "HEAD")
    except subprocess.CalledProcessError:
        h = "NO_COMMIT"
    dirty = bool(run("status", "--porcelain", "--untracked-files=no"))
    return GitInfo(hash=h, dirty=dirty)


def lerobot_version() -> str:
    """Installed LeRobot version from package metadata (does not import lerobot)."""
    try:
        return metadata.version("lerobot")
    except metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def hostname() -> str:
    return socket.gethostname()


# ---------------------------------------------------------------------------- validation

_REQUIRED = {
    "schema_version": int,
    "run_id": str,
    "spec": dict,
    "kind": str,
    "git": dict,
    "configs": list,
    "lerobot_version": str,
    "command_lines": list,
    "seeds": dict,
    "per_episode_success": list,
    "n_episodes": int,
    "success_rate": (int, float),
    "wilson_95ci": list,
    "wall_clock_s": (int, float),
    "hostname": str,
    "created_utc": str,
}


def validate(d: dict[str, Any]) -> None:
    """Raise ResultsValidationError if `d` is not a valid results record."""
    for key, typ in _REQUIRED.items():
        if key not in d:
            raise ResultsValidationError(f"missing field {key!r}")
        if not isinstance(d[key], typ) or (typ is int and isinstance(d[key], bool)):
            raise ResultsValidationError(f"field {key!r} has type {type(d[key]).__name__}")
    for key in ("checkpoint_step", "peak_gpu_mem_mb"):
        if key not in d:
            raise ResultsValidationError(f"missing field {key!r} (nullable, but must be present)")
    if d["schema_version"] != SCHEMA_VERSION:
        raise ResultsValidationError(f"schema_version {d['schema_version']} != {SCHEMA_VERSION}")
    if d["kind"] not in KINDS:
        raise ResultsValidationError(f"kind {d['kind']!r} not in {KINDS}")
    if set(d["git"]) != {"hash", "dirty"}:
        raise ResultsValidationError("git must have exactly hash and dirty")
    for c in d["configs"]:
        if set(c) != {"path", "sha256"} or len(c["sha256"]) != 64:
            raise ResultsValidationError(f"bad config ref {c!r}")
    pes = d["per_episode_success"]
    if not all(isinstance(x, bool) for x in pes):
        raise ResultsValidationError("per_episode_success must be a list of bools")
    if d["n_episodes"] != len(pes) or len(pes) == 0:
        raise ResultsValidationError("n_episodes must equal len(per_episode_success) and be > 0")
    rate = sum(pes) / len(pes)
    if not math.isclose(d["success_rate"], rate, abs_tol=1e-12):
        raise ResultsValidationError(f"success_rate {d['success_rate']} != {rate}")
    ci = d["wilson_95ci"]
    if len(ci) != 2 or not (0.0 <= ci[0] <= rate <= ci[1] <= 1.0):
        raise ResultsValidationError(f"wilson_95ci {ci} inconsistent with rate {rate}")
    _check_wilson(sum(pes), len(pes), ci)
    if d["kind"] == "result":
        if d["git"]["dirty"]:
            raise ResultsValidationError("result runs require a clean git tree (rule 8)")
        if d["n_episodes"] < 200:
            raise ResultsValidationError("result runs require >= 200 eval rollouts (rule 2)")
        if d["checkpoint_step"] is None:
            raise ResultsValidationError("result runs must record checkpoint_step (rule 3)")
    if "train_seed" not in d["seeds"]:
        raise ResultsValidationError("seeds must include train_seed")


def _check_wilson(k: int, n: int, ci: list[float]) -> None:
    from robocurate.report.stats import wilson_ci

    lo, hi = wilson_ci(k, n)
    if not (math.isclose(lo, ci[0], abs_tol=1e-9) and math.isclose(hi, ci[1], abs_tol=1e-9)):
        raise ResultsValidationError(f"wilson_95ci {ci} != recomputed {[lo, hi]}")


# ---------------------------------------------------------------------------- io


def write_atomic(path: str | Path, data: dict[str, Any]) -> Path:
    """Validate, then write JSON via temp file + rename (atomic on POSIX)."""
    validate(data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def load(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        d = json.load(f)
    validate(d)
    return d


def is_valid_file(path: str | Path) -> bool:
    try:
        load(path)
        return True
    except (OSError, json.JSONDecodeError, ResultsValidationError):
        return False


def load_dir(results_dir: str | Path) -> list[dict[str, Any]]:
    """Load every valid results JSON under a directory (non-recursive into hidden files)."""
    out = []
    for p in sorted(Path(results_dir).glob("*.json")):
        if p.name.startswith("."):
            continue
        out.append(load(p))
    return out
