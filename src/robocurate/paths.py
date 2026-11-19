"""Scratch-path resolution under the CLAUDE.md storage rule.

Rule: code refuses to run if ROBOCURATE_SCRATCH is unset or inside the repo. On hokkaido it must be
on the scratch disk, not under $HOME. On a Mac any directory outside the repo is fine
(the documented default is ~/robocurate-scratch, set in scripts/env.example.sh -- never in code).

HF_HOME and HF_LEROBOT_HOME must point inside the scratch root. LeRobot reads HF_LEROBOT_HOME once,
at import time (lerobot/utils/constants.py:69), so `require_storage_env()` must run before
`import lerobot` anywhere that touches datasets.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

ENV_SCRATCH = "ROBOCURATE_SCRATCH"
HOKKAIDO_PREFIX = "hokkaido"


class StorageError(RuntimeError):
    """Raised when the storage environment violates the CLAUDE.md storage rule."""


def repo_root() -> Path:
    """Repository root (the directory containing pyproject.toml above this file)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "robocurate").is_dir():
            return parent
    raise StorageError(f"cannot locate repo root above {here}")


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def is_hokkaido(hostname: str | None = None) -> bool:
    return (hostname or socket.gethostname()).lower().startswith(HOKKAIDO_PREFIX)


def scratch_root(
    env: dict[str, str] | None = None,
    *,
    hostname: str | None = None,
    repo: Path | None = None,
    home: Path | None = None,
) -> Path:
    """Return the validated scratch root. Never creates or defaults it."""
    env = dict(os.environ) if env is None else env
    raw = env.get(ENV_SCRATCH, "").strip()
    if not raw:
        raise StorageError(
            f"{ENV_SCRATCH} is unset. Source scripts/env.sh (see scripts/env.example.sh) first."
        )
    root = Path(raw).expanduser().resolve()
    repo = (repo or repo_root()).resolve()
    if _is_within(root, repo) or _is_within(repo, root):
        raise StorageError(f"{ENV_SCRATCH}={root} overlaps the repo ({repo}); it must be outside it.")
    if is_hokkaido(hostname):
        home = (home or Path.home()).resolve()
        if _is_within(root, home):
            raise StorageError(
                f"{ENV_SCRATCH}={root} is under $HOME ({home}); on hokkaido use the scratch disk."
            )
    return root


def require_storage_env(env: dict[str, str] | None = None, **kw) -> Path:
    """Validate ROBOCURATE_SCRATCH plus HF_HOME / HF_LEROBOT_HOME (must live under scratch)."""
    env = dict(os.environ) if env is None else env
    root = scratch_root(env, **kw)
    for var in ("HF_HOME", "HF_LEROBOT_HOME"):
        val = env.get(var, "").strip()
        if not val:
            raise StorageError(f"{var} is unset; expected {root}/{'hf' if var == 'HF_HOME' else 'lerobot'}")
        if not _is_within(Path(val).expanduser().resolve(), root):
            raise StorageError(f"{var}={val} is not inside {ENV_SCRATCH}={root}")
    return root


def scratch_dir(*parts: str, create: bool = True) -> Path:
    """A subdirectory of the validated scratch root."""
    p = scratch_root().joinpath(*parts)
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p
