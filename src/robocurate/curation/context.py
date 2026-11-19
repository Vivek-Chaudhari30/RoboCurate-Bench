"""What a cleaner may receive during the prepare stage (D-022), and nothing else.

CurationContext fields:
  backend           "fake" (tests / conformance) or "local" (real GPU host)
  rollouts          RolloutService (only if the cleaner declares requires_rollouts > 0)
  reference_policy  ReferencePolicy (only if requires_reference_policy): read-only full-data checkpoint
  progress_model    ProgressModel provider (Robometer on GPU hosts; a deterministic mock in fake/test mode)
  dataset_root      read-only path of the LeRobot dataset the view was built from (checker adapters only; the
                    labels sidecar lives BESIDE this directory, never inside it, D-008)
  device            torch device string
"""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ReferencePolicyModified(RuntimeError):
    pass


def tree_sha256(path: str | Path) -> str:
    """Content hash of a file or directory tree (relative paths + bytes)."""
    path = Path(path)
    h = hashlib.sha256()
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    for f in files:
        h.update(str(f.relative_to(path) if path.is_dir() else f.name).encode() + b"\0")
        h.update(f.read_bytes())
    return h.hexdigest()


@dataclass
class ReferencePolicy:
    """Read-only access to the full-data checkpoint of the first train seed (D-022).

    `path` is a directory; files are chmod'ed read-only for the duration of the prepare cell, and the tree hash
    is re-checked afterwards (`verify_unchanged`). `loader(path)` returns a policy object (LeRobot
    PreTrainedPolicy on GPU hosts, MLPPolicy in tests) exposing loss(batch) / parameters().
    """

    path: Path
    loader: Callable[[Path], Any]
    sha256: str = ""
    _modes: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(f"reference policy checkpoint not found: {self.path}")
        self.sha256 = tree_sha256(self.path)

    def lock(self) -> None:
        for f in [self.path, *self.path.rglob("*")]:
            m = f.stat().st_mode
            self._modes[f] = m
            if f.is_file():
                os.chmod(f, m & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))

    def unlock(self) -> None:
        for f, m in self._modes.items():
            if f.exists():
                os.chmod(f, m)
        self._modes.clear()

    def load(self) -> Any:
        return self.loader(self.path)

    def verify_unchanged(self) -> None:
        if tree_sha256(self.path) != self.sha256:
            raise ReferencePolicyModified(f"reference policy at {self.path} was modified during curation")


@dataclass
class CurationContext:
    backend: str = "fake"
    rollouts: Any = None
    reference_policy: ReferencePolicy | None = None
    progress_model: Any = None
    dataset_root: Path | None = None
    device: str = "cpu"
    extra: dict[str, Any] = field(default_factory=dict)
