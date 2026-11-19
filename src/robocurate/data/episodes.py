"""DatasetView: read-only per-episode access for cleaners. Contains NO labels (D-008).

Per episode: actions, state, length, task string, timestamps. Arrays are read-only numpy views.
Images are intentionally not exposed in this milestone (no cleaner in scope needs them).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np


def _ro(a: np.ndarray) -> np.ndarray:
    a = np.array(a, copy=True)
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class EpisodeData:
    episode_index: int
    task: str
    actions: np.ndarray  # (T, action_dim)
    state: np.ndarray  # (T, state_dim)
    timestamps: np.ndarray  # (T,)

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])

    def content_hash(self) -> str:
        """sha256 over task, actions and state bytes. Independent of episode_index (D-021)."""
        h = hashlib.sha256()
        h.update(f"{self.task}|{self.actions.shape}|{self.state.shape}|".encode())
        h.update(np.ascontiguousarray(self.actions).tobytes())
        h.update(np.ascontiguousarray(self.state).tobytes())
        return h.hexdigest()


class DatasetView:
    """Immutable, label-free view over a dataset's episodes."""

    __slots__ = ("_episodes", "_fps", "_name", "_salt")

    def __init__(self, episodes: Sequence[EpisodeData], fps: int, name: str = "dataset", tie_break_salt: str = ""):
        idx = [e.episode_index for e in episodes]
        if len(set(idx)) != len(idx):
            raise ValueError("duplicate episode_index in view")
        object.__setattr__(self, "_episodes", {e.episode_index: e for e in episodes})
        object.__setattr__(self, "_fps", int(fps))
        object.__setattr__(self, "_name", str(name))
        object.__setattr__(self, "_salt", str(tie_break_salt))

    def __setattr__(self, k, v):  # read-only
        raise AttributeError("DatasetView is read-only")

    @classmethod
    def from_arrays(
        cls,
        actions: Sequence[np.ndarray],
        states: Sequence[np.ndarray],
        tasks: Sequence[str],
        fps: int,
        name: str = "dataset",
        episode_indices: Sequence[int] | None = None,
        tie_break_salt: str = "",
    ) -> DatasetView:
        n = len(actions)
        if not (len(states) == len(tasks) == n):
            raise ValueError("actions, states, tasks must have equal length")
        ids = list(range(n)) if episode_indices is None else list(episode_indices)
        eps = []
        for i, a, s, t in zip(ids, actions, states, tasks, strict=True):
            if len(a) != len(s):
                raise ValueError(f"episode {i}: actions and state lengths differ")
            eps.append(EpisodeData(int(i), str(t), _ro(a), _ro(s), _ro(np.arange(len(a)) / fps)))
        return cls(eps, fps, name, tie_break_salt)

    @property
    def name(self) -> str:
        return self._name

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def tie_break_salt(self) -> str:
        """protocol.tie_break_salt (D-021); "" only in tests/conformance views."""
        return self._salt

    def tie_break_key(self, i: int) -> str:
        """D-021: sha256(salt || episode content hash). Exact-duplicate episodes share a key; among those
        (identical content) the caller may fall back to episode_index."""
        return hashlib.sha256((self._salt + self._episodes[i].content_hash()).encode()).hexdigest()

    @property
    def n_episodes(self) -> int:
        return len(self._episodes)

    @property
    def episode_indices(self) -> list[int]:
        return sorted(self._episodes)

    def episode(self, i: int) -> EpisodeData:
        return self._episodes[i]

    def __iter__(self) -> Iterator[EpisodeData]:
        for i in self.episode_indices:
            yield self._episodes[i]

    def lengths(self) -> dict[int, int]:
        return {i: self._episodes[i].length for i in self.episode_indices}

    def fingerprint(self) -> str:
        """sha256 over episode ids, lengths, tasks and array bytes (identifies the data a cleaner saw)."""
        h = hashlib.sha256()
        for e in self:
            h.update(f"{e.episode_index}|{e.length}|{e.task}|".encode())
            h.update(np.ascontiguousarray(e.actions).tobytes())
            h.update(np.ascontiguousarray(e.state).tobytes())
        return h.hexdigest()


def synthetic_view(n_episodes: int = 40, seed: int = 0, fps: int = 50, tie_break_salt: str = "") -> DatasetView:
    """Small synthetic dataset for conformance checks and tests (no LeRobot)."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(200, 500, size=n_episodes)
    acts = [rng.normal(size=(int(L), 14)).astype(np.float32) for L in lengths]
    sts = [np.cumsum(a, axis=0) * 0.01 for a in acts]
    return DatasetView.from_arrays(acts, sts, ["Transfer the cube."] * n_episodes, fps, name="synthetic",
                                   tie_break_salt=tie_break_salt)
