"""Typed loaders for protocol, task and grid YAMLs.

The literal string "TBD" in any YAML becomes the `TBD` sentinel. Values that depend on GPU
measurements (steps, seed count, per-run cost) stay TBD until calibration (D-005, D-014).

Mode semantics:
- "result": reading a TBD field raises `TBDError` (result runs are blocked until calibrated).
- "calibration" / "fake" / "plan": reading a TBD field returns the `TBD` sentinel, which callers must
  handle explicitly (e.g. the planner prints "TBD").
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

Mode = Literal["result", "calibration", "fake", "plan"]
MODES: tuple[str, ...] = ("result", "calibration", "fake", "plan")


class _TBDType:
    _instance: _TBDType | None = None

    def __new__(cls) -> _TBDType:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "TBD"

    def __str__(self) -> str:
        return "TBD"

    def __bool__(self) -> bool:  # TBD must never silently act as a truthy number
        raise TypeError("TBD has no truth value; check `is TBD` explicitly")

    def __reduce__(self):
        return (_TBDType, ())


TBD = _TBDType()


class TBDError(RuntimeError):
    """A TBD value was read in result mode."""


class ConfigError(ValueError):
    pass


def is_tbd(x: Any) -> bool:
    return x is TBD


def _convert_tbd(obj: Any) -> Any:
    if isinstance(obj, str) and obj.strip() == "TBD":
        return TBD
    if isinstance(obj, dict):
        return {k: _convert_tbd(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_tbd(v) for v in obj]
    return obj


def contains_tbd(obj: Any) -> bool:
    if obj is TBD:
        return True
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return any(contains_tbd(object.__getattribute__(obj, f.name)) for f in dataclasses.fields(obj))
    if isinstance(obj, dict):
        return any(contains_tbd(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(contains_tbd(v) for v in obj)
    return False


def tbd_fields(obj: Any, prefix: str = "") -> list[str]:
    """Dotted names of every TBD field (for error messages and the planner)."""
    out: list[str] = []
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            if f.name.startswith("_"):
                continue
            out += tbd_fields(object.__getattribute__(obj, f.name), f"{prefix}{f.name}.")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out += tbd_fields(v, f"{prefix}{k}.")
    elif obj is TBD:
        out.append(prefix.rstrip("."))
    return out


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return _convert_tbd(data)


@dataclass
class _Guarded:
    """Base: reading a TBD field raises in result mode."""

    _mode: str = field(default="plan", repr=False, compare=False, kw_only=True)
    _path: str | None = field(default=None, repr=False, compare=False, kw_only=True)
    _sha256: str | None = field(default=None, repr=False, compare=False, kw_only=True)

    def __getattribute__(self, name: str) -> Any:
        value = object.__getattribute__(self, name)
        if value is TBD and not name.startswith("_") and object.__getattribute__(self, "_mode") == "result":
            cls = type(self).__name__
            raise TBDError(f"{cls}.{name} is TBD; result runs are blocked until it is measured (D-014)")
        return value

    def raw(self, name: str) -> Any:
        """Field value without the TBD guard (for planners/reports that print "TBD")."""
        return object.__getattribute__(self, name)

    def with_mode(self, mode: Mode) -> Any:
        if mode not in MODES:
            raise ConfigError(f"unknown mode {mode!r}")
        kwargs = {f.name: self.raw(f.name) for f in dataclasses.fields(self) if f.init}
        kwargs["_mode"] = mode
        return type(self)(**kwargs)

    def __repr__(self) -> str:
        body = ", ".join(
            f"{f.name}={self.raw(f.name)!r}" for f in dataclasses.fields(self) if not f.name.startswith("_")
        )
        return f"{type(self).__name__}({body})"

    @property
    def path(self) -> str | None:
        return self._path

    @property
    def sha256(self) -> str | None:
        return self._sha256


def _build(cls: type, data: dict[str, Any], path: Path, mode: Mode) -> Any:
    names = {f.name for f in dataclasses.fields(cls) if not f.name.startswith("_")}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"{path}: unknown keys for {cls.__name__}: {sorted(unknown)}")
    try:
        obj = cls(**data, _mode=mode, _path=str(path), _sha256=sha256_file(path))
    except TypeError as e:
        raise ConfigError(f"{path}: {e}") from e
    return obj


# ---------------------------------------------------------------------------- protocol


@dataclass(repr=False, eq=False)
class ProtocolConfig(_Guarded):
    eval_seed_start: int
    n_eval_rollouts: int
    checkpoint_rule: str
    n_train_seeds: Any
    train_seeds: Any
    random_subset_seeds: list[int]
    curation_seed_range: list[int]
    alpha: float
    test: str
    n_random_subsets_min: int = 3

    def __post_init__(self) -> None:
        g = lambda n: object.__getattribute__(self, n)  # noqa: E731  raw access, no TBD guard
        if g("checkpoint_rule") != "final":
            raise ConfigError("checkpoint_rule must be 'final' (D-003)")
        if g("n_eval_rollouts") < 200:
            raise ConfigError("n_eval_rollouts must be >= 200 (CLAUDE.md rule 2)")
        if len(g("random_subset_seeds")) < g("n_random_subsets_min") or g("n_random_subsets_min") < 3:
            raise ConfigError("need >= 3 random subset seeds (CLAUDE.md rule 1)")
        lo, hi = g("curation_seed_range")
        if not lo <= hi:
            raise ConfigError("curation_seed_range must be [lo, hi] with lo <= hi")
        ts, nts = g("train_seeds"), g("n_train_seeds")
        if ts is not TBD:
            if nts is TBD or len(ts) != nts:
                raise ConfigError("train_seeds length must equal n_train_seeds")
            if nts < 3:
                raise ConfigError("n_train_seeds floor is 3 (D-005)")
        if g("test") != "welch_holm":
            raise ConfigError("test must be 'welch_holm' (D-002)")

    def eval_seed_range(self, batch_size: int | None = None) -> tuple[int, int]:
        """Half-open range of env seeds lerobot-eval can touch (see eval/lerobot_eval.py docstring).

        With batch_size b, the last batch may run up to b-1 extra (discarded) episodes.
        """
        n = self.n_eval_rollouts
        if batch_size:
            n = -(-n // batch_size) * batch_size
        return (self.eval_seed_start, self.eval_seed_start + n)

    def check_seed_disjointness(self, batch_size: int | None = None) -> None:
        e0, e1 = self.eval_seed_range(batch_size)
        c0, c1 = self.curation_seed_range
        if not (e1 <= c0 or c1 + 1 <= e0):
            raise ConfigError(
                f"eval seeds [{e0},{e1}) overlap curation_seed_range {self.curation_seed_range}"
            )


def load_protocol(path: str | Path, mode: Mode = "plan") -> ProtocolConfig:
    path = Path(path)
    cfg = _build(ProtocolConfig, load_yaml(path), path, mode)
    cfg.check_seed_disjointness()
    return cfg


# ---------------------------------------------------------------------------- task


@dataclass(repr=False, eq=False)
class TaskConfig(_Guarded):
    name: str
    env_type: str
    env_task: str
    policy_type: str
    source_datasets: list[str]
    train_dataset: str
    steps: Any
    per_run_gpu_h: Any
    batch_size: Any = 8
    eval_batch_size: Any = 50
    save_freq: Any = None
    notes: str = ""


def load_task(path: str | Path, mode: Mode = "plan") -> TaskConfig:
    path = Path(path)
    return _build(TaskConfig, load_yaml(path), path, mode)


# ---------------------------------------------------------------------------- grid


@dataclass
class ConditionSpec:
    kind: str  # full | random | cleaner | oracle
    name: str | None = None
    version: str | None = None
    n_subsets: int | None = None  # for random


@dataclass(repr=False, eq=False)
class GridConfig(_Guarded):
    name: str
    task: str  # path to task YAML (relative to repo root or absolute)
    protocol: str  # path to protocol YAML
    budgets: list[float]
    conditions: list[dict[str, Any]]

    def __post_init__(self) -> None:
        for b in object.__getattribute__(self, "budgets"):
            if not 0 < b <= 1:
                raise ConfigError(f"budget {b} must be a fraction in (0, 1]")
        self.condition_specs()  # validate eagerly

    def condition_specs(self) -> list[ConditionSpec]:
        out = []
        for c in object.__getattribute__(self, "conditions"):
            c = dict(c)
            kind = c.pop("kind", None)
            if kind not in ("full", "random", "cleaner", "oracle"):
                raise ConfigError(f"unknown condition kind {kind!r}")
            spec = ConditionSpec(kind=kind, **c)
            if kind == "cleaner" and not (spec.name and spec.version):
                raise ConfigError("cleaner condition needs name and version")
            if kind == "random" and (spec.n_subsets or 0) < 3:
                raise ConfigError("random condition needs n_subsets >= 3 (CLAUDE.md rule 1)")
            out.append(spec)
        kinds = [s.kind for s in out]
        if "cleaner" in kinds and not ("random" in kinds and "full" in kinds):
            raise ConfigError("cleaners must be compared to random subsets AND full data (rule 1)")
        return out


def load_grid(path: str | Path, mode: Mode = "plan") -> GridConfig:
    path = Path(path)
    return _build(GridConfig, load_yaml(path), path, mode)


def resolve_config_path(p: str | Path, base: Path | None = None) -> Path:
    """Resolve a config path relative to the repo root unless absolute."""
    p = Path(p)
    if p.is_absolute():
        return p
    if base is None:
        from robocurate.paths import repo_root

        base = repo_root()
    return base / p
