"""Run specification and run_id.

A RunSpec is one training run (the unit of analysis, D-002). run_id is a short sha256 over the canonical
JSON of the spec plus the hashes of the configs that change what the run computes (protocol, task, and
backend parameters). The grid file is NOT hashed in: adding a condition to a grid must not invalidate
existing runs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

CONDITION_KINDS = ("full", "random", "cleaner", "oracle", "weighting", "weight_placebo")
NAMED_KINDS = ("cleaner", "weighting", "weight_placebo")  # carry the entrant's name + version
INDEXED_KINDS = ("random", "weight_placebo")  # carry subset_index (random subset / weight permutation index)
FULL_DATA_KINDS = ("full", "weighting", "weight_placebo")  # train on all episodes (budget 1.0)


@dataclass(frozen=True)
class Condition:
    kind: str
    name: str | None = None
    version: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in CONDITION_KINDS:
            raise ValueError(f"unknown condition kind {self.kind!r}")
        if self.kind in NAMED_KINDS and not (self.name and self.version):
            raise ValueError(f"{self.kind} condition needs name and version")
        if self.kind not in NAMED_KINDS and (self.name or self.version):
            raise ValueError(f"{self.kind} condition takes no name/version")

    @property
    def label(self) -> str:
        """Grouping label used by the report and fake-effect lookup (version-free)."""
        return f"{self.kind}:{self.name}" if self.kind in NAMED_KINDS else self.kind

    @classmethod
    def full(cls) -> Condition:
        return cls("full")

    @classmethod
    def random(cls) -> Condition:
        return cls("random")

    @classmethod
    def oracle(cls) -> Condition:
        return cls("oracle")

    @classmethod
    def cleaner(cls, name: str, version: str) -> Condition:
        return cls("cleaner", name, str(version))

    @classmethod
    def weighting(cls, name: str, version: str) -> Condition:
        return cls("weighting", name, str(version))

    @classmethod
    def weight_placebo(cls, name: str, version: str) -> Condition:
        """D-020: the entrant's own weight vector, randomly permuted across episodes."""
        return cls("weight_placebo", name, str(version))


@dataclass(frozen=True)
class RunSpec:
    task: str
    condition: Condition
    budget: float  # fraction of episodes kept; 1.0 for full
    subset_index: int | None
    train_seed: int

    def __post_init__(self) -> None:
        if (self.condition.kind in INDEXED_KINDS) != (self.subset_index is not None):
            raise ValueError("subset_index is set iff the condition is random or weight_placebo")
        if self.condition.kind in FULL_DATA_KINDS and self.budget != 1.0:
            raise ValueError(f"{self.condition.kind} condition has budget 1.0")
        if not 0 < self.budget <= 1:
            raise ValueError("budget must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["condition"] = {k: v for k, v in d["condition"].items() if v is not None}
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunSpec:
        return cls(
            task=d["task"],
            condition=Condition(**d["condition"]),
            budget=float(d["budget"]),
            subset_index=d["subset_index"],
            train_seed=int(d["train_seed"]),
        )

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def run_id(spec: RunSpec, config_hashes: dict[str, str], length: int = 16) -> str:
    payload = json.dumps(
        {"spec": json.loads(spec.canonical_json()), "configs": dict(sorted(config_hashes.items()))},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:length]
