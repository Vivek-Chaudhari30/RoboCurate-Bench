"""Expand a grid YAML into RunSpecs, with a TBD-aware cost estimate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from robocurate.config import (
    TBD,
    ConfigError,
    GridConfig,
    ProtocolConfig,
    TaskConfig,
    is_tbd,
    load_grid,
    load_protocol,
    load_task,
    resolve_config_path,
)
from robocurate.runs.spec import Condition, RunSpec


@dataclass(frozen=True)
class Unit:
    """A condition x budget x subset cell; multiplied by the train seeds to give runs."""

    condition: Condition
    budget: float
    subset_index: int | None
    subset_seed: int | None


@dataclass
class Plan:
    grid: GridConfig
    protocol: ProtocolConfig
    task: TaskConfig
    units: list[Unit]
    train_seeds: Any  # list[int] or TBD
    notes: list[str] = field(default_factory=list)
    steps_override: Any = None  # calibration configs carry their own step count
    kind: str = "grid"  # "grid" | "calibration"

    @property
    def steps(self) -> Any:
        return self.steps_override if self.kind == "calibration" else self.task.raw("steps")

    @property
    def n_runs(self) -> Any:
        return TBD if is_tbd(self.train_seeds) else len(self.units) * len(self.train_seeds)

    @property
    def gpu_hours(self) -> Any:
        per = self.task.raw("per_run_gpu_h")
        if is_tbd(per) or is_tbd(self.n_runs):
            return TBD
        return self.n_runs * float(per)

    def specs(self) -> list[RunSpec]:
        if is_tbd(self.train_seeds):
            raise ConfigError("train_seeds is TBD; cannot expand runs (supply measured seeds first)")
        return [
            RunSpec(self.task.name, u.condition, u.budget, u.subset_index, s)
            for u in self.units
            for s in self.train_seeds
        ]

    def subset_seed(self, spec: RunSpec) -> int | None:
        if spec.subset_index is None:
            return None
        return self.protocol.random_subset_seeds[spec.subset_index]

    def summary_lines(self) -> list[str]:
        seeds = self.train_seeds
        n_seeds = "TBD" if is_tbd(seeds) else str(len(seeds))
        per = self.task.raw("per_run_gpu_h")
        lines = [
            f"{self.kind}: {self.grid.name}   task: {self.task.name}   "
            f"budgets: {getattr(self.grid, 'budgets', [1.0])}",
            f"train seeds: {seeds}  (n = {n_seeds})",
            f"condition cells: {len(self.units)}",
        ]
        for u in self.units:
            sub = f"  subset {u.subset_index} (seed {u.subset_seed})" if u.subset_index is not None else ""
            lines.append(f"  - {u.condition.label:<28} budget {u.budget}{sub}")
        lines.append(f"runs: {len(self.units)} cells x {n_seeds} seeds = {self.n_runs}")
        lines.append(f"steps per run: {self.steps}")
        lines.append(f"per-run GPU-h: {per}   total GPU-h: {self.gpu_hours}")
        lines += [f"note: {n}" for n in self.notes]
        return lines


def build_units(grid: GridConfig, protocol: ProtocolConfig) -> list[Unit]:
    units: list[Unit] = []
    specs = grid.condition_specs()
    if any(c.kind == "full" for c in specs):
        units.append(Unit(Condition.full(), 1.0, None, None))
    for b in grid.budgets:
        for c in specs:
            if c.kind == "random":
                seeds = protocol.random_subset_seeds
                if c.n_subsets > len(seeds):
                    raise ConfigError(
                        f"grid asks for {c.n_subsets} random subsets, protocol has {len(seeds)}"
                    )
                for i in range(c.n_subsets):
                    units.append(Unit(Condition.random(), float(b), i, seeds[i]))
            elif c.kind == "cleaner":
                units.append(Unit(Condition.cleaner(c.name, c.version), float(b), None, None))
            elif c.kind == "oracle":
                units.append(Unit(Condition.oracle(), float(b), None, None))
    return units


def make_plan(
    grid_path: str,
    mode: str = "plan",
    train_seeds_override: list[int] | None = None,
) -> Plan:
    grid = load_grid(grid_path, mode=mode)
    protocol = load_protocol(resolve_config_path(grid.protocol), mode=mode)
    task = load_task(resolve_config_path(grid.task), mode=mode)
    notes = []
    if train_seeds_override is not None:
        if mode == "result":
            raise ConfigError("train seed overrides are not allowed in result mode")
        seeds: Any = list(train_seeds_override)
        notes.append(f"train seeds OVERRIDDEN for {mode} mode: {seeds}")
    else:
        seeds = protocol.raw("train_seeds")
    return Plan(grid, protocol, task, build_units(grid, protocol), seeds, notes)


def make_calibration_plan(cal_path: str, mode: str = "calibration") -> Plan:
    from robocurate.config import load_calibration

    cal = load_calibration(cal_path, mode=mode)
    protocol = load_protocol(resolve_config_path(cal.protocol), mode=mode)
    task = load_task(resolve_config_path(cal.task), mode=mode)
    units = [Unit(Condition.full(), 1.0, None, None)]
    notes = [f"calibration on {cal.dataset}: full data, {cal.n_final_rollouts} final rollouts, "
             f"{cal.n_intermediate_rollouts} rollouts at intermediate checkpoints "
             f"(save every {cal.raw('intermediate_save_freq')} steps)"]
    return Plan(cal, protocol, task, units, list(cal.train_seeds), notes,
                steps_override=cal.raw("steps"), kind="calibration")
