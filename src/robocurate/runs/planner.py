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
        """Random-subset seed (random) or weight-permutation seed (weight_placebo, D-020)."""
        if spec.subset_index is None:
            return None
        if spec.condition.kind == "weight_placebo":
            return self.protocol.weight_permutation_seeds[spec.subset_index]
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
    for c in specs:  # weighting track (D-020): full data, budget-independent
        if c.kind == "weighting":
            seeds = protocol.weight_permutation_seeds or []
            if c.n_permutations > len(seeds):
                raise ConfigError(f"grid asks for {c.n_permutations} weight permutations, protocol has {len(seeds)}")
            units.append(Unit(Condition.weighting(c.name, c.version), 1.0, None, None))
            for i in range(c.n_permutations):
                units.append(Unit(Condition.weight_placebo(c.name, c.version), 1.0, i, seeds[i]))
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


PLACEBO_KINDS = ("random", "weight_placebo")


def unit_seed_counts(plan: Plan) -> dict[tuple[str, float, int | None], int]:
    """Training runs (seeds) per unit (condition x budget x subset/permutation)."""
    counts: dict[tuple[str, float, int | None], int] = {}
    for s in plan.specs():
        key = (s.condition.label, s.budget, s.subset_index)
        counts[key] = counts.get(key, 0) + 1
    return counts


def placebo_seed_counts(plan: Plan) -> dict[tuple[str, float, int], int]:
    """n_p for each random subset / weight permutation."""
    return {k: n for k, n in unit_seed_counts(plan).items() if k[0].split(":")[0] in PLACEBO_KINDS}


def check_result_design(plan: Plan) -> None:
    """D-026: result-mode runs need a balanced design: every condition unit (entrants, references, each placebo
    subset / weight permutation) has the same n seeds, n >= 3 (n_c = n_p = n), and K >= 8 placebo units per
    track (K for selection is checked on the grid by `check_result_subsets`; permutations here)."""
    from robocurate.config import RESULT_MIN_SUBSETS, SEED_FLOOR

    counts = unit_seed_counts(plan)
    ns = sorted(set(counts.values()))
    if len(ns) > 1:
        lo = min(counts.items(), key=lambda kv: kv[1])
        hi = max(counts.items(), key=lambda kv: kv[1])
        raise ConfigError(f"result mode refused: unbalanced design (seeds per unit range {ns[0]}..{ns[-1]}, e.g. "
                          f"{lo[0][0]}: {lo[1]} vs {hi[0][0]}: {hi[1]}); D-026 requires n_c = n_p = n")
    if ns and ns[0] < SEED_FLOOR:
        raise ConfigError(f"result mode refused: n = {ns[0]} seeds per condition; D-026 requires n >= {SEED_FLOOR}")
    perms: dict[str, set] = {}
    for (label, _b, idx) in counts:
        if label.startswith("weight_placebo"):
            perms.setdefault(label, set()).add(idx)
    for label, idx in perms.items():
        if len(idx) < RESULT_MIN_SUBSETS:
            raise ConfigError(f"result mode refused: {label} has K = {len(idx)} weight permutations; D-026 requires "
                              f"K >= {RESULT_MIN_SUBSETS}")


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
