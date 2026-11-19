"""`robocurate prepare`: compute selections (selection track) and progress/weights (weighting track) ONCE per
(task, budget, mixture replicate, entrant), cache them by input hashes, and record curation cost (D-022).

Runs never call a cleaner; they only read this cache (`load_record`).

Cache layout: <cache>/<task>/<name>-v<version>/budget<b>-rep<r>.json (+ progress.parquet, perm<i>.parquet for
the weighting track). The record's `key` is sha256 over the inputs: dataset fingerprint (label-free view), budget,
replicate, cleaner name/version, protocol tie_break_salt, max_curation_rollouts, selection seed, reference-policy
tree hash and progress-model revision. A record whose inputs no longer match is stale and is recomputed.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from robocurate.cleaners import registry
from robocurate.cleaners.base import Cleaner, WeightingCleaner, budget_count, validate_resources
from robocurate.curation.context import CurationContext, ReferencePolicy
from robocurate.curation.progress import CachedProgress, MockProgressModel
from robocurate.curation.rollouts import RolloutResult, RolloutService
from robocurate.curation.weights import permute_progress, write_progress_parquet
from robocurate.data.episodes import DatasetView, synthetic_view

SCHEMA = 1
FAKE_VIEW_EPISODES = 50


class PrepareError(RuntimeError):
    pass


def selection_seed(task: str, name: str, version: str, budget: float, replicate: int = 0) -> int:
    """rng seed handed to a cleaner (same subset for every train seed). Replicate 0 keeps the pre-D-022 value."""
    tag = f"{task}|{name}|{version}|{budget}" + (f"|rep{replicate}" if replicate else "")
    return int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16)


def fake_view(protocol: Any) -> DatasetView:
    """The label-free view the fake backend uses for prepare and runs (deterministic)."""
    return synthetic_view(FAKE_VIEW_EPISODES, seed=0, tie_break_salt=protocol.tie_break_salt or "")


@dataclass(frozen=True)
class Cell:
    task: str
    kind: str  # cleaner | weighting
    name: str
    version: str
    budget: float
    replicate: int = 0
    n_permutations: int = 0

    def relpath(self) -> Path:
        return Path(self.task) / f"{self.name}-v{self.version}" / f"budget{self.budget}-rep{self.replicate}.json"


def cells_for_plan(plan: Any, replicate: int = 0) -> list[Cell]:
    out, seen = [], set()
    for u in plan.units:
        c = u.condition
        if c.kind not in ("cleaner", "weighting"):
            continue
        cell = Cell(plan.task.name, c.kind, c.name, c.version, u.budget, replicate)
        if cell in seen:
            continue
        seen.add(cell)
        if c.kind == "weighting":
            n = sum(1 for v in plan.units if v.condition.kind == "weight_placebo" and v.condition.name == c.name)
            cell = Cell(cell.task, cell.kind, cell.name, cell.version, cell.budget, replicate, n)
        out.append(cell)
    return out


def _key(inputs: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class FakeRolloutBackend:
    """Synthetic rollouts for fake prepare / conformance: success ~ Bernoulli(0.5) seeded by the env seed;
    states/actions shaped like the view. Says nothing about any policy."""

    def __init__(self, state_dim: int, action_dim: int, length: int = 50):
        self.sd, self.ad, self.length = state_dim, action_dim, length

    def run(self, policy: Any, seeds) -> list[RolloutResult]:
        out = []
        for s in seeds:
            r = np.random.default_rng(int(s))
            out.append(RolloutResult(int(s), bool(r.random() < 0.5), self.length,
                                     r.normal(size=(self.length, self.sd)).astype(np.float32),
                                     r.normal(size=(self.length, self.ad)).astype(np.float32)))
        return out


def fake_reference_policy(view: DatasetView, cache_dir: Path) -> ReferencePolicy:
    """A small MLPPolicy BC-trained briefly on the view: a stand-in full-data checkpoint for tests/conformance."""
    from robocurate.curation.policy import MLPPolicy, train_bc

    path = Path(cache_dir) / "_fake_reference" / view.fingerprint()[:16]
    if not (path / "mlp_policy.pt").exists():
        eps = list(view)
        S = np.concatenate([e.state for e in eps]).astype(np.float32)
        A = np.concatenate([e.actions for e in eps]).astype(np.float32)
        pol = train_bc(MLPPolicy(S.shape[1], A.shape[1], seed=0), S, A, steps=100, seed=0)
        pol.save(path)
    return ReferencePolicy(path, loader=MLPPolicy.load)


def fake_context_factory(protocol: Any, view: DatasetView, cache_dir: Path) -> Callable[[Cleaner], CurationContext]:
    def make(cleaner: Cleaner) -> CurationContext:
        e0 = next(iter(view))
        ctx = CurationContext(backend="fake", progress_model=CachedProgress(MockProgressModel(), cache_dir))
        if cleaner.requires_rollouts:
            ctx.rollouts = RolloutService.from_protocol(
                protocol, FakeRolloutBackend(e0.state.shape[1], e0.actions.shape[1]), cap=cleaner.requires_rollouts)
        if cleaner.requires_reference_policy:
            ctx.reference_policy = fake_reference_policy(view, cache_dir)
        return ctx

    return make


def _inputs(cell: Cell, cleaner: Cleaner, view: DatasetView, protocol: Any, ctx: CurationContext) -> dict[str, Any]:
    return {
        "schema": SCHEMA, "task": cell.task, "kind": cell.kind, "budget": cell.budget, "replicate": cell.replicate,
        "cleaner": cleaner.name, "version": cleaner.version, "view_fingerprint": view.fingerprint(),
        "n_episodes": view.n_episodes, "tie_break_salt": view.tie_break_salt,  # the salt actually used (D-021)
        "max_curation_rollouts": protocol.max_curation_rollouts,
        "selection_seed": selection_seed(cell.task, cleaner.name, cleaner.version, cell.budget, cell.replicate),
        "reference_policy_sha256": ctx.reference_policy.sha256 if ctx.reference_policy else None,
        "progress_model": getattr(ctx.progress_model, "revision", None),
        "weight_permutation_seeds": list(protocol.weight_permutation_seeds or [])[: cell.n_permutations]
        if cell.kind == "weighting" else None,
    }


def prepare_cell(cell: Cell, cleaner: Cleaner, view: DatasetView, protocol: Any, ctx: CurationContext,
                 cache_dir: str | Path, backend: str) -> dict[str, Any]:
    problems = validate_resources(cleaner, protocol.max_curation_rollouts)
    if problems:
        raise PrepareError(f"{cleaner.name}: invalid resource declaration: {problems}")
    if (cell.kind == "weighting") != isinstance(cleaner, WeightingCleaner):
        raise PrepareError(f"{cleaner.name}: grid kind {cell.kind!r} does not match the entrant's track")
    if cleaner.requires_gpu and backend == "local" and not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        # same rule as runs.backends.local.require_gpu_env, without importing runs/ (import-graph rule)
        raise PrepareError(f"{cleaner.name} requires a GPU; CUDA_VISIBLE_DEVICES is empty (GPU use not authorized)")
    cache_dir = Path(cache_dir)
    out = cache_dir / cell.relpath()
    inputs = _inputs(cell, cleaner, view, protocol, ctx)
    key = _key(inputs)
    if out.exists():
        rec = json.loads(out.read_text())
        if rec.get("key") == key:
            rec["_cached"] = True
            return rec
    cleaner.context = ctx
    rng = np.random.default_rng(inputs["selection_seed"])
    rp = ctx.reference_policy
    if rp:
        rp.lock()
    t0 = time.time()
    try:
        rec: dict[str, Any] = {"schema_version": SCHEMA, "key": key, "inputs": inputs, "cleaner": cleaner.metadata(),
                               "backend": backend}
        if cell.kind == "cleaner":
            k = budget_count(view.n_episodes, cell.budget)
            ids = [int(i) for i in cleaner.select(view, k, rng)]
            if len(ids) != k or len(set(ids)) != k or not set(ids) <= set(view.episode_indices):
                raise PrepareError(f"{cleaner.name} violated the selection contract")
            rec["selected_episodes"] = sorted(ids)
            rec["budget_count"] = k
        else:
            prog = cleaner.frame_progress(view, rng)
            lens = view.lengths()
            if set(prog) != set(view.episode_indices) or any(len(prog[i]) != lens[i] for i in lens):
                raise PrepareError(f"{cleaner.name}: frame_progress must give one value per frame of every episode")
            if any(not np.all(np.isfinite(p)) or np.min(p) < 0 or np.max(p) > 1 for p in prog.values()):
                raise PrepareError(f"{cleaner.name}: progress must be finite and in [0, 1]")
            d = out.parent / out.stem
            meta = {"cleaner": f"{cleaner.name}@{cleaner.version}", "key": key}
            rec["progress_parquet"] = str(write_progress_parquet(prog, d / "progress.parquet", meta))
            rec["placebo_parquets"] = {
                str(i): str(write_progress_parquet(permute_progress(prog, s), d / f"perm{i}.parquet",
                                                   {**meta, "permutation_seed": s}))
                for i, s in enumerate(inputs["weight_permutation_seeds"])}
            rec["progress_summary"] = {"mean_final": float(np.mean([p[-1] for p in prog.values()])),
                                       "n_frames": int(sum(lens.values()))}
    finally:
        cleaner.context = None
        if rp:
            rp.unlock()
    wall = time.time() - t0
    if rp:
        rp.verify_unchanged()
    from robocurate.results import git_info

    g = git_info()
    rec.update({
        "wall_clock_s": wall,
        "gpu_h": wall / 3600.0 if (cleaner.requires_gpu and backend == "local") else 0.0,
        "rollouts_used": ctx.rollouts.used if ctx.rollouts else 0,
        "curation_seeds_used": list(ctx.rollouts.seeds_used) if ctx.rollouts else [],
        "git": {"hash": g.hash, "dirty": g.dirty},
        "created_utc": datetime.now(UTC).isoformat(),
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=2) + "\n")
    tmp.replace(out)
    rec["_cached"] = False
    return rec


def prepare_plan(plan: Any, view: DatasetView, cache_dir: str | Path, backend: str,
                 context_factory: Callable[[Cleaner], CurationContext], replicate: int = 0) -> list[dict[str, Any]]:
    out = []
    for cell in cells_for_plan(plan, replicate):
        cleaner = registry.get(cell.name, cell.version)
        out.append(prepare_cell(cell, cleaner, view, plan.protocol, context_factory(cleaner), cache_dir, backend))
    return out


def load_record(cache_dir: str | Path, task: str, kind: str, name: str, version: str, budget: float,
                view: DatasetView, replicate: int = 0) -> dict[str, Any]:
    """The cached record a run must consume. Refuses missing records and records computed on other data."""
    p = Path(cache_dir) / Cell(task, kind, name, version, budget, replicate).relpath()
    if not p.exists():
        raise PrepareError(f"no prepared selection for {name} v{version} budget {budget} rep {replicate}: "
                           f"run `robocurate prepare` first ({p})")
    rec = json.loads(p.read_text())
    if rec["inputs"]["view_fingerprint"] != view.fingerprint():
        raise PrepareError(f"{p}: prepared on different data (fingerprint mismatch); re-run `robocurate prepare`")
    if rec["inputs"]["tie_break_salt"] != view.tie_break_salt:
        raise PrepareError(f"{p}: prepared with a different tie_break_salt; re-run `robocurate prepare`")
    return rec
