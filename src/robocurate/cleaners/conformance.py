"""`robocurate submit`: conformance checks for a cleaner.

Checks (all must pass):
  1. static      the cleaner's source cannot reach robocurate.data.labels / robocurate.eval /
                 robocurate.runs.oracle (transitively through robocurate's own modules), and uses no dynamic
                 import / exec / eval. Run BEFORE the module is imported.
  2. loads       the module imports and yields exactly one Cleaner with non-empty name and version.
  3. rollouts    requires_rollouts <= CURATION_ROLLOUT_CAP (D-007).
  4. count       select() returns exactly `budget` ids.
  5. unique      ... with no duplicates.
  6. valid       ... all of which are episode indices of the view (ints).
  7. determinism two calls with identically seeded Generators return identical ids.
  8. runtime     each select() call finishes within the time limit.
  9. no-leak     no forbidden module appears in sys.modules after running the cleaner.
On a pass, writes a selection JSON for the conformance view.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from robocurate.cleaners import importguard, registry
from robocurate.cleaners.base import CURATION_ROLLOUT_CAP, Cleaner
from robocurate.data.episodes import DatasetView, synthetic_view

CONFORMANCE_SEED = 12345


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class ConformanceReport:
    target: str
    checks: list[Check] = field(default_factory=list)
    cleaner: Cleaner | None = None
    selection: list[int] | None = None

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    def add(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append(Check(name, bool(ok), detail))
        return bool(ok)

    def lines(self) -> list[str]:
        out = [f"conformance: {self.target}"]
        for c in self.checks:
            out.append(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}" + (f": {c.detail}" if c.detail else ""))
        out.append(f"RESULT: {'PASS' if self.passed else 'FAIL'}")
        return out


class _Timeout(Exception):
    pass


def _raise_timeout(signum, frame):
    raise _Timeout()


def _call_with_timeout(fn: Callable[[], Any], seconds: float) -> Any:
    """SIGALRM-based timeout on POSIX in the main thread; elsewhere only post-hoc timing applies."""
    use_alarm = hasattr(signal, "setitimer") and threading.current_thread() is threading.main_thread()
    if not use_alarm:
        return fn()
    old = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _source_for_target(target: str) -> tuple[str, str, bool, Path | None]:
    """(source, module name, is_pkg, file path) without importing the target."""
    p = Path(target)
    if p.suffix == ".py" and p.is_file():
        return p.read_text(), "__submitted__", False, p
    mod = target.split(":", 1)[0]
    if mod in registry.BUILTINS:
        cls = registry.BUILTINS[mod]
        mod = cls.__module__
    else:
        eps = registry.plugin_entry_points()
        if mod in eps:
            mod = eps[mod].value.split(":", 1)[0]
    spec = importlib.util.find_spec(mod)
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        raise ModuleNotFoundError(f"cannot locate source for {target!r}")
    return Path(spec.origin).read_text(), mod, spec.submodule_search_locations is not None, Path(spec.origin)


def _load_cleaner(target: str) -> Cleaner:
    p = Path(target)
    if p.suffix == ".py" and p.is_file():
        name = f"_robocurate_submitted_{abs(hash(str(p.resolve())))}"
        spec = importlib.util.spec_from_file_location(name, p)
        module: ModuleType = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        cands = [
            v for v in vars(module).values()
            if inspect.isclass(v) and issubclass(v, Cleaner) and not inspect.isabstract(v)
            and v.__module__ == module.__name__
        ]
        if len(cands) != 1:
            raise registry.RegistryError(f"{p}: expected exactly one concrete Cleaner class, found {len(cands)}")
        return registry.instantiate(cands[0])
    if ":" in target:
        mod, attr = target.split(":", 1)
        return registry.instantiate(getattr(importlib.import_module(mod), attr))
    return registry.get(target)


def check(
    target: str,
    view: DatasetView | None = None,
    budget_fraction: float = 0.5,
    time_limit_s: float = 60.0,
    seed: int = CONFORMANCE_SEED,
) -> ConformanceReport:
    rep = ConformanceReport(target)
    view = view or synthetic_view()
    budget = max(1, int(round(view.n_episodes * budget_fraction)))

    # 1. static (before import)
    try:
        src, mod, is_pkg, _ = _source_for_target(target)
    except (ModuleNotFoundError, ValueError) as e:
        rep.add("static", False, str(e))
        return rep
    reach = importguard.forbidden_reach(src, mod, is_pkg)
    sus = importguard.suspicious_uses(src)
    detail = "; ".join([" -> ".join(r) for r in reach] + sus)
    if not rep.add("static", not reach and not sus, detail):
        return rep  # never import code that fails the static check

    already = {m for m in sys.modules if importguard._is_forbidden(m)}

    # 2. loads
    try:
        cleaner = _load_cleaner(target)
    except Exception as e:  # noqa: BLE001 - report any load failure
        rep.add("loads", False, f"{type(e).__name__}: {e}")
        return rep
    rep.cleaner = cleaner
    rep.add("loads", True, f"{cleaner.name} v{cleaner.version}")

    # 3. rollouts cap
    rep.add(
        "rollouts",
        isinstance(cleaner.requires_rollouts, int) and 0 <= cleaner.requires_rollouts <= CURATION_ROLLOUT_CAP,
        f"requires_rollouts={cleaner.requires_rollouts} (cap {CURATION_ROLLOUT_CAP})",
    )

    # run twice with identically seeded generators
    outs: list[list[int]] = []
    for _ in range(2):
        t0 = time.time()
        try:
            ids = _call_with_timeout(
                lambda: cleaner.select(view, budget, np.random.default_rng(seed)), time_limit_s
            )
        except _Timeout:
            rep.add("runtime", False, f"exceeded {time_limit_s}s")
            return rep
        except Exception as e:  # noqa: BLE001
            rep.add("select", False, f"{type(e).__name__}: {e}")
            return rep
        dt = time.time() - t0
        if dt > time_limit_s:
            rep.add("runtime", False, f"{dt:.1f}s > {time_limit_s}s")
            return rep
        outs.append(list(ids))
    a, b = outs
    valid_ids = set(view.episode_indices)
    rep.add("runtime", True, f"limit {time_limit_s}s")
    rep.add("count", len(a) == budget, f"returned {len(a)}, budget {budget}")
    rep.add("unique", len(set(a)) == len(a), f"{len(a) - len(set(a))} duplicates")
    ints = all(isinstance(i, (int, np.integer)) and not isinstance(i, bool) for i in a)
    rep.add("valid", ints and set(a) <= valid_ids, "ids must be episode indices of the view")
    rep.add("determinism", a == b, "same rng seed must give the same ids")
    leaked = {m for m in sys.modules if importguard._is_forbidden(m)} - already
    rep.add("no-leak", not leaked, ", ".join(sorted(leaked)))
    if rep.passed:
        rep.selection = sorted(int(i) for i in a)
    return rep


def write_selection(rep: ConformanceReport, view: DatasetView, out: str | Path, budget_fraction: float) -> Path:
    if not rep.passed or rep.cleaner is None or rep.selection is None:
        raise ValueError("refusing to write a selection for a failed conformance check")
    from robocurate.results import git_info

    g = git_info()
    d = {
        "schema_version": 1,
        "cleaner": rep.cleaner.name,
        "cleaner_version": rep.cleaner.version,
        "target": rep.target,
        "dataset": view.name,
        "dataset_fingerprint": view.fingerprint(),
        "n_episodes": view.n_episodes,
        "budget_fraction": budget_fraction,
        "budget": len(rep.selection),
        "rng_seed": CONFORMANCE_SEED,
        "selected_episode_indices": rep.selection,
        "git": {"hash": g.hash, "dirty": g.dirty},
        "created_utc": datetime.now(UTC).isoformat(),
        "checks": [c.__dict__ for c in rep.checks],
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, indent=2) + "\n")
    return out
