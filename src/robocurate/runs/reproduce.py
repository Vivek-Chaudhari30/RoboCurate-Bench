"""`robocurate reproduce <cell-id>`: resolve one leaderboard cell (task x budget x condition) to its runs, the
git hash, the configs (path + sha256) and the selection cache entry, print the exact commands, and with
`--backend fake` re-execute every run of the cell and compare per-episode outcomes bit for bit."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from robocurate import results
from robocurate.config import sha256_file
from robocurate.report.report import cell_id
from robocurate.runs.spec import Condition, RunSpec


class ReproduceError(RuntimeError):
    pass


def cells(recs: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        sp = r["spec"]
        out[cell_id(sp["task"], sp["budget"], Condition(**sp["condition"]).label)].append(r)
    return out


def resolve(results_dir: str | Path, cid: str) -> dict[str, Any]:
    recs = results.load_dir(results_dir)
    cs = cells(recs)
    if cid not in cs:
        raise ReproduceError(f"unknown cell {cid!r}; known: {sorted(cs)}")
    rs = cs[cid]
    git = sorted({(r["git"]["hash"], r["git"]["dirty"]) for r in rs})
    # configs are recorded in a fixed order by every backend: protocol, task, grid[, fake params]
    cfgs = {c["path"]: c["sha256"] for c in rs[0]["configs"]}
    if any({c["path"]: c["sha256"] for c in r["configs"]} != cfgs for r in rs):
        raise ReproduceError(f"runs of cell {cid} were produced from different configs")
    checks = {p: ("ok" if Path(p).exists() and sha256_file(p) == h else "MISMATCH or missing") for p, h in cfgs.items()}
    sel = sorted({r.get("extra", {}).get("selection_key") for r in rs} - {None})
    sp = rs[0]["spec"]
    return {"cell_id": cid, "task": sp["task"], "budget": sp["budget"], "condition": sp["condition"],
            "kind": rs[0]["kind"], "runs": rs, "git": git, "configs": cfgs, "config_checks": checks,
            "selection_keys": sel, "command_lines": [r["command_lines"] for r in rs]}


def commands(info: dict[str, Any], current_head: str | None) -> list[str]:
    L = []
    for h, dirty in info["git"]:
        if h != current_head:
            L.append(f"git checkout {h}" + ("   # NOTE: original tree was dirty" if dirty else ""))
    first = info["command_lines"][0]
    for cmd in first:
        L.append(" ".join(map(str, cmd)))
    if info["kind"] == "fake":
        grid = list(info["configs"])[2]
        L.insert(len(L) - len(first), f"robocurate prepare {grid} --backend fake --selections-dir <SELECTIONS>")
    return L


def rerun_fake(info: dict[str, Any], selections_dir: str | Path) -> tuple[int, list[str]]:
    """Re-execute every run of a FAKE cell; returns (#matching runs, diffs)."""
    from robocurate.runs.backends.fake import FakeBackend, load_fake_params
    from robocurate.runs.planner import make_plan
    from robocurate.runs.spec import run_id

    if info["kind"] != "fake":
        raise ReproduceError("only fake cells can be re-executed on CPU")
    bad = [p for p, s in info["config_checks"].items() if s != "ok"]
    if bad:
        raise ReproduceError(f"config files changed since the runs: {bad}")
    paths = list(info["configs"])  # protocol, task, grid, fake params
    if len(paths) < 4:
        raise ReproduceError("fake run records must list protocol, task, grid and fake-params configs")
    grid, fake = paths[2], paths[3]
    params = load_fake_params(fake)
    plan = make_plan(grid, mode="fake", train_seeds_override=params.train_seeds)
    be = FakeBackend(plan, params, command_line=["robocurate", "reproduce"], selections_dir=selections_dir)
    ok, diffs = 0, []
    for r in info["runs"]:
        spec = RunSpec.from_dict(r["spec"])
        rid = run_id(spec, be.config_hashes())
        if rid != r["run_id"]:
            diffs.append(f"{r['run_id']}: run_id now {rid} (config hashes differ)")
            continue
        new = be.run(spec, rid)
        if new["per_episode_success"] == r["per_episode_success"] and \
                new.get("extra", {}).get("selection_key") == r.get("extra", {}).get("selection_key"):
            ok += 1
        else:
            diffs.append(f"{r['run_id']}: outcome or selection differs")
    return ok, diffs
