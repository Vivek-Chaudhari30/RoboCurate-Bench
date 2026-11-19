"""`robocurate` command-line interface: plan | run | report | submit."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from robocurate.config import ConfigError, config_type, is_tbd, tbd_fields

FROZEN_MARKER = Path("src/robocurate/eval/FROZEN")


def _repo_path(p: str | Path) -> Path:
    from robocurate.paths import repo_root

    return repo_root() / p


def _make_plan(path: str, mode: str, seeds_override=None):
    from robocurate.runs.planner import make_calibration_plan, make_plan

    if config_type(path) == "calibration":
        return make_calibration_plan(path, mode=mode)
    return make_plan(path, mode=mode, train_seeds_override=seeds_override)


def cmd_plan(args) -> int:
    plan = _make_plan(args.config, "plan")
    print("\n".join(plan.summary_lines()))
    missing = tbd_fields(plan.protocol) + [f"task.{f}" for f in tbd_fields(plan.task)]
    if missing:
        print(f"TBD (blocks result runs, D-014): {', '.join(missing)}")
    return 0


def _result_mode_guard(plan) -> None:
    from robocurate.config import check_result_subsets
    from robocurate.runs.planner import check_result_design

    check_result_subsets(plan.grid)  # D-019/D-026: K >= 8
    if not is_tbd(plan.train_seeds):  # TBD seeds are refused below
        check_result_design(plan)  # D-026: balanced n_c = n_p = n >= 3, K >= 8 permutations
    if not _repo_path(FROZEN_MARKER).exists():
        raise ConfigError(f"result mode refused: {FROZEN_MARKER} does not exist (eval code not frozen)")
    missing = tbd_fields(plan.protocol) + [f"task.{f}" for f in tbd_fields(plan.task)]
    if missing:
        raise ConfigError(f"result mode refused: TBD values remain: {', '.join(missing)}")


def _selections_dir(args, plan, backend: str) -> Path:
    if getattr(args, "selections_dir", None):
        return Path(args.selections_dir)
    from robocurate.paths import scratch_dir

    return scratch_dir("selections", create=False) / f"{plan.grid.name}-{backend}"


def cmd_prepare(args) -> int:
    """D-022: compute + cache selections / weighting progress once per (task, budget, replicate, entrant)."""
    from robocurate.curation import prepare as P
    from robocurate.paths import require_storage_env

    require_storage_env()
    plan = _make_plan(args.config, "fake" if args.backend == "fake" else "plan")
    out = _selections_dir(args, plan, args.backend)
    if args.backend == "fake":
        view = P.fake_view(plan.protocol)
        factory = P.fake_context_factory(plan.protocol, view, out)
    else:
        raise ConfigError("prepare --backend local needs the mixture dataset, Robometer weights and (for "
                          "reference-policy entrants) the M0 full-data checkpoint; not available in this milestone")
    recs = P.prepare_plan(plan, view, out, args.backend, factory, replicate=args.replicate)
    for r in recs:
        c = r["cleaner"]
        what = (f"{len(r['selected_episodes'])} episodes" if "selected_episodes" in r
                else f"progress + {len(r['placebo_progress_parquets'])} D-028 placebo progress parquets")
        print(f"{c['name']} v{c['version']} [{c['track']}] budget {r['inputs']['budget']}: {what}; "
              f"{'cached' if r['_cached'] else 'computed'} in {r['wall_clock_s']:.2f}s, GPU-h {r['gpu_h']:.3f}, "
              f"rollouts {r['rollouts_used']}, key {r['key'][:12]}")
    print(f"selections -> {out}")
    return 0


def cmd_run(args) -> int:
    from robocurate.paths import require_storage_env
    from robocurate.runs import executor

    is_cal = config_type(args.config) == "calibration"
    if args.calibration != is_cal:
        raise ConfigError("--calibration must be given exactly when the config is a calibration config")

    if args.backend == "fake":
        from robocurate.runs.backends.fake import FakeBackend, load_fake_params

        if not args.fake_params:
            raise ConfigError("--backend fake requires --fake-params")
        params = load_fake_params(args.fake_params)
        plan = _make_plan(args.config, "fake", None if is_cal else params.train_seeds)
        backend = FakeBackend(plan, params, command_line=args._command_line,
                              selections_dir=_selections_dir(args, plan, "fake"), replicate=args.replicate)
    else:
        if args.fake_params:
            raise ConfigError("--fake-params is only valid with --backend fake")
        mode = "calibration" if is_cal else "result"
        plan = _make_plan(args.config, mode)
        if mode == "result":
            _result_mode_guard(plan)
        from robocurate.runs.backends.local import LocalBackend, require_gpu_env

        require_gpu_env()  # refuses when CUDA_VISIBLE_DEVICES is empty (CPU-only tasks)
        dataset = plan.grid.dataset if is_cal else plan.task.train_dataset
        root = args.dataset_root
        if root is None and not is_cal:  # mixtures are built under $ROBOCURATE_SCRATCH/datasets/<repo_id>
            from robocurate.paths import scratch_dir

            root = scratch_dir("datasets", create=False) / dataset
        backend = LocalBackend(plan, mode, dataset, root, command_line=args._command_line,
                               selections_dir=_selections_dir(args, plan, "local"), replicate=args.replicate)

    scratch = require_storage_env()
    out = Path(args.results_dir) if args.results_dir else scratch / "results" / f"{plan.grid.name}-{args.backend}"
    specs = plan.specs()
    print(f"{len(specs)} runs -> {out}")
    rep = executor.execute(specs, backend, out)
    print(
        f"done={len(rep.done)} skipped_existing={len(rep.skipped_existing)} "
        f"skipped_locked={len(rep.skipped_locked)} reclaimed_stale_locks={len(rep.reclaimed_stale)}"
    )
    return 0


def cmd_report(args) -> int:
    from robocurate.report.report import write_report

    a = write_report(args.results_dir, args.out, n_boot=args.n_boot)
    out = Path(args.out or args.results_dir)
    if args.site:
        from robocurate.report.site import write_site

        print(f"site: {write_site(a, args.site)}")
    if a["fake"]:
        print("*** FAKE RESULTS (synthetic backend) ***")
    from robocurate.report.report import _panels

    for task, title, b in _panels(a):
        print(f"task {task} {title}:")
        if not b.get("result_eligible", False):
            from robocurate.report.report import INELIGIBLE

            print(f"  *** {INELIGIBLE}: {'; '.join(b.get('eligibility') or [])} ***")
        for c in b["conditions"]:
            print(f"  {c['condition']:<28} runs={c['n_runs']:<3} mean={c['mean']:.3f}  cell {c['cell_id']}")
        for c in b["cleaner_vs_placebo"]:
            w = c.get("welch_secondary") or {}
            print(f"  {c['name']} vs placebo [D-026]: diff={c['diff']:+.3f} SE={c['se']:.3f} df={c['df']:.0f} "
                  f"p_holm={c['p_holm']:.3g} {'SIGNIFICANT' if c['significant'] else 'not significant'}"
                  f"   (Welch secondary p_holm={w.get('p_holm', float('nan')):.3g})")
        for c in b["reference_vs_placebo"]:
            print(f"  [ref] {c['condition']} vs placebo: diff={c['diff']:+.3f} p={c['p']:.3g} "
                  f"{'SIGNIFICANT' if c['significant'] else 'not significant'}")
        if b["mde"]:
            m = b["mde"]
            print(f"  s_sub={m['s_sub']:.3f}{' (truncated)' if m['s_sub_truncated'] else ''} s_seed={m['s_seed']:.3f} "
                  f"K={m['K']} n_p={m['n_p']} n_c={m['n_c']}  MDE (80% power): {m['mde']:.3f}")
        for i in b["issues"]:
            print(f"  INVALID: {i}")
    print(f"wrote {out / 'report.md'}, {out / 'leaderboard.json'}")
    return 0


def cmd_submit(args) -> int:
    from robocurate.cleaners import conformance
    from robocurate.data.episodes import synthetic_view

    view = synthetic_view()
    rep = conformance.check(args.cleaner, view, args.budget_fraction, args.time_limit)
    print("\n".join(rep.lines()))
    if not rep.passed:
        return 1
    if args.out:
        out = Path(args.out)
    else:
        from robocurate.paths import scratch_dir

        out = scratch_dir("selections") / f"{rep.cleaner.name}-v{rep.cleaner.version}.json"
    conformance.write_selection(rep, view, out, args.budget_fraction)
    print(f"selection written: {out}")
    return 0


def cmd_validate_defects(args) -> int:
    """Defect statistics from actions/state/timestamps only (no video). Args: local dirs or repo_id[@revision]."""
    from robocurate.data.validate import analyze, fetch_non_video, load_parquet_dataset
    from robocurate.paths import require_storage_env

    require_storage_env()
    out = {}
    for src in args.datasets:
        p = Path(src)
        if p.is_dir():
            root, rev = p, None
        else:
            repo, _, rev = src.partition("@")
            if not rev:
                raise ConfigError(f"{src}: pin a revision (repo_id@sha)")
            root = fetch_non_video(repo, rev)
        info, eps = load_parquet_dataset(root)
        a = analyze(info, eps)
        a["source"], a["revision"] = src, rev
        out[src] = a
        print(f"{src}: {a['n_episodes']} eps @ {a['fps']:g} fps | pause rate {a['pause']['episode_rate']:.2f} "
              f"(interior {a['pause']['interior_episode_rate']:.2f}) | "
              f"jitter>0.10 {a['jitter']['episodes_above_0.10']:.2f} | exact dups "
              f"{a['duplicates']['exact_duplicate_episodes']}, near-dup pairs {a['duplicates']['near_duplicate_pairs']}"
              f" | len p50 {a['length_seconds'].get('p50', 0):.1f}s | tasks {a['tasks']['distinct']}")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, default=float) + "\n")
        print(f"wrote {args.out}")
    return 0


def cmd_reproduce(args) -> int:
    from robocurate.runs import reproduce as RP

    info = RP.resolve(args.results_dir, args.cell_id)
    head = results_git_head()
    print(f"cell {info['cell_id']}: task {info['task']} budget {info['budget']} condition {info['condition']} "
          f"({len(info['runs'])} runs, kind {info['kind']})")
    print("git:", ", ".join(f"{h[:10]}{' (dirty)' if d else ''}" for h, d in info["git"]))
    for p, st in info["config_checks"].items():
        print(f"config {p}: sha256 {info['configs'][p][:12]} {st}")
    if info["selection_keys"]:
        print("selection cache keys:", ", ".join(k[:12] for k in info["selection_keys"]))
    print("commands:")
    for c in RP.commands(info, head):
        print("  " + c)
    if args.backend == "fake":
        ok, diffs = RP.rerun_fake(info, args.selections_dir)
        print(f"re-executed: MATCH {ok}/{len(info['runs'])}")
        for d in diffs:
            print("  DIFF " + d)
        return 0 if not diffs else 1
    return 0


def results_git_head() -> str | None:
    from robocurate.results import git_info

    return git_info().hash


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="robocurate", description="RoboCurate-Bench")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="expand a grid or calibration config; run count and TBD-aware cost")
    p.add_argument("config")
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("prepare", help="compute + cache entrant selections / weights once (D-022)")
    p.add_argument("config")
    p.add_argument("--backend", choices=["fake", "local"], required=True)
    p.add_argument("--selections-dir", help="default: $ROBOCURATE_SCRATCH/selections/<grid>-<backend>")
    p.add_argument("--replicate", type=int, default=0, help="mixture replicate index")
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("run", help="execute a grid (idempotent, restart-safe)")
    p.add_argument("config")
    p.add_argument("--backend", choices=["fake", "local"], required=True)
    p.add_argument("--calibration", action="store_true", help="config is a calibration config")
    p.add_argument("--fake-params", help="fake backend parameters YAML (fake only)")
    p.add_argument("--results-dir", help="default: $ROBOCURATE_SCRATCH/results/<name>-<backend>")
    p.add_argument("--dataset-root", help="local root of the training dataset (local backend)")
    p.add_argument("--selections-dir", help="prepare cache (default: $ROBOCURATE_SCRATCH/selections/<grid>-<backend>)")
    p.add_argument("--replicate", type=int, default=0, help="mixture replicate index")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("report", help="report.md + leaderboard.json + plot from results JSONs")
    p.add_argument("results_dir")
    p.add_argument("--out", help="output dir (default: results_dir)")
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--site", help="also write a static leaderboard (index.html + leaderboard.json) here")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("reproduce", help="resolve a leaderboard cell; print its exact commands; rerun (fake)")
    p.add_argument("cell_id")
    p.add_argument("--results-dir", required=True)
    p.add_argument("--backend", choices=["fake"], help="re-execute the cell's runs and compare (fake only)")
    p.add_argument("--selections-dir", help="prepare cache used by the original runs (fake re-execution)")
    p.set_defaults(fn=cmd_reproduce)

    p = sub.add_parser("validate-defects", help="defect stats from actions/state/timestamps (no video download)")
    p.add_argument("datasets", nargs="+", help="local dataset dirs or repo_id@revision")
    p.add_argument("--out", help="write the full statistics as JSON")
    p.set_defaults(fn=cmd_validate_defects)

    p = sub.add_parser("submit", help="conformance-check a cleaner (path.py, module:Class, or registered name)")
    p.add_argument("cleaner")
    p.add_argument("--out", help="selection JSON path (default: $ROBOCURATE_SCRATCH/selections/)")
    p.add_argument("--budget-fraction", type=float, default=0.5)
    p.add_argument("--time-limit", type=float, default=60.0)
    p.set_defaults(fn=cmd_submit)
    return ap


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    args = build_parser().parse_args(argv)
    # the exact command recorded in results (regression: in-process calls recorded the caller's sys.argv)
    args._command_line = ["robocurate", *(sys.argv[1:] if argv is None else list(argv))]
    from robocurate.curation.prepare import PrepareError
    from robocurate.paths import StorageError
    from robocurate.runs.backends.local import LocalBackendRefused
    from robocurate.runs.reproduce import ReproduceError

    try:
        return args.fn(args)
    except (ConfigError, StorageError, LocalBackendRefused, PrepareError, ReproduceError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
