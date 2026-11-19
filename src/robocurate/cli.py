"""`robocurate` command-line interface: plan | run | report | submit."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from robocurate.config import ConfigError, config_type, tbd_fields

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
    if not _repo_path(FROZEN_MARKER).exists():
        raise ConfigError(f"result mode refused: {FROZEN_MARKER} does not exist (eval code not frozen)")
    missing = tbd_fields(plan.protocol) + [f"task.{f}" for f in tbd_fields(plan.task)]
    if missing:
        raise ConfigError(f"result mode refused: TBD values remain: {', '.join(missing)}")


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
        backend = FakeBackend(plan, params, command_line=[Path(sys.argv[0]).name, *sys.argv[1:]])
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
        backend = LocalBackend(plan, mode, dataset, root, command_line=[Path(sys.argv[0]).name, *sys.argv[1:]])

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
    if a["fake"]:
        print("*** FAKE RESULTS (synthetic backend) ***")
    for t in a["tasks"]:
        for b in t["budgets"]:
            print(f"task {t['task']} budget {b['budget']}:")
            for c in b["conditions"]:
                print(f"  {c['condition']:<28} runs={c['n_runs']:<3} mean={c['mean']:.3f}")
            for c in b["cleaner_vs_placebo"]:
                print(f"  {c['name']} vs placebo: diff={c['diff']:+.3f} p_holm={c['p_holm']:.3g} "
                      f"{'SIGNIFICANT' if c['significant'] else 'not significant'}")
            for c in b["reference_vs_placebo"]:
                print(f"  [ref] {c['condition']} vs placebo: diff={c['diff']:+.3f} p={c['p']:.3g} "
                      f"{'SIGNIFICANT' if c['significant'] else 'not significant'}")
            if b["mde"]:
                print(f"  MDE (80% power): {b['mde']['mde']:.3f}")
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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="robocurate", description="RoboCurate-Bench")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="expand a grid or calibration config; run count and TBD-aware cost")
    p.add_argument("config")
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("run", help="execute a grid (idempotent, restart-safe)")
    p.add_argument("config")
    p.add_argument("--backend", choices=["fake", "local"], required=True)
    p.add_argument("--calibration", action="store_true", help="config is a calibration config")
    p.add_argument("--fake-params", help="fake backend parameters YAML (fake only)")
    p.add_argument("--results-dir", help="default: $ROBOCURATE_SCRATCH/results/<name>-<backend>")
    p.add_argument("--dataset-root", help="local root of the training dataset (local backend)")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("report", help="report.md + leaderboard.json + plot from results JSONs")
    p.add_argument("results_dir")
    p.add_argument("--out", help="output dir (default: results_dir)")
    p.add_argument("--n-boot", type=int, default=10_000)
    p.set_defaults(fn=cmd_report)

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
    from robocurate.paths import StorageError
    from robocurate.runs.backends.local import LocalBackendRefused

    try:
        return args.fn(args)
    except (ConfigError, StorageError, LocalBackendRefused) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
