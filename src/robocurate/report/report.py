"""`robocurate report`: report.md + leaderboard.json + one PNG from a directory of results JSONs.

Statistics follow D-002 / D-015: one number per training run (its final-checkpoint success rate); a
cleaner's runs vs ALL placebo (random-subset) runs at the same budget, Welch t-test, Holm across cleaners.
The oracle (D-012) and full data are reference conditions: reported with their own (unadjusted) Welch test
against the placebo, never in the cleaners' Holm family. Rollouts are never pooled across runs.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as sps

from robocurate import results
from robocurate.report import stats as S

FAKE_BANNER = "# ⚠️ FAKE RESULTS — SYNTHETIC BACKEND, NOT REAL POLICY PERFORMANCE ⚠️"

# chart tokens (dataviz reference palette, light mode)
SURFACE, INK, INK2, MUTED, GRID, AXIS, SERIES1 = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#2a78d6",
)


class ReportError(ValueError):
    pass


def _label(cond: dict[str, Any]) -> str:
    return f"cleaner:{cond['name']}" if cond["kind"] == "cleaner" else cond["kind"]


def run_level_ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """t-interval for the mean of run-level success rates (runs are the unit, D-002)."""
    n = len(x)
    if n < 2:
        return (float("nan"), float("nan"))
    half = sps.t.ppf(0.5 + level / 2, n - 1) * x.std(ddof=1) / math.sqrt(n)
    return (float(x.mean() - half), float(x.mean() + half))


def _consistency(recs: list[dict]) -> list[str]:
    issues = []
    for key in ("checkpoint_step", "n_episodes"):
        vals = {r[key] for r in recs}
        if len(vals) > 1:
            issues.append(f"runs differ in {key}: {sorted(map(str, vals))} (rule 3/4 violation)")
    cfgs = {tuple(sorted((c["path"].split("/")[-1], c["sha256"]) for c in r["configs"])) for r in recs}
    if len(cfgs) > 1:
        issues.append("runs were produced from different config files (hashes differ)")
    return issues


def analyze(recs: list[dict], alpha: float = 0.05, n_boot: int = 10_000) -> dict[str, Any]:
    if not recs:
        raise ReportError("no results")
    kinds = {r["kind"] for r in recs}
    if len(kinds) > 1:
        raise ReportError(f"refusing to mix result kinds {sorted(kinds)}")
    kind = kinds.pop()
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_task[r["spec"]["task"]].append(r)

    out: dict[str, Any] = {"kind": kind, "fake": kind == "fake", "alpha": alpha, "tasks": []}
    for task, trs in sorted(by_task.items()):
        full = [r for r in trs if r["spec"]["condition"]["kind"] == "full"]
        budgets = sorted({r["spec"]["budget"] for r in trs if r["spec"]["condition"]["kind"] != "full"})
        tinfo: dict[str, Any] = {"task": task, "issues": _consistency(trs), "budgets": []}
        for b in budgets:
            at = [r for r in trs if r["spec"]["budget"] == b and r["spec"]["condition"]["kind"] != "full"]
            groups: dict[str, list[dict]] = defaultdict(list)
            for r in at:
                groups[_label(r["spec"]["condition"])].append(r)
            if full:
                groups["full"] = full
            rate = {k: np.array([r["success_rate"] for r in v]) for k, v in groups.items()}
            placebo = rate.get("random")
            n_subsets = len({r["spec"]["subset_index"] for r in groups.get("random", [])})
            issues = []
            if placebo is None or n_subsets < 3:
                issues.append(f"placebo has {n_subsets} random subsets at budget {b}; need >= 3 (rule 1)")
            if not full:
                issues.append("no full-data runs (rule 1)")

            conditions = []
            for k in sorted(rate, key=lambda s: (["full", "random", "oracle"].index(s) if s in
                                                  ("full", "random", "oracle") else 3, s)):
                x = rate[k]
                lo, hi = run_level_ci(x)
                conditions.append({"condition": k, "n_runs": int(len(x)), "mean": float(x.mean()),
                                   "sd": float(x.std(ddof=1)) if len(x) > 1 else None,
                                   "ci95_run_level": [lo, hi], "run_rates": [float(v) for v in x]})

            cleaners = {k: v for k, v in rate.items() if k.startswith("cleaner:")}
            comps, refs, mde_v = [], [], None
            if placebo is not None and len(placebo) >= 2:
                for c in S.compare_cleaners_to_placebo(cleaners, placebo, alpha=alpha, n_boot=n_boot):
                    d = asdict(c)
                    d["boot_ci"] = list(c.boot_ci)
                    comps.append(d)
                rng = np.random.default_rng(1)
                for ref in ("oracle", "full"):
                    if ref in rate and len(rate[ref]) >= 2:
                        w = S.welch_t_test(rate[ref], placebo)
                        refs.append({"condition": ref, "diff": w.diff, "t": w.t, "df": w.df, "p": w.p,
                                     "significant": bool(w.p <= alpha),
                                     "boot_ci": list(S.bootstrap_diff_ci(rate[ref], placebo, n_boot=n_boot, rng=rng)),
                                     "note": "reference condition; not in the cleaners' Holm family"})
                n_cl = max([len(v) for v in cleaners.values()] or [len(placebo)])
                sigma = float(placebo.std(ddof=1))
                mde_v = {"sigma_run_placebo": sigma, "n_cleaner_runs": n_cl, "n_placebo_runs": int(len(placebo)),
                         "n_comparisons": max(1, len(cleaners)), "power": 0.8,
                         "mde": S.mde(sigma, n_cl, len(placebo), alpha, 0.8, max(1, len(cleaners)))}
            tinfo["budgets"].append({"budget": b, "issues": issues, "conditions": conditions,
                                     "cleaner_vs_placebo": comps, "reference_vs_placebo": refs, "mde": mde_v})
        out["tasks"].append(tinfo)
    return out


def _fmt(x: float | None, nd=3) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def render_markdown(a: dict[str, Any], plot_name: str | None, n_files: int) -> str:
    L: list[str] = []
    if a["fake"]:
        L += [FAKE_BANNER, "", "**These numbers come from the fake backend with injected effect sizes. They "
              "verify the pipeline; they say nothing about any cleaner.**", ""]
    L += [f"# RoboCurate-Bench report ({a['kind']})", "",
          f"Results files: {n_files}. Unit of analysis: training run (D-002). Primary test: Welch t-test of "
          f"each cleaner's runs vs all placebo runs, Holm-corrected across cleaners, alpha = {a['alpha']} "
          "(D-002, D-015). Oracle and full data are reference conditions (unadjusted, outside the Holm family).",
          ""]
    for t in a["tasks"]:
        L += [f"## Task: {t['task']}", ""]
        L += [f"- ⚠️ {i}" for i in t["issues"]]
        for b in t["budgets"]:
            L += [f"### Budget {b['budget']}", ""]
            L += [f"- ⚠️ INVALID COMPARISON: {i}" for i in b["issues"]]
            L += ["| condition | runs | mean success | SD | run-level 95% CI |", "|---|---|---|---|---|"]
            for c in b["conditions"]:
                lo, hi = c["ci95_run_level"]
                L.append(f"| {c['condition']} | {c['n_runs']} | {_fmt(c['mean'])} | {_fmt(c['sd'])} | "
                         f"[{_fmt(lo)}, {_fmt(hi)}] |")
            L.append("")
            if b["cleaner_vs_placebo"]:
                L += ["**Cleaners vs placebo (Holm across cleaners)**", "",
                      "| cleaner | diff | bootstrap 95% CI | t | df | p (raw) | p (Holm) | significant |",
                      "|---|---|---|---|---|---|---|---|"]
                for c in b["cleaner_vs_placebo"]:
                    L.append(f"| {c['name']} | {c['diff']:+.3f} | [{c['boot_ci'][0]:+.3f}, {c['boot_ci'][1]:+.3f}] "
                             f"| {c['t']:.2f} | {c['df']:.1f} | {c['p_raw']:.2g} | {c['p_holm']:.2g} | "
                             f"{'YES' if c['significant'] else 'no'} |")
                L.append("")
            if b["reference_vs_placebo"]:
                L += ["**Reference conditions vs placebo (not entrants)**", "",
                      "| condition | diff | bootstrap 95% CI | t | df | p | significant |",
                      "|---|---|---|---|---|---|---|"]
                for c in b["reference_vs_placebo"]:
                    L.append(f"| {c['condition']} | {c['diff']:+.3f} | [{c['boot_ci'][0]:+.3f}, "
                             f"{c['boot_ci'][1]:+.3f}] | {c['t']:.2f} | {c['df']:.1f} | {c['p']:.2g} | "
                             f"{'YES' if c['significant'] else 'no'} |")
                L.append("")
            if b["mde"]:
                m = b["mde"]
                L += [f"**Minimum detectable effect** (80% power, Bonferroni over {m['n_comparisons']} "
                      f"cleaner(s), sigma_run from placebo runs = {m['sigma_run_placebo']:.3f}, "
                      f"{m['n_cleaner_runs']} vs {m['n_placebo_runs']} runs): **{m['mde']:.3f}** "
                      f"({100 * m['mde']:.1f} pp).", ""]
    if plot_name:
        L += [f"![per-run success by condition]({plot_name})", ""]
    if a["fake"]:
        L += ["", FAKE_BANNER]
    return "\n".join(L) + "\n"


def render_plot(a: dict[str, Any], path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [(t["task"], b) for t in a["tasks"] for b in t["budgets"]]
    fig, axes = plt.subplots(1, len(panels), figsize=(max(7.5, 7.5 * len(panels)), 4.6), squeeze=False,
                             facecolor=SURFACE)
    for ax, (task, b) in zip(axes[0], panels, strict=True):
        ax.set_facecolor(SURFACE)
        conds = b["conditions"]
        rng = np.random.default_rng(0)
        for i, c in enumerate(conds):
            x = np.asarray(c["run_rates"])
            jit = rng.uniform(-0.12, 0.12, len(x))
            ax.scatter(i + jit, x, s=42, color=SERIES1, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            lo, hi = c["ci95_run_level"]
            if not math.isnan(lo):
                ax.plot([i + 0.28, i + 0.28], [lo, hi], color=INK, linewidth=2, solid_capstyle="round", zorder=4)
            ax.plot([i + 0.18, i + 0.38], [c["mean"]] * 2, color=INK, linewidth=2, zorder=4)
        ax.set_xticks(range(len(conds)), [c["condition"].replace("cleaner:", "") for c in conds],
                      color=INK2, fontsize=9)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("success rate (final checkpoint)", color=INK2, fontsize=9)
        ax.tick_params(axis="y", colors=MUTED, labelsize=8)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.set_title(f"Per-run success by condition — {task}, budget {b['budget']}\n"
                     "dots = training runs; bar = mean with run-level 95% CI",
                     color=INK, fontsize=10, loc="left")
        if a["fake"]:
            ax.text(0.5, 0.5, "FAKE", transform=ax.transAxes, fontsize=110, color="#c0392b", alpha=0.18,
                    ha="center", va="center", rotation=20, fontweight="bold", zorder=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def write_report(results_dir: str | Path, out_dir: str | Path | None = None, n_boot: int = 10_000) -> dict:
    results_dir = Path(results_dir)
    out_dir = Path(out_dir) if out_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = results.load_dir(results_dir)
    a = analyze(recs, n_boot=n_boot)
    plot = render_plot(a, out_dir / ("FAKE_success_by_condition.png" if a["fake"] else "success_by_condition.png"))
    (out_dir / "report.md").write_text(render_markdown(a, plot.name, len(recs)))
    a["generated_utc"] = datetime.now(UTC).isoformat()
    a["n_results"] = len(recs)
    g = results.git_info()
    a["git"] = {"hash": g.hash, "dirty": g.dirty}
    (out_dir / "leaderboard.json").write_text(json.dumps(a, indent=2) + "\n")
    return a
