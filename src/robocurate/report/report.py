"""`robocurate report`: report.md + leaderboard.json + one PNG from a directory of results JSONs.

Statistics (D-002 as amended by D-016, D-019 and D-026): one number per training run (its final-checkpoint success
rate). Primary test (D-026): exact balanced prediction t-test of each entrant's runs vs the K placebo subsets,
SE^2 = (1 + 1/K) var(subset means), df = K - 1, Holm across entrants. The D-016/D-019 variance-components test is a
labeled secondary diagnostic. Weighting track (D-020): each entrant vs its own permuted-weights
placebo, Holm across weighting entrants; unweighted full data is a reference. Oracle (D-012) and full data are
reference conditions outside the Holm family. Welch vs pooled placebo runs is kept as a labeled SECONDARY
diagnostic. Rollouts are never pooled across runs.
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


def cell_id(task: str, budget: float, label: str) -> str:
    """Stable id of one leaderboard cell (task x budget x condition), used by `robocurate reproduce`."""
    import hashlib

    return hashlib.sha256(f"{task}|{float(budget)}|{label}".encode()).hexdigest()[:12]


def _cond_summary(task, budget, label, rs):
    x = np.array([r["success_rate"] for r in rs])
    lo, hi = run_level_ci(x)
    cost = [r.get("extra", {}).get("curation_gpu_h") for r in rs]
    cost = next((c for c in cost if c is not None), None)
    # the cell id uses the runs' own budget (full data is 1.0), so a cell has one id across panels/tracks
    return {"condition": label, "cell_id": cell_id(task, rs[0]["spec"]["budget"], label), "n_runs": int(len(x)),
            "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else None,
            "ci95_run_level": [lo, hi], "run_rates": [float(v) for v in x], "curation_gpu_h": cost,
            "run_ids": [r["run_id"] for r in rs]}


def _by_subset(rs) -> list[list[float]]:
    g: dict[int, list[float]] = defaultdict(list)
    for r in rs:
        g[int(r["spec"]["subset_index"])].append(r["success_rate"])
    return [g[k] for k in sorted(g)]


def _comparisons(entrants: dict[str, list[float]], placebo: list[list[float]],
                 refs: dict[str, tuple[list[float], bool]],
                 alpha: float, n_boot: int, track: str, costs: dict[str, Any]) -> tuple[list, list, dict | None]:
    """D-026 primary test (Holm across `entrants`), references outside the family, MDE.

    References: a selection reference (oracle, own subset variance) uses the D-026 test too; full data
    (not a draw from the subset distribution) uses the variance-components SE without the own-subset term.
    """
    others = [v for v, _ in refs.values()]
    comps_raw, vc = S.compare_entrants_d026(entrants, placebo, others, alpha=alpha, n_boot=n_boot)
    comps = []
    for c in comps_raw:
        d = asdict(c)
        d["boot_ci"] = list(c.boot_ci)
        d["track"] = track
        d["curation_gpu_h"] = costs.get(c.name)
        comps.append(d)
    pm = float(np.mean([np.mean(g) for g in placebo]))
    out_refs = []
    for name, (runs, own) in refs.items():
        if own:
            t = S.balanced_test(runs, placebo)
            test = "d026"
        else:
            t = S.subset_test(runs, vc, pm, own_subset_var=False)
            test = "d016 variance components, no own-subset term (full data is not a subset draw)"
        out_refs.append({"condition": name, "diff": t.diff, "se": t.se, "t": t.t, "df": t.df, "p": t.p,
                         "significant": bool(t.p <= alpha), "test": test,
                         "note": "reference condition; not in the Holm family"})
    n_p = int(round(np.mean([len(g) for g in placebo])))
    n_c = max([len(v) for v in entrants.values()] or [n_p])
    m = max(1, len(entrants))
    tau = float(np.std([np.mean(g) for g in placebo], ddof=1))  # SD of subset means (the D-026 SE input)
    mde = {"tau": tau, "s_sub": vc.s_sub, "s_seed": vc.s_seed, "s_sub_truncated": vc.truncated, "K": vc.K,
           "n_p": n_p, "n_c": n_c, "n_comparisons": m, "power": 0.8, "df": vc.K - 1, "test": "d026",
           "mde": S.balanced_mde(tau, 0.0, max(1, n_p), vc.K, 1, alpha, 0.8, m)}
    return comps, out_refs, mde


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

    out: dict[str, Any] = {"kind": kind, "fake": kind == "fake", "alpha": alpha, "tasks": [],
                           "primary_test": "D-026: t = (mean_c - mean_p) / sqrt((1 + 1/K) var(subset means)), "
                                           "df = K-1, balanced n_c = n_p = n >= 3, K >= 8, Holm across entrants",
                           "secondary": "D-016/D-019 variance-components test and Welch vs pooled placebo runs "
                                        "(diagnostics only)"}
    for task, trs in sorted(by_task.items()):
        kind_of = lambda r: r["spec"]["condition"]["kind"]  # noqa: E731
        full = [r for r in trs if kind_of(r) == "full"]
        sel_budgets = sorted({r["spec"]["budget"] for r in trs if kind_of(r) in ("random", "cleaner", "oracle")})
        tinfo: dict[str, Any] = {"task": task, "issues": _consistency(trs), "budgets": [], "weighting": []}
        for b in sel_budgets:
            at = [r for r in trs if r["spec"]["budget"] == b and kind_of(r) in ("random", "cleaner", "oracle")]
            groups: dict[str, list[dict]] = defaultdict(list)
            for r in at:
                groups[_label(r["spec"]["condition"])].append(r)
            if full:
                groups["full"] = full
            n_subsets = len({r["spec"]["subset_index"] for r in groups.get("random", [])})
            issues = []
            if n_subsets < 3:
                issues.append(f"placebo has {n_subsets} random subsets at budget {b}; need >= 3 (rule 1)")
            if not full:
                issues.append("no full-data runs (rule 1)")
            if kind == "result" and n_subsets < 8:
                issues.append(f"K = {n_subsets} < 8 random subsets: not result-grade (D-019, D-026)")
            order = ["full", "random", "oracle"]
            conditions = [_cond_summary(task, b, k, groups[k]) for k in
                          sorted(groups, key=lambda s: (order.index(s) if s in order else 3, s))]
            comps, refs, mde_v = [], [], None
            if n_subsets >= 2:
                rate = {k: [r["success_rate"] for r in v] for k, v in groups.items()}
                cleaners = {k: v for k, v in rate.items() if k.startswith("cleaner:")}
                refsin = {k: (rate[k], k != "full") for k in ("oracle", "full") if k in rate}
                costs = {c["condition"]: c["curation_gpu_h"] for c in conditions}
                try:
                    comps, refs, mde_v = _comparisons(cleaners, _by_subset(groups["random"]), refsin, alpha, n_boot,
                                                      "selection", costs)
                except ValueError as e:
                    issues.append(f"primary test not computable: {e}")
            tinfo["budgets"].append({"budget": b, "issues": issues, "conditions": conditions,
                                     "cleaner_vs_placebo": comps, "reference_vs_placebo": refs, "mde": mde_v})
        # weighting track (D-020): entrant vs its own permuted-weights placebo; unweighted full data as reference
        wnames = sorted({r["spec"]["condition"]["name"] for r in trs if kind_of(r) == "weighting"})
        if wnames:
            ent, plac, issues, conds = {}, {}, [], []
            for n in wnames:
                w = [r for r in trs if kind_of(r) == "weighting" and r["spec"]["condition"]["name"] == n]
                wp = [r for r in trs if kind_of(r) == "weight_placebo" and r["spec"]["condition"]["name"] == n]
                ent[f"weighting:{n}"] = [r["success_rate"] for r in w]
                plac[n] = _by_subset(wp)
                conds += [_cond_summary(task, 1.0, f"weighting:{n}", w),
                          _cond_summary(task, 1.0, f"weight_placebo:{n}", wp)]
                if len(plac[n]) < 8:
                    issues.append(f"{n}: {len(plac[n])} weight permutations; D-020 requires >= 8")
            if full:
                conds.insert(0, _cond_summary(task, 1.0, "full", full))
            comps, refs, mde_v = [], [], None
            # each entrant has its own placebo; Holm across weighting entrants on the per-entrant p-values
            per = []
            for n in wnames:
                if len(plac[n]) >= 2:
                    c, r, m = _comparisons({f"weighting:{n}": ent[f"weighting:{n}"]}, plac[n],
                                           {"full": ([x["success_rate"] for x in full], False)} if full else {},
                                           alpha, n_boot, "weighting",
                                           {c["condition"]: c["curation_gpu_h"] for c in conds})
                    per.append((c[0], r, m))
            if per:
                p_adj = S.holm_adjust([c["p_raw"] for c, _, _ in per])
                for (c, r, m), pa in zip(per, p_adj, strict=True):
                    c["p_holm"], c["significant"] = float(pa), bool(pa <= alpha)
                    comps.append(c)
                    refs += [{**x, "vs": c["name"]} for x in r]
                    mde_v = m
            tinfo["weighting"].append({"budget": 1.0, "issues": issues, "conditions": conds,
                                       "cleaner_vs_placebo": comps, "reference_vs_placebo": refs, "mde": mde_v})
        out["tasks"].append(tinfo)
    return out


def _fmt(x: float | None, nd=3) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _panels(a: dict[str, Any]) -> list[tuple[str, str, dict]]:
    out = []
    for t in a["tasks"]:
        out += [(t["task"], f"budget {b['budget']} (selection)", b) for b in t["budgets"]]
        out += [(t["task"], "full data (weighting)", w) for w in t.get("weighting", [])]
    return out


def render_markdown(a: dict[str, Any], plot_name: str | None, n_files: int) -> str:
    L: list[str] = []
    if a["fake"]:
        L += [FAKE_BANNER, "", "**These numbers come from the fake backend with injected effect sizes. They "
              "verify the pipeline; they say nothing about any cleaner.**", ""]
    L += [f"# RoboCurate-Bench report ({a['kind']})", "",
          f"Results files: {n_files}. Unit of analysis: training run (D-002). Primary test (D-026): each "
          "entrant vs its placebo (K random subsets, or K weight permutations for the weighting track, D-020), "
          "exact balanced prediction t-test, SE^2 = (1 + 1/K) var(subset means), df = K - 1, valid for n_c = n_p "
          f"= n >= 3 and K >= 8; Holm across entrants of a track; alpha = {a['alpha']}. Oracle and full data are "
          "references outside the Holm family. The D-016/D-019 variance-components test and Welch vs pooled "
          "placebo runs are SECONDARY diagnostics only.", ""]
    for task, title, b in _panels(a):
        L += [f"## {task} - {title}", ""]
        L += [f"- ⚠️ {i}" for i in b["issues"]]
        L += ["| condition | cell | runs | mean success | SD | run-level 95% CI | curation GPU-h |",
              "|---|---|---|---|---|---|---|"]
        for c in b["conditions"]:
            lo, hi = c["ci95_run_level"]
            L.append(f"| {c['condition']} | `{c['cell_id']}` | {c['n_runs']} | {_fmt(c['mean'])} | {_fmt(c['sd'])} | "
                     f"[{_fmt(lo)}, {_fmt(hi)}] | {_fmt(c['curation_gpu_h'], 2)} |")
        L.append("")
        if b["cleaner_vs_placebo"]:
            L += ["**Entrants vs placebo (primary, D-026; Holm across entrants)**", "",
                  "| entrant | diff | SE | t | df | p (raw) | p (Holm) | significant | D-016 p (Holm, secondary) "
                  "| Welch p (Holm, secondary) |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            for c in b["cleaner_vs_placebo"]:
                w = c.get("welch_secondary") or {}
                v = c.get("vc_secondary") or {}
                L.append(f"| {c['name']} | {c['diff']:+.3f} | {c['se']:.3f} | {c['t']:.2f} | {c['df']:.0f} | "
                         f"{c['p_raw']:.2g} | {c['p_holm']:.2g} | {'YES' if c['significant'] else 'no'} | "
                         f"{_fmt(v.get('p_holm'), 3)} | {_fmt(w.get('p_holm'), 3)} |")
            L.append("")
        if b["reference_vs_placebo"]:
            L += ["**Reference conditions vs placebo (not entrants, unadjusted)**", "",
                  "| condition | diff | SE | t | df | p | significant |", "|---|---|---|---|---|---|---|"]
            for c in b["reference_vs_placebo"]:
                L.append(f"| {c['condition']} | {c['diff']:+.3f} | {c['se']:.3f} | {c['t']:.2f} | {c['df']:.0f} | "
                         f"{c['p']:.2g} | {'YES' if c['significant'] else 'no'} |")
            L.append("")
        if b["mde"]:
            m = b["mde"]
            L += [f"**MDE and variance components**: SD of placebo subset means = {m['tau']:.3f} (the D-026 SE "
                  f"input); diagnostics s_sub = {m['s_sub']:.3f}"
                  f"{' (truncated at 0)' if m['s_sub_truncated'] else ''}, s_seed = {m['s_seed']:.3f}; design "
                  f"K = {m['K']}, n_p = {m['n_p']}, n_c = {m['n_c']}, df = {m['df']}; MDE at 80% power with "
                  f"Bonferroni over {m['n_comparisons']} entrant(s): **{m['mde']:.3f}** ({100 * m['mde']:.1f} pp).", ""]
    if plot_name:
        L += [f"![per-run success by condition]({plot_name})", ""]
    if a["fake"]:
        L += ["", FAKE_BANNER]
    return "\n".join(L) + "\n"


def render_plot(a: dict[str, Any], path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [(task, title, b) for task, title, b in _panels(a)]
    fig, axes = plt.subplots(1, len(panels), figsize=(max(7.5, 7.5 * len(panels)), 4.6), squeeze=False,
                             facecolor=SURFACE)
    for ax, (task, title, b) in zip(axes[0], panels, strict=True):
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
        ax.set_xticks(range(len(conds)), [c["condition"].replace("cleaner:", "").replace("weight_placebo:",
                      "perm:").replace("weighting:", "") for c in conds], color=INK2, fontsize=9)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("success rate (final checkpoint)", color=INK2, fontsize=9)
        ax.tick_params(axis="y", colors=MUTED, labelsize=8)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.set_title(f"Per-run success by condition — {task}, {title}\n"
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
