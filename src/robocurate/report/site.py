"""`robocurate report --site out/`: a static leaderboard, ONE self-contained HTML file plus leaderboard.json.

No framework, no external requests (system font stack, inline CSS and SVG). Per cell: mean success with the
run-level 95% CI (a small inline interval bar), the difference vs placebo with the D-026 test, MDE, curation
cost and the track. FAKE data is watermarked on every view (banner + diagonal watermark + title prefix).
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

from robocurate.report.report import INELIGIBLE, _panels

CSS = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--rule:#e1e0d9;--accent:#2a78d6;--warn:#b5462b;
--chip:#f1f0ec}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121212;--ink:#f2f1ee;--ink2:#c3c2b7;
--muted:#8e8d86;--rule:#2b2b29;--accent:#6aa5f0;--warn:#f08a6a;--chip:#1d1d1b}}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",
Inter,Helvetica,Arial,sans-serif;font-feature-settings:"tnum" 1}
main{max-width:980px;margin:0 auto;padding:40px 16px 80px}
h1{font-size:28px;letter-spacing:-.02em;margin:0 0 6px}h2{font-size:18px;margin:44px 0 4px;letter-spacing:-.01em}
.sub{color:var(--ink2);margin:0 0 24px;max-width:70ch}.meta{color:var(--muted);font-size:13px}
.track{display:inline-block;font-size:12px;padding:1px 8px;border-radius:999px;background:var(--chip);color:var(--ink2);
margin-left:8px;vertical-align:2px}
.scroll{overflow-x:auto;margin:12px 0}table{border-collapse:collapse;width:100%;min-width:640px;font-size:14px}
th{font-weight:500;color:var(--muted);text-align:left;font-size:12px;text-transform:uppercase;letter-spacing:.04em;
border-bottom:1px solid var(--rule);padding:8px 10px}
td{border-bottom:1px solid var(--rule);padding:9px 10px;vertical-align:middle}
td.num{text-align:right;white-space:nowrap}
tr.ref td:first-child{color:var(--ink2)}code{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--muted)}
.sig{color:var(--accent);font-weight:600}.ns{color:var(--muted)}.issue{color:var(--warn);font-size:14px;margin:4px 0}
.inel{border:2px solid var(--warn);color:var(--warn);padding:8px 12px;font-weight:700;margin:8px 0;
letter-spacing:.02em}.inel span{font-weight:400;color:var(--ink2)}
.fake{background:var(--warn);color:#fff;padding:10px 16px;font-weight:600;text-align:center;letter-spacing:.02em}
.wm{position:fixed;inset:0;pointer-events:none;display:flex;align-items:center;justify-content:center;
font-size:22vw;font-weight:800;color:var(--warn);opacity:.07;transform:rotate(-20deg);z-index:0}
main{position:relative;z-index:1}.foot{margin-top:56px;color:var(--muted);font-size:13px;max-width:75ch}
svg.ci{display:block}
"""


def _f(x: Any, nd: int = 3, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _ci_svg(mean: float, lo: float | None, hi: float | None) -> str:
    w, h = 120, 14
    x = lambda v: 4 + max(0.0, min(1.0, v)) * (w - 8)  # noqa: E731
    bar = "" if lo is None or math.isnan(lo) else (
        f'<line x1="{x(lo):.1f}" x2="{x(hi):.1f}" y1="7" y2="7" stroke="var(--ink2)" stroke-width="2" '
        'stroke-linecap="round"/>')
    return (f'<svg class="ci" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" '
            f'aria-label="mean {mean:.3f}"><line x1="4" x2="{w - 4}" y1="7" y2="7" stroke="var(--rule)"/>{bar}'
            f'<circle cx="{x(mean):.1f}" cy="7" r="4" fill="var(--accent)" stroke="var(--bg)" stroke-width="2"/></svg>')


def render_site_html(a: dict[str, Any]) -> str:
    fake = a["fake"]
    title = ("FAKE — " if fake else "") + "RoboCurate-Bench leaderboard"
    E = html.escape
    P: list[str] = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
                    f"content='width=device-width,initial-scale=1'><title>{E(title)}</title><style>{CSS}</style>"
                    "</head><body>"]
    if fake:
        P.append("<div class='fake'>FAKE RESULTS — synthetic backend with injected effects. Not real policy "
                 "performance.</div><div class='wm' aria-hidden='true'>FAKE</div>")
    P.append(f"<main><h1>{E(title)}</h1><p class='sub'>Does cleaning robot data help? Each entrant is compared with "
             "size-matched random subsets (selection track) or with its own weights randomly permuted across "
             "episodes (weighting track). The unit of analysis is the training run. Primary test (D-026, exact for a "
             "balanced design n<sub>c</sub> = n<sub>p</sub> = n ≥ 3, K ≥ 8): t = (mean<sub>entrant</sub> − "
             "mean<sub>placebo</sub>) / √((1 + 1/K)·var(subset means)), df = K − 1, Holm across the entrants of a "
             "track.</p>")
    P.append(f"<p class='meta'>kind: {E(a['kind'])} · α = {a['alpha']} · results: {a.get('n_results', '—')} · "
             f"git {E(str((a.get('git') or {}).get('hash', '—'))[:10])} · generated "
             f"{E(str(a.get('generated_utc', '—'))[:19])} UTC</p>")
    for task, ptitle, b in _panels(a):
        track = "weighting" if "weighting" in ptitle else "selection"
        P.append(f"<h2>{E(task)} · {E(ptitle.split(' (')[0])}<span class='track'>{track} track</span></h2>")
        if not b.get("result_eligible", False):
            why = "; ".join(b.get("eligibility") or ["design not checked"])
            P.append(f"<div class='inel' role='note'>{E(INELIGIBLE)} <span>— {E(why)}</span></div>")
        for i in b["issues"]:
            P.append(f"<p class='issue'>⚠ {E(i)}</p>")
        comp = {c["name"]: c for c in b["cleaner_vs_placebo"]}
        ref = {r["condition"]: r for r in b["reference_vs_placebo"]}
        P.append("<div class='scroll'><table><thead><tr><th>condition</th><th>mean success · 95% CI</th>"
                 "<th></th><th class='num'>runs</th><th class='num'>Δ vs placebo</th><th class='num'>SE</th>"
                 "<th class='num'>p (Holm)</th><th class='num'>curation GPU-h</th><th>cell</th></tr></thead><tbody>")
        for c in b["conditions"]:
            name = c["condition"]
            lo, hi = c["ci95_run_level"]
            cc, rr = comp.get(name), ref.get(name)
            if cc:
                d, se = _f(cc["diff"], sign=True), _f(cc["se"])
                p = f"<span class='{'sig' if cc['significant'] else 'ns'}'>{cc['p_holm']:.3g}</span>"
            elif rr:
                d, se = _f(rr["diff"], sign=True), _f(rr["se"])
                p = f"<span class='ns'>{rr['p']:.3g} (ref, unadj.)</span>"
            else:
                d = se = p = "—"
            cls = " class='ref'" if name in ("full", "oracle", "random") or name.startswith("weight_placebo") else ""
            P.append(f"<tr{cls}><td>{E(name)}</td><td>{_ci_svg(c['mean'], lo, hi)}</td><td class='num'>"
                     f"{_f(c['mean'])} [{_f(lo)}, {_f(hi)}]</td><td class='num'>{c['n_runs']}</td>"
                     f"<td class='num'>{d}</td><td class='num'>{se}</td><td class='num'>{p}</td>"
                     f"<td class='num'>{_f(c['curation_gpu_h'], 2)}</td><td><code>{E(c['cell_id'])}</code></td></tr>")
        P.append("</tbody></table></div>")
        if b["mde"]:
            m = b["mde"]
            trunc = " (truncated at 0)" if m["s_sub_truncated"] else ""
            P.append(f"<p class='meta'>SD of placebo subset means = {m['tau']:.3f} · diagnostics: "
                     f"s<sub>sub</sub> = {m['s_sub']:.3f}{trunc}"
                     f" · s<sub>seed</sub> = {m['s_seed']:.3f} · K = {m['K']}, n<sub>p</sub> = {m['n_p']}, "
                     f"n<sub>c</sub> = {m['n_c']}, df = {m['df']} · MDE (80% power, Bonferroni over "
                     f"{m['n_comparisons']}): <strong>{100 * m['mde']:.1f} pp</strong></p>")
    P.append("<p class='foot'>Oracle (uses hidden labels) and full data are reference conditions outside the Holm "
             "family. The D-016/D-019 variance-components test and Welch vs pooled placebo runs are secondary "
             "diagnostics only (leaderboard.json). "
             "Reproduce any cell with <code>robocurate reproduce &lt;cell&gt;</code>.</p></main></body></html>")
    return "".join(P)


def write_site(a: dict[str, Any], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(render_site_html(a))
    (out / "leaderboard.json").write_text(json.dumps(a, indent=2) + "\n")
    return out / "index.html"
