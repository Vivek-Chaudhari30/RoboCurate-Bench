"""CI driver for cleaner submissions (.github/workflows/submit.yml); also runnable locally.

Finds the cleaner packages touched by a change set: a directory with a pyproject.toml that declares entry points in
the `robocurate.cleaners` group (the benchmark's own pyproject at the repo root is never one). For each package it
optionally `pip install -e --no-deps` it, runs `robocurate submit <entry-point name>` (CPU conformance on a
synthetic dataset), and writes a markdown table to the job summary. Exit 1 if any submission fails.

  python scripts/ci_submit.py --base origin/main --install --summary "$GITHUB_STEP_SUMMARY"
  python scripts/ci_submit.py --changed examples/cleaner_template/src/rcb_template_cleaner/cleaner.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GROUP = "robocurate.cleaners"


def entry_points(pkg: Path) -> dict[str, str]:
    data = tomllib.loads((pkg / "pyproject.toml").read_text())
    return dict(data.get("project", {}).get("entry-points", {}).get(GROUP, {}))


def package_roots(changed: list[str], repo: Path = REPO) -> list[Path]:
    roots: set[Path] = set()
    for f in changed:
        p = (repo / f).resolve()
        for d in [p, *p.parents]:
            if d == repo or repo not in d.parents:
                break
            if (d / "pyproject.toml").is_file() and entry_points(d):
                roots.add(d)
                break
    return sorted(roots)


def changed_files(base: str, repo: Path = REPO) -> list[str]:
    out = subprocess.run(["git", "-C", str(repo), "diff", "--name-only", f"{base}...HEAD"], capture_output=True,
                         text=True, check=True).stdout
    return [x for x in out.splitlines() if x.strip()]


def run(pkgs: list[Path], python: str, install: bool) -> list[dict]:
    rows = []
    for pkg in pkgs:
        rel = pkg.relative_to(REPO) if REPO in pkg.parents else pkg
        if install:
            r = subprocess.run([python, "-m", "pip", "install", "-q", "--no-deps", "--no-build-isolation", "-e",
                                str(pkg)], capture_output=True, text=True)
            if r.returncode != 0:
                rows.append({"package": str(rel), "name": "-", "passed": False, "output": r.stderr[-3000:]})
                continue
        for name in entry_points(pkg):
            with tempfile.TemporaryDirectory() as td:
                r = subprocess.run([python, "-m", "robocurate.cli", "submit", name, "--out", f"{td}/sel.json"],
                                   capture_output=True, text=True)
            out = (r.stdout + r.stderr)[-6000:]
            rows.append({"package": str(rel), "name": name, "passed": r.returncode == 0 and "RESULT: PASS" in out,
                         "output": out})
    return rows


def summary(rows: list[dict]) -> str:
    if not rows:
        return "## Cleaner submission conformance\n\nNo cleaner package changed in this pull request.\n"
    L = ["## Cleaner submission conformance", "", "| package | cleaner | result |", "|---|---|---|"]
    L += [f"| `{r['package']}` | `{r['name']}` | {'PASS' if r['passed'] else '**FAIL**'} |" for r in rows]
    for r in rows:
        L += ["", f"<details><summary>{r['name']} ({r['package']})</summary>", "", "```text", r["output"].strip(),
              "```", "", "</details>"]
    L += ["", "Conformance checks the contract (static import rules, determinism, runtime, no-leak) on a synthetic "
              "dataset. It says nothing about the method's quality (docs/submit.md)."]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--base", help="git ref to diff against (base...HEAD)")
    g.add_argument("--changed", nargs="+", help="explicit changed paths (repo-relative)")
    ap.add_argument("--summary", help="markdown output (the job summary file in CI)")
    ap.add_argument("--install", action="store_true", help="pip install -e each package (--no-deps)")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args(argv)
    files = a.changed if a.changed else changed_files(a.base)
    rows = run(package_roots(files), a.python, a.install)
    md = summary(rows)
    if a.summary:
        with open(a.summary, "a") as f:
            f.write(md)
    print(md)
    return 0 if all(r["passed"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
