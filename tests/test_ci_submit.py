"""The cleaner-submission CI path (.github/workflows/submit.yml -> scripts/ci_submit.py), tested locally."""

import importlib.util
import subprocess
import venv

import pytest

from robocurate.paths import repo_root

REPO = repo_root()
_spec = importlib.util.spec_from_file_location("ci_submit", REPO / "scripts/ci_submit.py")
CI = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CI)
TPL = REPO / "examples/cleaner_template"


def test_changed_cleaner_files_map_to_their_package():
    assert CI.package_roots(["examples/cleaner_template/src/rcb_template_cleaner/cleaner.py"]) == [TPL]
    assert CI.package_roots(["examples/cleaner_template/README.md", "src/robocurate/cli.py", "README.md"]) == [TPL]
    assert CI.package_roots(["src/robocurate/cli.py", "pyproject.toml"]) == []  # the benchmark is not a submission
    assert CI.entry_points(TPL) == {"template_smoothness": "rcb_template_cleaner.cleaner:SmoothestEpisodes"}


def test_summary_markdown():
    assert "No cleaner package changed" in CI.summary([])
    md = CI.summary([{"package": "p", "name": "a", "passed": True, "output": "RESULT: PASS"},
                     {"package": "q", "name": "b", "passed": False, "output": "RESULT: FAIL"}])
    assert "| `p` | `a` | PASS |" in md and "| `q` | `b` | **FAIL** |" in md


def test_workflow_and_issue_form_are_wired():
    import yaml

    wf = yaml.safe_load((REPO / ".github/workflows/submit.yml").read_text())
    on = wf[True] if True in wf else wf["on"]  # YAML 1.1 parses the key `on` as True
    assert set(on["pull_request"]["paths"]) == {"submissions/**", "examples/**"}
    assert wf["permissions"] == {"contents": "read"}
    run = wf["jobs"]["conformance"]["steps"][-1]["run"]
    assert "scripts/ci_submit.py" in run and "--install" in run and "GITHUB_STEP_SUMMARY" in run
    form = yaml.safe_load((REPO / ".github/ISSUE_TEMPLATE/cleaner_submission.yml").read_text())
    assert {"name", "description", "body"} <= set(form) and any(b["type"] == "checkboxes" for b in form["body"])


@pytest.mark.slow
def test_ci_submit_end_to_end_on_template(tmp_path):
    env_dir = tmp_path / "venv"
    venv.EnvBuilder(system_site_packages=True, with_pip=True).create(env_dir)
    out = tmp_path / "summary.md"
    r = subprocess.run([str(env_dir / "bin/python"), str(REPO / "scripts/ci_submit.py"), "--changed",
                        "examples/cleaner_template/src/rcb_template_cleaner/cleaner.py", "--install", "--python",
                        str(env_dir / "bin/python"), "--summary", str(out)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "| `examples/cleaner_template` | `template_smoothness` | PASS |" in out.read_text()
