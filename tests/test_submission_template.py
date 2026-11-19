"""Item 9: the example template installs as a plugin package and passes `robocurate submit`."""

import os
import subprocess
import sys
import venv

import pytest

from robocurate.paths import repo_root


@pytest.mark.slow
def test_template_installs_and_passes_submit(tmp_path):
    env_dir = tmp_path / "venv"
    venv.EnvBuilder(system_site_packages=True, with_pip=True).create(env_dir)  # sees robocurate + numpy
    py = env_dir / "bin" / "python"
    tpl = repo_root() / "examples/cleaner_template"
    env = {**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
    r = subprocess.run([str(py), "-m", "pip", "install", "--no-deps", "--no-build-isolation", "--no-index", "-q",
                        "-e", str(tpl)], capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr
    out = tmp_path / "sel.json"
    r = subprocess.run([str(py), "-m", "robocurate.cli", "submit", "template_smoothness", "--out", str(out)],
                       capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "RESULT: PASS" in r.stdout and out.exists()
    assert sys.prefix != str(env_dir)  # the test env itself was not modified
