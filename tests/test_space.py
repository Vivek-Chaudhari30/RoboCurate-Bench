"""space/ as a Hugging Face static Space, and `robocurate report --site space/` (no deployment)."""

import shutil

import pytest

from robocurate.paths import repo_root
from robocurate.report import site as SITE

REPO = repo_root()


def test_space_front_matter_uses_documented_keys_only():
    fm = SITE.space_front_matter(REPO / "space")
    assert fm is not None and fm["sdk"] == "static" and fm["app_file"] == "index.html"
    SITE.check_space(fm)
    assert (REPO / "space/index.html").exists()
    with pytest.raises(SITE.SiteError, match="undocumented"):
        SITE.check_space({**fm, "licence": "x"})
    with pytest.raises(SITE.SiteError, match="static"):
        SITE.check_space({**fm, "sdk": "gradio"})
    with pytest.raises(SITE.SiteError, match="colorTo"):
        SITE.check_space({**fm, "colorTo": "orange"})


def test_report_site_into_space_keeps_readme_and_refuses_fake(tmp_path, scratch_env, capsys):
    from robocurate.cli import main

    res = tmp_path / "res"
    assert main(["run", str(REPO / "configs/calibration/m0_act_transfer_cube_human.yaml"), "--calibration",
                 "--backend", "fake", "--fake-params", str(REPO / "configs/fake/m0_dryrun.yaml"),
                 "--results-dir", str(res)]) == 0
    space = tmp_path / "space"
    shutil.copytree(REPO / "space", space)
    readme = (space / "README.md").read_text()
    with pytest.raises(SITE.SiteError, match="FAKE"):
        main(["report", str(res), "--site", str(space)])
    assert main(["report", str(res), "--site", str(space), "--allow-fake-site"]) == 0  # local preview only
    assert (space / "README.md").read_text() == readme
    assert "FAKE" in (space / "index.html").read_text() and (space / "leaderboard.json").exists()


def test_sdist_manifest_prunes_everything_but_the_package():
    lines = (REPO / "MANIFEST.in").read_text().splitlines()
    assert {"prune tests", "prune configs", "prune scripts", "prune space"} <= set(lines)
