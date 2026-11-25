import os

import pytest

# CPU-only guard for the whole test session (task rule: no GPU use).
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# MuJoCo reads this on first import (scripts/env.example.sh). State-only rollouts do not render.
os.environ.setdefault("MUJOCO_GL", "egl")

# LeRobot resolves HF_LEROBOT_HOME once, at import (lerobot/utils/constants.py:69). Import it here, under the
# session's real storage env, so a first import inside a `scratch_env` test (which monkeypatches HF_LEROBOT_HOME to a
# tmp dir) cannot freeze a tmp path for later tests. Regression: test_load_human_metadata failed after
# test_balanced_k8_design_is_eligible imported lerobot under scratch_env.
import lerobot.utils.constants  # noqa: E402, F401  (lerobot is a core dependency)


@pytest.fixture
def scratch_env(tmp_path, monkeypatch):
    """A valid storage environment pointing at a pytest tmp dir (outside the repo)."""
    root = tmp_path / "scratch"
    root.mkdir()
    monkeypatch.setenv("ROBOCURATE_SCRATCH", str(root))
    monkeypatch.setenv("HF_HOME", str(root / "hf"))
    monkeypatch.setenv("HF_LEROBOT_HOME", str(root / "lerobot"))
    return root
