import os

import pytest

# CPU-only guard for the whole test session (task rule: no GPU use).
os.environ["CUDA_VISIBLE_DEVICES"] = ""


@pytest.fixture
def scratch_env(tmp_path, monkeypatch):
    """A valid storage environment pointing at a pytest tmp dir (outside the repo)."""
    root = tmp_path / "scratch"
    root.mkdir()
    monkeypatch.setenv("ROBOCURATE_SCRATCH", str(root))
    monkeypatch.setenv("HF_HOME", str(root / "hf"))
    monkeypatch.setenv("HF_LEROBOT_HOME", str(root / "lerobot"))
    return root
