"""Adapters for existing open-source dataset checkers (wrapped, optional extra `checkers`).

Common mapping (m1-core item 5a): each tool's per-episode output becomes a rank class
  0 = clean / keep / PASS,  1 = review / WARN,  2 = flagged / exclude / FAIL
and an optional tool score (lower = better). ScoreCleaner keeps the top-k of
  score = -(1000 * rank_class + min(tool_score, 999))
so flagged episodes rank last, then the tool's own score; remaining ties use D-021 (salted content hash).

Every adapter needs `context.dataset_root` (the LeRobot dataset directory) because the tools read parquet files
themselves. The tools were NOT installed in this session (install into the shared env was not permitted); the
mapping functions are unit-tested on outputs shaped exactly like each tool's documented/inspected structures.

Sources inspected (read-only, published artifacts):
  lerobot-doctor 0.2.0 (PyPI, Apache-2.0): lerobot_doctor/dataset_loader.py load_local; runner.py run_checks;
      checks/per_episode.py: per-episode flags only as WARN messages "Episode {idx}: r1; r2", and ONLY THE
      FIRST 20 flagged episodes in index order (then "...and N more flagged episodes"); no numeric score.
  kinema 0.1.0 (PyPI, Apache-2.0): kinema/dataset.py load_dataset; checks/__init__.py ALL_CHECKS;
      checks/base.py Finding(check_id, severity: Severity IntEnum OK0/INFO1/WARN2/FAIL3/SKIP4, episode_index);
      cli.py _run_checks loop `check.run(ctx, CheckConfig())`.
  roboaudit 0.3.1 (PyPI, MIT): roboaudit/audit.py audit_dataset(root) -> {"dispositions": {ep: exclude|review|
      keep}, "excluded_episodes"/"review_episodes": [{"episode_index", "checks"}]} (audit.py:92-211).
  RDA = robot-data-audit 0.12.5, github.com/liesliy/rda@47c69459e201cdc1f509f84b027c2407ab949e30 (MIT):
      rda/io/lerobot_loader.py load_lerobot_dataset, iter_episodes; rda/audit/dataset_audit.py DatasetAuditor
      .audit_dataset(info, episodes).episodes[idx] -> EpisodeAuditResult(verdict PASS/REVIEW/EXCLUDE/
      UNVERIFIABLE, combined_score / deviation_score, behavior_severity) (episode_audit.py:104-151).
Not adapted: trajlens 0.4.0 (requires pyarrow<25; our lock pins 26.0.1), retake (github.com/TheNandinee/retake,
  would upgrade numpy/pandas/huggingface_hub/pillow and calls external LLM/vision APIs).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from robocurate.cleaners.base import ScoreCleaner
from robocurate.data.episodes import DatasetView, EpisodeData


def combine(rank_class: int, tool_score: float = 0.0) -> float:
    return -(1000.0 * int(rank_class) + float(min(max(tool_score, 0.0), 999.0)))


class CheckerCleaner(ScoreCleaner):
    implementation = "wrapped"
    tool: str = ""

    def select(self, view: DatasetView, budget: int, rng: np.random.Generator) -> list[int]:
        root = getattr(self.context, "dataset_root", None) if self.context is not None else None
        if root is None:
            raise RuntimeError(f"{self.name}: needs context.dataset_root (the LeRobot dataset directory)")
        self._ranks = self.episode_ranks(self.run_tool(Path(root)))
        return super().select(view, budget, rng)

    def score(self, episode: EpisodeData) -> float:
        rc, ts = self._ranks.get(episode.episode_index, (0, 0.0))
        return combine(rc, ts)

    def run_tool(self, root: Path) -> Any:  # pragma: no cover - needs the optional tool installed
        raise NotImplementedError

    @staticmethod
    def episode_ranks(output: Any) -> dict[int, tuple[int, float]]:
        raise NotImplementedError


_EP = re.compile(r"^Episode (\d+): (.*)$")


class LerobotDoctor(CheckerCleaner):
    """lerobot-doctor 0.2.0 per-episode summary. Flagged -> class 2, tool score = number of reasons.
    Limitation: the tool reports at most 20 flagged episodes (lowest indices first); episodes beyond that are
    treated as unflagged, which is an index-order bias inherited from the tool."""

    name, version, tool = "lerobot_doctor", "1", "lerobot-doctor==0.2.0"
    source = "pypi lerobot-doctor 0.2.0 (Apache-2.0), checks/per_episode.py"

    def run_tool(self, root: Path):  # pragma: no cover
        from lerobot_doctor.dataset_loader import load_local
        from lerobot_doctor.runner import run_checks

        rep = run_checks(load_local(root), checks=["per_episode"])
        return [m.message for r in rep.results for m in r.messages]

    @staticmethod
    def episode_ranks(messages: list[str]) -> dict[int, tuple[int, float]]:
        out = {}
        for m in messages:
            g = _EP.match(m)
            if g:
                out[int(g.group(1))] = (2, float(len(g.group(2).split("; "))))
        return out


class Kinema(CheckerCleaner):
    """kinema 0.1.0: per-episode max finding severity (FAIL -> 2, WARN -> 1, INFO/OK -> 0; SKIP ignored);
    tool score = number of WARN-or-worse findings. Dataset-level findings (episode_index None) are ignored."""

    name, version, tool = "kinema", "1", "kinema==0.1.0"
    source = "pypi kinema 0.1.0 (Apache-2.0), checks/base.py Finding/Severity"

    def run_tool(self, root: Path):  # pragma: no cover
        from kinema.checks import ALL_CHECKS, CheckConfig
        from kinema.dataset import load_dataset

        ctx, cfg = load_dataset(str(root)), CheckConfig()
        return [(f.episode_index, int(f.severity)) for c in ALL_CHECKS for f in c.run(ctx, cfg).findings]

    @staticmethod
    def episode_ranks(findings: list[tuple[int | None, int]]) -> dict[int, tuple[int, float]]:
        worst: dict[int, int] = {}
        count: dict[int, int] = {}
        for ep, sev in findings:
            if ep is None or sev >= 4:  # dataset-level or SKIP
                continue
            worst[ep] = max(worst.get(ep, 0), sev)
            count[ep] = count.get(ep, 0) + (sev >= 2)
        return {ep: ({3: 2, 2: 1}.get(s, 0), float(count[ep])) for ep, s in worst.items()}


class RoboAudit(CheckerCleaner):
    """roboaudit 0.3.1 dispositions: exclude -> 2, review -> 1, keep -> 0; tool score = number of distinct
    checks that flagged the episode."""

    name, version, tool = "roboaudit", "1", "roboaudit==0.3.1"
    source = "pypi roboaudit 0.3.1 (MIT), audit.py audit_dataset"

    def run_tool(self, root: Path):  # pragma: no cover
        from roboaudit.audit import audit_dataset

        return audit_dataset(root)

    @staticmethod
    def episode_ranks(report: dict) -> dict[int, tuple[int, float]]:
        checks = {int(e["episode_index"]): len(e.get("checks", []))
                  for k in ("excluded_episodes", "review_episodes") for e in report.get(k, [])}
        cls = {"exclude": 2, "review": 1, "keep": 0}
        return {int(ep): (cls[d], float(checks.get(int(ep), 0))) for ep, d in report["dispositions"].items()}


class RDA(CheckerCleaner):
    """RDA (robot-data-audit 0.12.5): EXCLUDE -> 2, REVIEW -> 1, PASS -> 0, UNVERIFIABLE -> 1 (not inspectable is
    not evidence of quality); tool score = combined_score (deviation from reference; None without a reference
    profile) else behavior_severity."""

    name, version, tool = "rda", "1", "robot-data-audit @ git+https://github.com/liesliy/rda@47c69459e2"
    source = "github.com/liesliy/rda@47c69459e201cdc1f509f84b027c2407ab949e30 (MIT), rda/audit/episode_audit.py"

    def run_tool(self, root: Path):  # pragma: no cover
        from rda.audit.dataset_audit import DatasetAuditor
        from rda.io.lerobot_loader import iter_episodes, load_lerobot_dataset

        res = DatasetAuditor().audit_dataset(load_lerobot_dataset(str(root)), iter_episodes(str(root)))
        return {i: {"verdict": getattr(r.verdict, "value", str(r.verdict)), "combined_score": r.combined_score,
                    "behavior_severity": r.behavior_severity} for i, r in res.episodes.items()}

    @staticmethod
    def episode_ranks(eps: dict[int, dict]) -> dict[int, tuple[int, float]]:
        cls = {"EXCLUDE": 2, "REVIEW": 1, "UNVERIFIABLE": 1, "PASS": 0}
        out = {}
        for i, r in eps.items():
            s = r.get("combined_score")
            out[int(i)] = (cls[str(r["verdict"]).upper()], float(s if s is not None else r.get("behavior_severity", 0)))
        return out
