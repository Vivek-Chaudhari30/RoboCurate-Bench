"""Shared test helpers: write a self-contained protocol/task/grid/fake-params set into a tmp dir."""

from __future__ import annotations

from pathlib import Path

PROTOCOL = """
eval_seed_start: 100000
n_eval_rollouts: 200
checkpoint_rule: final
n_train_seeds: TBD
train_seeds: TBD
random_subset_seeds: [2000, 2001, 2002]
curation_seed_range: [500000, 599999]
alpha: 0.05
test: d026_balanced
"""

TASK = """
name: aloha_transfer_cube
env_type: aloha
env_task: AlohaTransferCube-v0
policy_type: act
source_datasets: [lerobot/aloha_sim_transfer_cube_human]
train_dataset: robocurate/task_a_v0
steps: TBD
per_run_gpu_h: TBD
"""

GRID = """
name: test_grid
task: {task}
protocol: {protocol}
budgets: [0.5]
conditions:
  - {{kind: full}}
  - {{kind: random, n_subsets: 3}}
  - {{kind: cleaner, name: length_heuristic, version: "1"}}
  - {{kind: oracle}}
"""

FAKE = """
base: 0.5
sigma_seed: 0.06
train_seeds: [0, 1, 2, 3, 4, 5]
effects: {{full: 0.0, random: 0.0, "cleaner:length_heuristic": {length}, oracle: {oracle},
          "weighting:rabc": 0.0, "weight_placebo:rabc": 0.0}}
sleep_s: {sleep}
"""


def write_configs(d: Path, *, length=0.0, oracle=0.25, sleep=0.0, task=TASK, protocol=PROTOCOL) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    (d / "protocol.yaml").write_text(protocol)
    (d / "task.yaml").write_text(task)
    (d / "grid.yaml").write_text(GRID.format(task=d / "task.yaml", protocol=d / "protocol.yaml"))
    (d / "fake.yaml").write_text(FAKE.format(length=length, oracle=oracle, sleep=sleep))
    return {k: d / f"{k}.yaml" for k in ("protocol", "task", "grid", "fake")}
