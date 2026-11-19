# RoboCurate-Bench

Does cleaning robot demonstration data actually help? RoboCurate-Bench tests whether robot-data "cleaners"
(episode filters) improve imitation-learning policy success **compared with size-matched random subsets of the
same data** (the placebo), in simulation.

**Status: software core only. No results yet.** Next milestone: GPU calibration (M0).

## Protocol in brief

- Every cleaner is compared with >= 3 size-matched random subsets and with full data, at the same episode
  budget (launch budget: keep 50%).
- Identical training config for every condition: same step count and hyperparameters (LeRobot, pinned 0.6.1).
- Only the final checkpoint at a fixed step count is evaluated, with >= 200 rollouts on fixed eval seeds.
- Unit of analysis = training run. Primary test: Welch t-test of a cleaner's runs vs all placebo runs, with Holm
  correction across cleaners. At least 6 training seeds per condition (D-015).
- A label oracle (uses hidden ground-truth labels) is reported as a reference, not an entrant.
- Cleaners never see labels, eval code, eval seeds or eval results; `robocurate submit` checks this.
- Normalization stats come from each condition's own training subset.

Full rules: [CLAUDE.md](CLAUDE.md). Decisions: [DECISIONS.md](DECISIONS.md). Status: [PROGRESS.md](PROGRESS.md).

## Quick start (CPU)

```bash
conda create -n rcb python=3.12 && conda activate rcb
pip install -e ".[dev]"
cp scripts/env.example.sh scripts/env.sh && source scripts/env.sh
robocurate plan configs/grids/task_a_v0.yaml
robocurate run configs/grids/task_a_v0.yaml --backend fake --fake-params configs/fake/e2e_test.yaml
robocurate report "$ROBOCURATE_SCRATCH/results/task_a_v0-fake"
robocurate submit tests/fixtures/cleaners/good_cleaner.py
```

The fake backend produces synthetic results with known effect sizes, to verify the pipeline. Its reports are
labeled FAKE.
