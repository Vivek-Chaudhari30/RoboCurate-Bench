# CLAUDE.md — RoboCurate-Bench

Read fully before acting. These rules override your defaults. If a task prompt conflicts with this file, stop and ask.

## What this repo is
RoboCurate-Bench tests whether robot-data "cleaners" (episode filters) improve imitation-learning policy success versus size-matched RANDOM subsets (the placebo), in simulation. The protocol is the product. A fast wrong number is worse than no number.

## Protocol rules (non-negotiable)
1. Every cleaner is compared to >=3 size-matched random subsets AND full data, at the same episode budget.
2. Seed count per condition comes from configs/protocol.yaml (floor 3). >=200 eval rollouts per evaluated checkpoint.
3. Results use only the checkpoint protocol.yaml names (final checkpoint at fixed step count). Never select a "best" checkpoint.
4. Identical training config across conditions: same steps, same hyperparameters. No per-method tweaks.
5. Unit of statistical analysis = training run. Never treat rollouts pooled across runs as independent samples.
6. Eval seeds come only from configs/protocol.yaml. Cleaners that roll out policies use curation_seed_range (disjoint) and their declared rollout cap.
7. Cleaners never receive ground-truth labels (*.labels.json), eval code, eval seeds, or eval results.
8. Every results JSON records: git hash (clean tree), config path + sha256, lerobot version, exact command lines, all seeds, wall-clock, peak GPU memory, hostname.
9. Never tune anything after seeing entrant results. Negative results are reported exactly like positive ones.

## Frozen paths
- src/robocurate/eval/, configs/protocol.yaml and configs/frozen/ are FROZEN once src/robocurate/eval/FROZEN exists.
- Never create, edit or delete FROZEN. Never edit frozen paths. If you think frozen code has a bug, stop, write it under "## Blocked" in PROGRESS.md, and report. Do not work around it.
- Before FROZEN exists, edit eval/ only if your task prompt explicitly scopes it.

## Libraries
- Build on LeRobot (exact pin in pyproject.toml). Never reimplement training, eval, datasets, policies or reward models it provides.
- Before using ANY LeRobot API, CLI flag or config field, read the installed source or the `--help` output:
  python -c "import lerobot, os; print(os.path.dirname(lerobot.__file__))"
  Never guess or invent flags. In your report, cite the file:line you relied on.
- No new or upgraded dependencies unless the task allows it. If allowed: pin exactly and regenerate requirements.lock.

## GPU etiquette (one shared RTX A6000 on hokkaido)
- Before ANY GPU work, run nvidia-smi. If a process you don't own uses >2 GB or >10% utilization, do not start. Wait (poll every 10 min, within the task's limit) or exit and report.
- Anything longer than 10 minutes runs inside tmux.
- Kill your own idle or finished processes. Never touch other users' processes.
- Stay within the task's GPU-hour budget. Log GPU-hours used in PROGRESS.md.
- Use HF Jobs only if the task prompt authorizes it with a dollar cap. Estimate cost first.

## Storage
- Nothing large goes in $HOME or the repo. Datasets, checkpoints, caches, videos and logs go under $ROBOCURATE_SCRATCH.
- Required env vars (verify LeRobot's names in source): ROBOCURATE_SCRATCH, HF_HOME=$ROBOCURATE_SCRATCH/hf, HF_LEROBOT_HOME=$ROBOCURATE_SCRATCH/lerobot, MUJOCO_GL=egl.
- Code must refuse to run if ROBOCURATE_SCRATCH is unset or inside $HOME.
- Small results JSONs are committed under results/. Checkpoints never are.

## Git and logging
- Work on the branch your task prompt names. Never commit directly to main.
- Commit after every working step, with message format "<area>: <what>". Tests must pass first.
- Forbidden: force push, reset --hard, git clean, rebasing pushed commits, deleting branches, rm -rf outside your own run dirs under $ROBOCURATE_SCRATCH.
- After each step, append to PROGRESS.md: UTC timestamp, what changed, commands run, outcome, GPU-hours.
- Do not make protocol decisions. Propose them in your report. Accepted decisions go in DECISIONS.md (D-###: what / why / rejected alternative).

## Tests
- Default: pytest -q -m "not gpu and not slow"  (must pass before every commit)
- GPU and slow tests: pytest -q -m "gpu or slow"  (run when the task says so; check nvidia-smi first)
- ruff check . must be clean.
- Every bug fix gets a regression test.

## Scope
The current milestone is at the top of PROGRESS.md. Build only what your task scopes. No polish, UI, leaderboard, extra tasks, extra cleaners or hardware support unless scoped. Put ideas under "## Parking lot" in PROGRESS.md.

## When stuck
Same failure twice after two different fixes: stop. Write a diagnosis in PROGRESS.md (symptom, attempts, hypotheses, what info would resolve it) and report. Never loop.
