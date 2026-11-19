# PROGRESS

**Current milestone: M0 — software core (BLOCKED at Phase 2: null false-positive calibration test fails; see "## Blocked")**

## Blocked

### B-001 (2026-11-19) — Welch + Holm (D-002) fails the null family-wise false-positive test at 3 seeds

**Stop condition hit:** "The null false-positive test fails. Do not weaken the test; report it."

**Symptom.** `tests/test_stats.py::test_null_fwer_calibration` (task spec: 1000 null grids, 7 cleaners vs
placebo, sigma_seed=0.06, 200 rollouts, 3 seeds, alpha=0.05, Holm) gives a family-wise false-positive rate of
**0.087** (1000 sims) / **0.090** (20 000 sims). The threshold is <= 0.07.

**It is not an implementation bug.**
- `welch_arrays` matches `scipy.stats.ttest_ind(equal_var=False)` exactly (t and p, checked on 2000 rows).
- `holm_adjust` passes known-value tests.
- With pure Gaussian data (no binomial noise, no clipping), 7 cleaners x 3 runs vs 9 placebo runs:
  FWER = 0.089. So the binomial/clipping in the simulator is not the cause.

**Root cause.** Welch's t-test is anti-conservative in the far tail when one group has 3 runs (Welch df ~2-4).
Single-test type-I error, n=3 vs n=9, equal variance, 200 000 sims:

| nominal alpha | Welch actual | Student (pooled) actual |
|---|---|---|
| 0.05 | 0.061 | 0.050 |
| 0.05/7 = 0.0071 (Holm's first step) | **0.0136** | 0.0072 |

Holm's first step runs at alpha/7, exactly where Welch is ~2x too liberal, so the family-wise rate inflates.

**Measured alternatives** (same null simulator; FWER over 20 000 null grids unless noted; power = 20pp effect,
sigma_seed=0.05, one real effect among 7 cleaners, Holm):

| primary test | seeds/condition | null FWER | power (20pp) |
|---|---|---|---|
| Welch + Holm (D-002, current) | 3 | **0.090** FAIL | — |
| Welch + Holm | 4 | 0.083 FAIL | — |
| Welch + Holm | 5 | 0.069 (marginal pass) | 0.94 (5 000 sims) |
| Welch + Holm | 6 | 0.060 | — |
| Student pooled t + Holm | 3 | 0.042 | 0.90 (2 000 sims) |
| Student pooled t + Holm | 5 | 0.044 | — |
| Exact permutation (diff of means, C(12,3)=220 splits) + Holm | 3 | 0.030 (2 000 sims) | 0.79 (2 000 sims) |

**Hypotheses / options (a protocol decision, D-002 — not mine to make):**
1. Keep Welch, raise the seed floor to >= 6 per condition (D-005 power analysis would likely want more anyway).
   Cost: 2x training runs vs 3 seeds.
2. Switch the primary test to Student's pooled t. Valid here when cleaner and placebo run-level variances are
   similar; risk: the placebo pool also carries subset-to-subset variance, so variances may differ in reality
   (Student is conservative when the larger group has the larger variance, liberal in the reverse case).
3. Switch to an exact permutation test over runs. Exact under exchangeability; with 3 vs 9 runs the minimum p
   is 1/220 = 0.0045 < 0.05/7, so Holm can still reject. Lower power at 3 seeds.
4. Keep Welch as primary but require the bootstrap CI or a permutation p to agree (two-test gate).

**What would resolve it:** a decision on D-002 (test choice) and/or D-005 (seed floor). The test threshold
(<= 0.07) and scenario are unchanged; I did not weaken them.

**Reproduce:** `pytest -q tests/test_stats.py::test_null_fwer_calibration` (test file is in the working tree,
uncommitted — see phase log).

## Phase log

### 2026-11-19T03:55Z — Phase 0: bootstrap (done)
- `ee06c92` docs: add CLAUDE.md and decision log (main, pushed to origin/main).
- `79236d1` docs: storage rule for Mac dev, decisions D-011..D-014 (build-core, pushed).
- GPU-hours: 0.

### 2026-11-19T03:55Z — Phase 1: environment and foundations (done)
- Machine: Apple M5 Pro Mac (hostname `Viveks-Macbook-M5-Pro.local`), no NVIDIA GPU, `nvidia-smi` absent.
- `pip index versions lerobot` -> latest 0.6.x is **0.6.1** (0.7.0 exists; not used, D-001 pins 0.6.x).
  PyPI metadata: Requires-Python `>=3.12`. Extra providing the ALOHA sim env: **`aloha`**
  (`gym-aloha>=0.1.4,<0.2.0` + `lerobot[dataset]` + `lerobot[scipy-dep]`).
- `conda create -n rcb python=3.12`; `pip install "lerobot[aloha,training,evaluation]==0.6.1" ...`.
  `training`/`evaluation` extras added so `lerobot-train --help` / `lerobot-eval --help` run (flag-existence test).
- `PYTHONNOUSERSITE=1` is required: `~/.local/lib/python3.12/site-packages` leaks into the env otherwise.
  `requirements.lock` was produced with it set (`pip freeze --exclude-editable`, 103 lines).
- Files: `pyproject.toml`, `.gitignore`, `scripts/env.example.sh`, `src/robocurate/{paths,config,results}.py`,
  `src/robocurate/report/stats.py` (needed by `results.validate` for the Wilson CI check), `tests/conftest.py`,
  `tests/test_foundations.py`.
- Tests: `pytest -q tests/test_foundations.py` -> 31 passed. `ruff check .` clean.
- GPU-hours: 0.

### 2026-11-19T03:55Z — Phase 2: statistics (BLOCKED)
- `src/robocurate/report/stats.py`: Wilson CI, vectorized Welch, Holm, run-level bootstrap CI, MDE/power,
  `compare_cleaners_to_placebo`. Committed with Phase 1 because `results.py` uses `wilson_ci`.
- `tests/test_stats.py`: 16 tests. All known-value tests pass. `test_power_20pp_5_seeds` passes (0.943).
  **`test_null_fwer_calibration` FAILS (0.087).** Left uncommitted, because CLAUDE.md requires tests to pass
  before a commit and the prompt forbids weakening the test.
- Phases 3-8 not started.
- GPU-hours: 0.

## How GPU use was prevented
This Mac has no NVIDIA GPU. Every shell command ran with `CUDA_VISIBLE_DEVICES=""`. `tests/conftest.py` sets it
for the test session. The only LeRobot processes run were `lerobot-train --help` and `lerobot-eval --help`.

## LeRobot touchpoints (lerobot 0.6.1, site-packages/lerobot/)
Collected while reading the source; used by Phases 5-6 (not yet implemented).
- Episode subset option: `DatasetConfig.episodes: list[int] | None` — `configs/default.py:38`; passed to
  `LeRobotDataset(episodes=...)` in `datasets/factory.py:101,114`.
- **Normalization stats with `--dataset.episodes`:** `LeRobotDatasetMetadata._load_metadata` loads the
  FULL dataset's `meta/stats.json` (`datasets/dataset_metadata.py:219`), and `lerobot-train` passes
  `dataset.meta.stats` to the policy processors (`scripts/lerobot_train.py:351,376-377`). The episodes option
  therefore uses full-dataset stats, which **violates D-013**.
- **Path that satisfies D-013:** `dataset_tools.split_dataset(dataset, {"train": episodes}, output_dir)`
  (`datasets/dataset_tools.py:175`) materializes the subset and aggregates stats from the kept episodes' own
  per-episode stats (`_copy_and_reindex_episodes_metadata`, `dataset_tools.py:852-948`; aggregation at
  `:941-948`). Image stats for ACT are then replaced by ImageNet stats when `dataset.use_imagenet_stats=true`
  (default, `configs/default.py:41`; applied in `datasets/factory.py:137-142`), identically for every condition.
- Dataset writing API: `LeRobotDataset.create(repo_id, fps, features, root, robot_type, use_videos, ...)`
  (`datasets/lerobot_dataset.py:684`), `add_frame` (`:418`, frame must include `task`, must not include
  `timestamp`/`frame_index` — `datasets/dataset_writer.py:209-211`), `save_episode` (`:434`), `finalize` (`:474`).
- `HF_LEROBOT_HOME` env var read at import: `utils/constants.py:69`.
- Codebase version `v3.0` (`datasets/dataset_metadata.py:60`); Hub datasets `lerobot/aloha_sim_transfer_cube_human`
  and `lerobot/aloha_sim_transfer_cube_scripted` both exist and carry a `v3.0` tag (checked via the Hub API).
- Train config fields: `seed` (`configs/train.py:96`), `steps` (`:111`), `env_eval_freq` (`:113`),
  `save_freq` (`:123`), `batch_size` (`:102`), `output_dir` (`:86`), `wandb.enable` (`configs/default.py:82`),
  `policy.push_to_hub` default True (`configs/policies.py:70`, so it must be set false), `policy.device` (`:62`).
- Checkpoint layout: `output_dir/checkpoints/{step:0{max(6,len(str(total_steps)))}d}/pretrained_model`
  (`common/train_utils.py:44-52`, `utils/constants.py:47-49`).
- Eval: `--policy.path` via `EvalPipelineConfig.__get_path_fields__` (`configs/eval.py:77`). Without it the policy
  is built with random weights, with only a warning (`configs/eval.py:55-58`).
  `--env.type=aloha`, `--env.task` (AlohaEnv, `envs/configs.py:147-152`; gym id `gym_aloha/AlohaTransferCube-v0`).
- **Eval seed derivation:** `eval_main` passes `start_seed=cfg.seed` (`scripts/lerobot_eval.py:801`). Batch
  `batch_ix` uses seeds `range(start_seed + batch_ix*num_envs, start_seed + (batch_ix+1)*num_envs)`
  (`:528-533`); `num_envs = eval.batch_size` (`make_env(n_envs=cfg.eval.batch_size)`, `:752-757`). So episode i
  gets env seed `--seed + i`, independent of batch size. The last partial batch runs up to `batch_size-1` extra
  episodes (seeds beyond `seed + n_episodes - 1`), which are truncated by `[:n_episodes]` (`:662-667`).
- **Per-episode output:** `eval_info.json` (`scripts/lerobot_eval.py:819-820`) =
  `{"per_task": [{"task_group", "task_id", "metrics": {"sum_rewards", "max_rewards", "successes",
  "video_paths", "predicted_video_paths"}}], "per_group": {...}, "overall": {...}}` (`:1053-1107`). Per-episode
  seeds are **not** written (dropped in `eval_one`, `:877-884`).
- **No "video off" flag in 0.6.1:** `max_episodes_rendered = 0 if cfg.eval.recording else 10`
  (`scripts/lerobot_eval.py:785`). Up to 10 videos are always rendered into `output_dir/videos`.

## Open questions
1. B-001: which primary test / seed floor (D-002, D-005)?
2. lerobot-eval always renders up to 10 videos (no flag). Accept (videos go under scratch), or record a
   decision to patch around it?

## Parking lot
- (none)
