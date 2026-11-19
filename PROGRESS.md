# PROGRESS

**Current milestone: M0 — software core done; next = GPU calibration**

## Blocked
(none open)

- B-001 (2026-11-19) RESOLVED by D-015. Welch + Holm failed the null family-wise false-positive test at 3 seeds
  (FWER 0.090 > 0.07; Welch's tail is ~2x too liberal at alpha/7 with 3 runs). The user chose to keep Welch and
  raise the seed floor to 6: FWER 0.060 (20 000 grids), 0.058-0.065 across base rates 0.2/0.5/0.8.

## Open questions (protocol decisions, not made here)

1. **Subset-to-subset variance breaks the run-level null (most important).** A cleaner's runs all train on ONE
   subset; the placebo pools runs over 3 random subsets. If subsets differ in quality (they will), the cleaner's
   mean carries a subset effect that more seeds cannot average away. Simulated at 6 seeds, 7 cleaners:
   subset SD 0.00 -> FWER 0.060; **0.02 -> 0.151; 0.04 -> 0.419**. The task-spec null test does not model this,
   so it passes. M0 should measure between-subset variance (random subsets x seeds is a variance-components
   design). Candidate fixes: more random subsets, a mixed model / subset-level test, or testing against the
   placebo *subset* distribution. Needs a decision before any result run.
2. Values left TBD in configs: `task.batch_size` (LeRobot default 8, `configs/train.py:102`),
   `task.eval_batch_size`, `task.steps`, `task.per_run_gpu_h`, `protocol.n_train_seeds/train_seeds`,
   calibration `steps` and `intermediate_save_freq`.
3. M0 calibration config keeps the prompt's 3 seeds [1000, 1001, 1002]. That is enough for a rough sigma, but a
   power analysis on 3 seeds is noisy; consider 6 to match D-015.
4. Calibration evaluates intermediate checkpoints on the protocol eval seeds (50 rollouts). Choosing `steps` from
   those curves uses eval seeds before any entrant runs. Acceptable, or use curation-range seeds for that?
5. lerobot-eval 0.6.1 has no flag to disable video; it always renders up to 10 episodes
   (`scripts/lerobot_eval.py:785`). Accepted for now (videos land under scratch).
6. The mixture recipe `configs/mixtures/task_a_v0.yaml` is a DRAFT: source proportions and defect counts are
   placeholders. Quality tiers: pause/duplicate/wrong_task -> degraded; jitter/gripper_drop -> bad (future).
7. Cleaner selection rng is seeded by sha256(task|name|version|budget), so one subset per cleaner is shared
   across train seeds (seeds replicate training, not selection). This is what makes question 1 bite.

## Phase log

All times UTC, 2026-11-19. GPU-hours used in every phase: **0**.

| Phase | Commit(s) | Outcome |
|---|---|---|
| 0 bootstrap | `ee06c92` (main), `79236d1` | docs committed; main and build-core pushed |
| 1 foundations | `5f8b10c` | env `rcb` (py3.12), lerobot 0.6.1, paths/config/results |
| 2 stats | `29e59b8` (blocked), `ce47c94` D-015, `7095a4d` | null FWER 0.060 at 6 seeds; power 0.98 |
| 3 runner | `de5559a` | spec/run_id, planner, locked idempotent executor, fake backend |
| 4 cleaners | `b970018` | API, registry + entry points, built-ins, oracle, `submit`, import guard |
| 5 data | `2dc7f00` | hub, DatasetView, mixtures, defects, D-013 subset path; integration tests pass |
| 6 wrappers | `17d0bdb` | train/eval builders, eval_info parser, flag guard, local backend, GPU etiquette |
| 7 configs + CLI | `d3a5bfd` | configs, CLI, report (FAKE-labeled), MDE zero-sigma bug fix + regression test |
| 8 e2e proof | (this commit) | outputs below |

### Phase 1 details
- `pip index versions lerobot`: latest 0.6.x is **0.6.1** (0.7.0 exists, unused per D-001). Requires-Python
  `>=3.12`. ALOHA extra: **`aloha`** (gym-aloha >=0.1.4,<0.2.0). Installed
  `lerobot[aloha,training,evaluation]==0.6.1` (training/evaluation needed for the `--help` flag checks).
- `PYTHONNOUSERSITE=1` is required on this Mac (`~/.local` site-packages leaked into the env).
  `requirements.lock` = `pip freeze --exclude-editable` with it set.

### Phase 2 details
- Null FWER (7 cleaners, sigma_seed 0.06, 200 rollouts): 3 seeds 0.090, 4: 0.083, 5: 0.069, **6: 0.060**.
- The test uses 20 000 grids instead of the prompt's 1000: at a true FWER of ~0.061, 1000 grids give MC noise of
  ±0.0075 and seed 0 lands at 0.074. More grids make the test stricter, not weaker. No seed was searched.
- `test_three_seeds_is_not_enough` guards D-015 (asserts FWER > 0.07 at 3 seeds).
- Power: 20pp effect, sigma_seed 0.05, one real effect among 7 cleaners: 5 seeds 0.943, 6 seeds 0.98.

### Phase 5 details
- `lerobot/aloha_sim_transfer_cube_human`: 50 episodes x 400 frames, fps 50, features
  observation.images.top (video 480x640x3), observation.state (14), action (14), next.done; codebase v3.0.
- Integration run built a 6-episode mixture (5 human episodes + pause + wrong_task + duplicate), reloaded it with
  LeRobot, and checked lengths, held frames, task strings, duplicate equality, video decode and the labels
  sidecar (stored beside the dataset dir, not inside it).
- D-013: materialized a 3-of-6 subset via `split_dataset`; its `meta.stats["action"]` mean/min equal the kept
  episodes' own frames and differ from the full dataset's stats.

## Phase 8 — end-to-end proof (CPU, fake backend)

Machine: Apple M5 Pro (`Viveks-Macbook-M5-Pro.local`), scratch `/Users/vivek/robocurate-scratch`.

### 1. `robocurate plan configs/grids/task_a_v0.yaml`
```
grid: task_a_v0   task: aloha_transfer_cube   budgets: [0.5]
train seeds: TBD  (n = TBD)
condition cells: 6
  - full                         budget 1.0
  - random                       budget 0.5  subset 0 (seed 2000)
  - random                       budget 0.5  subset 1 (seed 2001)
  - random                       budget 0.5  subset 2 (seed 2002)
  - cleaner:length_heuristic     budget 0.5
  - oracle                       budget 0.5
runs: 6 cells x TBD seeds = TBD
steps per run: TBD
per-run GPU-h: TBD   total GPU-h: TBD
TBD (blocks result runs, D-014): n_train_seeds, train_seeds, task.steps, task.per_run_gpu_h, task.batch_size, task.eval_batch_size
```

### 2. `robocurate run configs/grids/task_a_v0.yaml --backend fake --fake-params configs/fake/e2e_test.yaml --results-dir $ROBOCURATE_SCRATCH/results/phase8-fake`
Fake params: base 0.5, sigma_seed 0.06, **6 seeds** (D-015 floor; the prompt said 3, written before D-015),
effects oracle +0.25, length_heuristic +0.0, full/random 0; sleep 0.25 s per run so it can be killed.
```
=== run 1: SIGKILL after 4.0 s (returncode -9) ===
after kill: results=12 locks=1 tmpfiles=0
stale lock: {"host": "Viveks-Macbook-M5-Pro.local", "pid": 40937}
=== run 2: restart ===
[done]    709771e207a8ced3  oracle b=0.5 seed=4
[done]    2768d72f7f167862  oracle b=0.5 seed=5
done=24 skipped_existing=12 skipped_locked=0 reclaimed_stale_locks=1
=== run 3: no-op ===
36 runs -> /Users/vivek/robocurate-scratch/results/phase8-fake
done=0 skipped_existing=36 skipped_locked=0 reclaimed_stale_locks=0
check: result files=36 unique run_ids=36 unique specs=36 max dup=1 filename==run_id: True locks left=0
```

### 3. `robocurate report $ROBOCURATE_SCRATCH/results/phase8-fake`
```
*** FAKE RESULTS (synthetic backend) ***
task aloha_transfer_cube budget 0.5:
  full                         runs=6   mean=0.513
  random                       runs=18  mean=0.487
  oracle                       runs=6   mean=0.729
  cleaner:length_heuristic     runs=6   mean=0.494
  cleaner:length_heuristic vs placebo: diff=+0.007 p_holm=0.763 not significant
  [ref] oracle vs placebo: diff=+0.242 p=0.00121 SIGNIFICANT
  [ref] full vs placebo: diff=+0.026 p=0.331 not significant
  MDE (80% power): 0.078
```
Matches the expected outcome: oracle significant (injected +0.25, recovered +0.242), length not significant
(injected 0, recovered +0.007). report.md and the PNG carry FAKE banners and a watermark.

### 4. `robocurate submit tests/fixtures/cleaners/good_cleaner.py`
```
conformance: tests/fixtures/cleaners/good_cleaner.py
  [PASS] static
  [PASS] loads: smoothness_fixture v1
  [PASS] rollouts: requires_rollouts=0 (cap 100)
  [PASS] runtime: limit 60.0s
  [PASS] count: returned 20, budget 20
  [PASS] unique: 0 duplicates
  [PASS] valid: ids must be episode indices of the view
  [PASS] determinism: same rng seed must give the same ids
  [PASS] no-leak
RESULT: PASS
selection written: /Users/vivek/robocurate-scratch/selections/smoothness_fixture-v1.json
```
`submit tests/fixtures/broken_cleaners/label_import.py` -> `[FAIL] static: robocurate.data.labels`, RESULT: FAIL,
exit 1.

### Test counts (final)
| Suite | Result |
|---|---|
| `pytest -q -m "not gpu and not slow and not integration"` | 128 passed |
| `pytest -q -m integration` (CPU, network) | 3 passed |
| `pytest -q -m slow` (FWER robustness x6, `--help` flag check) | 7 passed |
| `ruff check .` | clean |
| flag-existence tests (dataclasses + `--help`) | pass; also reject 5 invented flags |
| import-graph tests (cleaners cannot reach labels/eval/oracle) | pass |

## How GPU use was prevented
No NVIDIA GPU on this Mac (`nvidia-smi` not found; `torch.cuda.is_available()` = False). Every command ran
with `CUDA_VISIBLE_DEVICES=""` (scripts/env.sh, tests/conftest.py). `robocurate run --backend local` refuses
when it is empty (tested). No lerobot-train or lerobot-eval job was run; only their `--help`. GPU-hours: 0.

## LeRobot touchpoints (lerobot 0.6.1, paths relative to site-packages/lerobot/)
- **Subset training and D-013.** `DatasetConfig.episodes` exists (`configs/default.py:38`; passed to
  `LeRobotDataset` at `datasets/factory.py:101,114`), but metadata loads the FULL `meta/stats.json`
  (`datasets/dataset_metadata.py:219`) and lerobot-train feeds `dataset.meta.stats` to the processors
  (`scripts/lerobot_train.py:351,376-377`), so it violates D-013. We use
  `dataset_tools.split_dataset` (`datasets/dataset_tools.py:175`): it writes the subset as its own dataset with
  stats aggregated from the kept episodes only (`:852-948`, aggregation `:941-948`). It sorts episode ids
  (`:236`). ACT image stats are replaced by ImageNet stats (`use_imagenet_stats`, `configs/default.py:41`;
  `datasets/factory.py:137-142`) identically for every condition.
- **Dataset read:** `LeRobotDatasetMetadata` (`datasets/dataset_metadata.py:63`); `LeRobotDataset.__init__`
  (`datasets/lerobot_dataset.py:47`); `hf_dataset` (`:397`); `absolute_to_relative_idx` (`:405`; built at
  `datasets/dataset_reader.py:146-149`); episodes metadata columns `dataset_from_index`, `dataset_to_index`,
  `length`, `tasks`; codebase `v3.0` (`datasets/dataset_metadata.py:60`).
- **Dataset write:** `LeRobotDataset.create` (`datasets/lerobot_dataset.py:684`), `add_frame` (`:418`; frame
  needs `task`, must not carry DEFAULT_FEATURES, `datasets/feature_utils.py:221-245`; images may be HWC,
  `:330-351`), `save_episode` (`:434`), `finalize` (`:474`). DEFAULT_FEATURES: `utils/constants.py:81-87`.
- **Cache:** `HF_LEROBOT_HOME` read at import (`utils/constants.py:69`).
- **Train flags:** `seed` (`configs/train.py:96`), `steps` (`:111`), `batch_size` (`:102`), `env_eval_freq`
  (`:113`, in-training eval at `scripts/lerobot_train.py:629,715`), `save_checkpoint` (`:120`), `save_freq`
  (`:123`), `output_dir` (`:86`; must not exist unless resuming, `:259`), `job_name` (`:87`);
  `wandb.enable` (`configs/default.py:82`); `policy.push_to_hub` default True (`configs/policies.py:70`),
  `policy.device` (`:62`); ACT registered as `"act"` (`policies/act/configuration_act.py:22`).
- **Checkpoints:** final checkpoint always saved, `save_freq <= 0` disables periodic saves
  (`common/train_utils.py:55-63`); dir `<out>/checkpoints/<step:0Nd>/pretrained_model`, N=max(6,len(steps))
  (`common/train_utils.py:44-52`, `utils/constants.py:47-49`).
- **Eval flags:** `--policy.path` via `__get_path_fields__` (`configs/eval.py:77`); no path -> random weights
  with a warning only (`:55-58`); `--env.type=aloha`, `--env.task` (`envs/configs.py:147-152`, gym id
  `gym_aloha/AlohaTransferCube-v0`); `eval.n_episodes`/`eval.batch_size` (`configs/default.py:95,98`;
  batch_size 0 auto-sizes from CPUs, `:110-126`); `seed` (`configs/eval.py:39`).
- **Eval seed derivation:** `set_seed(cfg.seed)` (`scripts/lerobot_eval.py:748`); `make_env(n_envs=
  eval.batch_size)` (`:752-757`); batch `b` seeds `range(seed + b*num_envs, seed + (b+1)*num_envs)`
  (`:528-533`), applied once at reset (`:223`); truncated to n_episodes (`:662-667`). **Episode i uses env seed
  `--seed + i`, independent of batch size**; padding episodes touch seeds up to `seed + ceil(n/b)*b - 1`.
- **Eval output:** `eval_info.json` (`:819-820`) = `{per_task:[{task_group, task_id, metrics:{sum_rewards,
  max_rewards, successes, video_paths, predicted_video_paths}}], per_group, overall}` (`:1053-1107`;
  `ACC_KEYS` `:834`). Per-episode seeds are not written (`:877-884`). Single-task aloha -> one group, task_id 0
  (`envs/utils.py:487-493`).
- **Video:** no disable flag; `max_episodes_rendered = 0 if recording else 10` (`scripts/lerobot_eval.py:785`).

## Deviations from the task prompt
- Installed `lerobot[aloha,training,evaluation]` rather than only `[aloha]` (needed for `--help`).
- Null FWER test: 20 000 grids at the D-015 floor of 6 seeds (prompt: 1000 grids, 3 seeds). The 3-seed version
  failed (B-001); the user chose D-015.
- Phase 8 fake run: 6 seeds instead of 3 (D-015 floor).
- `results.py` was committed with Phase 1 together with `stats.py` (it validates the Wilson CI).
- `backends/local.py` was written in Phase 6, after the wrappers it wires; it is never executed.
- Ruff line length 120 (was 110).
- `task.batch_size` and `task.eval_batch_size` set to TBD (hyperparameters = protocol decisions).

## Parking lot
- Label diagnostics in the report (precision/recall of each cleaner's kept set vs labels, D-008).
- Replay-mode defect generation in sim (jitter/gripper drop) to build `bad`-tier episodes.
- A subset-variance-aware null simulation as a slow test, once open question 1 is decided.
