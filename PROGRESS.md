# PROGRESS

**Current milestone: M0 — BLOCKED at Part 1 (B-002: D-016 test fails null calibration at K=4); GPU work not started**

## Blocked

- **B-002 (2026-11-19T05:55Z, OPEN) — the D-016 test as specified fails the Part 1 null calibration at K=4.**
  Branch `m0-calibration`. Stop condition hit ("Any Part 1 FPR cell exceeds 0.07 (do not weaken the test)").
  Parts 2-4 (hokkaido, GPU) were NOT started; GPU-hours used: 0.
  - *Symptom.* Holm family-wise FPR (7 null cleaners, 200 rollouts, base 0.5, 40 000 grids/cell, MC SE in
    parentheses) exceeds 0.07 in 3 of 24 cells, all K=4: (sigma_sub, sigma_seed) = (0.04, 0.03) 0.097,
    (0.06, 0.03) 0.087, (0.06, 0.06) 0.095. Every K=8 cell passes (max 0.062).
  - *Implementation checked.* Scalar (`compare_cleaners_d016`) and vectorized (`subset_test_arrays`) versions agree
    to 1e-15 (`test_subset_test_vectorized_matches_scalar`), and SE^2 matches the D-016 formula exactly
    (`test_subset_test_se_formula_balanced`).
  - *Cause.* With K=4 the between-subset mean square has 3 df, so s_sub^2 is truncated to 0 in ~24% of grids even
    at true sigma_sub = 0.04. In those grids, SE drops the subset term entirely and df jumps to the pooled
    within df (~18), so the per-test rejection rate at alpha/7 is 0.019 vs nominal 0.0071. Untruncated grids
    have df ~2.6 (Satterthwaite with a negative MSW coefficient). Replacing only the truncated-case df with K-1
    (candidate C) brings FWER at K=4 to <= 0.006.
  - *Hypotheses / options (protocol decision, not made here).* (1) Keep the D-016 test, require K >= 8 (passes
    every cell). (2) Candidate C: on truncation, use df = K-1. Passes every cell, but power at K=4 collapses
    (<= 0.05 for 15pp). (3) Candidate F: always df = K-1. Passes every cell (max 0.068 at (0.06, 0.03, 8,1,3));
    it has MORE power than the spec at K=8 (Satterthwaite with negative weights gives tiny df). (4) A
    resampling reference distribution (parametric bootstrap of the null). Not simulated.
  - *Power is low everywhere for 15pp once sigma_sub >= 0.04* (<= 0.32 under any rule, K=8, R=1). That points to
    mixture replicates R or larger K, which M0's measured components will size.
  - *What would resolve it.* Pick a rule (1-4). The FPR assertion then becomes a hard slow test, and
    `robocurate report` switches its primary test to D-016 (Welch becomes a labeled secondary diagnostic).
  - Reproduce: `python scripts/d016_null_grid.py --n 40000` (~10 s on the Mac).

| s_sub | s_seed | K,n_p,n_c | FWER spec (MC SE) | FWER C | FWER F | power15 spec | C | F |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.03 | 4,2,3 | 0.0177 (0.0007) | 0.0000 | 0.0000 | 0.544 | 0.051 | 0.060 |
| 0.00 | 0.03 | 8,2,3 | 0.0158 (0.0006) | 0.0016 | 0.0021 | 0.658 | 0.600 | 0.767 |
| 0.00 | 0.03 | 8,1,3 | 0.0103 (0.0005) | 0.0021 | 0.0026 | 0.475 | 0.399 | 0.591 |
| 0.00 | 0.06 | 4,2,3 | 0.0176 (0.0007) | 0.0000 | 0.0000 | 0.311 | 0.004 | 0.004 |
| 0.00 | 0.06 | 8,2,3 | 0.0153 (0.0006) | 0.0018 | 0.0023 | 0.351 | 0.196 | 0.269 |
| 0.00 | 0.06 | 8,1,3 | 0.0097 (0.0005) | 0.0019 | 0.0022 | 0.243 | 0.137 | 0.183 |
| 0.02 | 0.03 | 4,2,3 | 0.0475 (0.0011) | 0.0000 | 0.0000 | 0.409 | 0.047 | 0.060 |
| 0.02 | 0.03 | 8,2,3 | 0.0362 (0.0009) | 0.0073 | 0.0103 | 0.476 | 0.432 | 0.610 |
| 0.02 | 0.03 | 8,1,3 | 0.0269 (0.0008) | 0.0085 | 0.0109 | 0.366 | 0.304 | 0.492 |
| 0.02 | 0.06 | 4,2,3 | 0.0302 (0.0009) | 0.0000 | 0.0000 | 0.267 | 0.005 | 0.005 |
| 0.02 | 0.06 | 8,2,3 | 0.0245 (0.0008) | 0.0038 | 0.0050 | 0.280 | 0.165 | 0.240 |
| 0.02 | 0.06 | 8,1,3 | 0.0169 (0.0006) | 0.0050 | 0.0060 | 0.210 | 0.126 | 0.174 |
| 0.04 | 0.03 | 4,2,3 | 0.0968 (0.0015) **>0.07** | 0.0010 | 0.0014 | 0.205 | 0.039 | 0.053 |
| 0.04 | 0.03 | 8,2,3 | 0.0473 (0.0011) | 0.0216 | 0.0374 | 0.224 | 0.210 | 0.324 |
| 0.04 | 0.03 | 8,1,3 | 0.0624 (0.0012) | 0.0293 | 0.0428 | 0.179 | 0.149 | 0.312 |
| 0.04 | 0.06 | 4,2,3 | 0.0671 (0.0013) | 0.0001 | 0.0002 | 0.193 | 0.007 | 0.008 |
| 0.04 | 0.06 | 8,2,3 | 0.0448 (0.0010) | 0.0114 | 0.0178 | 0.167 | 0.112 | 0.182 |
| 0.04 | 0.06 | 8,1,3 | 0.0394 (0.0010) | 0.0138 | 0.0181 | 0.164 | 0.106 | 0.159 |
| 0.06 | 0.03 | 4,2,3 | 0.0866 (0.0014) **>0.07** | 0.0060 | 0.0083 | 0.090 | 0.023 | 0.038 |
| 0.06 | 0.03 | 8,2,3 | 0.0314 (0.0009) | 0.0250 | 0.0483 | 0.109 | 0.106 | 0.157 |
| 0.06 | 0.03 | 8,1,3 | 0.0522 (0.0011) | 0.0338 | 0.0683 | 0.070 | 0.061 | 0.179 |
| 0.06 | 0.06 | 4,2,3 | 0.0949 (0.0015) **>0.07** | 0.0008 | 0.0010 | 0.123 | 0.009 | 0.012 |
| 0.06 | 0.06 | 8,2,3 | 0.0446 (0.0010) | 0.0198 | 0.0364 | 0.088 | 0.069 | 0.122 |
| 0.06 | 0.06 | 8,1,3 | 0.0620 (0.0012) | 0.0302 | 0.0450 | 0.103 | 0.070 | 0.126 |


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

## M0 log (branch m0-calibration)

| UTC | Step | Commit | Outcome | GPU-h |
|---|---|---|---|---|
| 2026-11-19T05:55Z | Part 0: D-016/017/018, protocol `calibration_seed_start: 300000`, CLAUDE.md rule 10 | `63a9a5a` | committed | 0 |
| 2026-11-19T05:55Z | Part 1: D-016 variance components, Satterthwaite test, Design/power/MDE, grid script | (this commit) | **blocked, B-002** | 0 |

Part 1 commands: `pytest -q -m "not gpu and not slow"` 143 passed; `-m slow` 8 passed; `-m integration` 3 passed;
`ruff check .` clean; `python scripts/d016_null_grid.py --n 40000`. `robocurate report` still uses Welch
(unchanged) until B-002 is resolved. `SEED_FLOOR` / `test: welch_holm` in the protocol are unchanged for the same reason.

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
