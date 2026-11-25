# GPU-phase runbook

This is what happens when GPU time is approved. Nothing here has been run on a GPU yet. The launch kit was built
and dry-run on CPU with the fake backend (PROGRESS.md, "Launch kit dry run"). Every step runs on hokkaido, inside
tmux, on the branch Vivek names, with a clean git tree.

## 0. Rules for every step

- **Scratch (D-027):** `ROBOCURATE_SCRATCH=/share/datasets/robocurate`. The kit refuses any other path on a real
  run. This is network storage, so stage b measures whether data loading bottlenecks training. The only allowed
  mitigation is DataLoader `num_workers` (speed, not results), and it is recorded per run.
- **Environment:** `source scripts/env.sh` **before** Python starts. LeRobot reads `HF_LEROBOT_HOME` once, at
  import (`lerobot/utils/constants.py:69`), and the kit checks this. Before the first real run, edit
  `scripts/env.sh` so that `ROBOCURATE_SCRATCH` is the D-027 path and `CUDA_VISIBLE_DEVICES` is set; its CPU-only
  line must go only when the task prompt allows GPU use.
- **Disk:** at most 200 GB under the scratch directory. df and quota are checked at startup, and the disk guard is
  checked again before every run.
  - D-018 cleanup: each run's materialized subset is deleted after the run.
  - Eval videos are deleted after `eval_info.json` is parsed.
  - Non-final checkpoints are deleted after their eval.
  - **Exception:** the full-data seed-1000 M0 run keeps its 10%, 30% and 60% checkpoints. They are the D-031
    policy-rollout sources. This needs Vivek's OK, because D-027 says "only final checkpoints are kept".
- **GPU etiquette (CLAUDE.md):**
  - Never start while another user's process uses more than 2 GB or 10% of the GPU. Stage a polls every 10 min
    and gives up after 72 h, and every run re-checks.
  - Never touch other users' processes.
  - Log GPU-h in PROGRESS.md.
- **Caps per M0 task:** 36 GPU-h and 72 h of run time. A run that fails twice is skipped; after 3 skipped runs the
  stage stops.

## 1. M0, Task A (ALOHA transfer-cube + ACT)

```bash
tmux new -s rcb-m0
source scripts/env.sh
scripts/m0_pipeline.sh --task a
```

| stage | does | output |
|---|---|---|
| a | waits for a free GPU | log line with hours waited |
| b, b2 | 500-step smoke train + 10 eval episodes on calibration seeds (300000+); if GPU util < 50% and data wait > 30%, retries with `num_workers` doubled (at most twice) | `smoke` in `m0_summary.json` |
| c | `pytest -m gpu tests/test_gpu.py`: training seed determinism (loss trajectory, 1e-3 rel.) and eval batch invariance (per-episode success equal at batch 5 and 25) | on failure, the pipeline stops with the pytest output (diagnose, do not work around) |
| d | eval batch = fastest of {5, 10, 25, 50} under 12 GB; S = LeRobot's default steps (100,000, `configs/train.py:111`) capped so one run (train + 200 final + 9 x 50 intermediate eval episodes) takes at most 2.5 h; stops if S < 25% of the default | writes `steps`, `intermediate_save_freq` (S/10), `eval_batch_size`, `batch_size` (LeRobot default 8, D-017), `num_workers` into the calibration config, and commits it |
| e | 11 runs: full data x seeds 1000/1001/1002 + K = 4 random subsets (seeds 2000-2003) x seeds 1000/1001, budget 0.5; packing (2 runs at once) only after a solo full-data run used < 50% GPU and < 15 GB | one results JSON per run in `results/m0/<cal>/`, a PROGRESS.md line and a commit per run |
| f | s_seed and s_sub with 95% CIs, step curve, cost, disk, D-026 power table (n in {3, 5}, K in {8, 12}, R in {1, 2, 3}), projected launch GPU-h, proposed design | `m0_summary.md` (also appended to PROGRESS.md) |

The run is resumable: rerunning skips finished runs (executor idempotence). `--stages e,f` restarts after stage d.

**GPU-h estimate:**

GPU-h(M0) ≈ 11 × per_run_h + smoke + autoconf

- per_run_h ≤ 2.5.
- smoke + autoconf ≈ 500 × step_s + 4 × t_eval(50) + 10-episode eval, usually < 1 h.
- Upper bound: **≈ 28.5 GPU-h**, under the 36 GPU-h cap. Packing lowers it when the GPU is under-used.

## 2. M0, Task B (PushT + Diffusion Policy, D-030)

**Prerequisite:** gym-pusht 0.1.8, pymunk 6.11.1, and diffusers 0.39.0, plus the transitive pins from the
dry run, are in `requirements.lock`. OpenCV is only `opencv-python-headless` (lerobot's pin). Install the
lock with `--no-deps` so gym-pusht does not also pull the `opencv-python` wheel.

```bash
scripts/m0_pipeline.sh --task b
```

Same stages, with the same estimate formula. **Known risk:** Diffusion Policy samples noise during inference, so
stage c's batch-invariance test may fail for Task B even though env seeds are batch-invariant. If it fails, the
pipeline stops by design, and Vivek decides (for example, fix the eval batch size per task in the protocol).

## 3. Mixture sources and curation scores

1. **Policy rollouts and replay-mode episodes** (`scripts/gpu_phase_rollouts.sh`). Curation seeds only, taken
   from the top of the range, with a separate offset per deferred source.
   - Sources are the M0 seed-1000 full-data checkpoints at 10/30/60/100% of S.
   - **Not built yet:** an adapter from a LeRobot checkpoint to an `obs -> action` callable that also records the
     camera frames the ACT/Diffusion mixtures need. A scripted transfer-cube policy is also missing (gym-aloha
     0.1.5 and lerobot 0.6.1 ship none). Until both exist, `--backend local` refuses with this list. The fake
     backend exercises seeds, labels and the LeRobot writer.
   - GPU-h ≈ (110 / 200) × eval_h_per_200 (Task A) and (100 / 200) × eval_h_per_200 (Task B).
2. **Build the mixtures.** Move each deferred source under `sources:` of its recipe, then run
   `build_mixture(...)`. Post-hoc pause rates then apply to all episodes. Run `robocurate validate-defects` on the
   result.
3. **GPU entrants' scores** (`scripts/prepare_gpu_scores.sh <grid> local`): Robometer progress for
   robometer_filter, rcb_score and RA-BC, cached once per (task, budget, replicate) (D-022).
   - **Not built yet:** `robocurate prepare --backend local` (it refuses today; it needs the mixture dataset,
     Robometer weights and the M0 reference checkpoint).
   - GPU-h = the Robometer pass over the mixture's frames, measured and reported as curation cost.

## 4. Freeze (eval-v1)

1. Fix n, K and R from the M0 analysis by the rule in docs/preregistration.md section 9. This is Vivek's decision;
   the kit only proposes. Fill `n_train_seeds`, `train_seeds` and the task fields (`steps`, `per_run_gpu_h`,
   batch sizes).
2. Log the values as a deviation-log entry in the pre-registration.
3. Vivek creates `src/robocurate/eval/FROZEN` and tags `prereg-v1`. From then on, `src/robocurate/eval/`,
   `configs/protocol.yaml` and `configs/frozen/` are never edited.

## 5. Full grids

```bash
robocurate run configs/grids/task_a_v1.yaml --backend local
robocurate run configs/grids/pusht_diffusion_v0.yaml --backend local
```

Then run `robocurate report <results> --site space/`.

**GPU-h estimate per task:** GPU-h = R × n × units × per_run_h.

- units = 1 (full) + K (random) + 1 (oracle) + m (selection entrants), plus 1 + K for the weighting track (Task A
  only).
- Task A, n = 3, K = 8, R = 1, m = 9: 3 × 28 = 84 runs.
- Task B: 3 × 19 = 57 runs.
- At the 2.5 h cap that is at most 210 and 143 GPU-h. The M0 analysis prints the exact projection for every
  design.

## 6. The first GPU step

On hokkaido, as the user:

1. Check that `/share/datasets/robocurate` exists and is writable: `mkdir -p` it once, then check with `df -h`.
2. Edit `scripts/env.sh` per section 0.
3. Start `tmux new -s rcb-m0`.
4. Run `source scripts/env.sh && scripts/m0_pipeline.sh --task a`.
