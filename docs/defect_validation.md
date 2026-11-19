# Defect validation: observed defect rates in public SO-100/SO-101 datasets vs our injection rates

Generated 2026-11-19 by `robocurate validate-defects` (src/robocurate/data/validate.py) from **actions, state and
timestamps only**. Only `meta/**` and `data/**` were downloaded (0.5–45.6 MB per dataset); no video. Full
statistics: [defect_validation_stats.json](defect_validation_stats.json). This informs the real mixture recipe.
No mixture config was edited.

The M0 calibration mixture output was **not available locally**: M0 was never built or launched (B-003), so it
is not included.

## Definitions

These were fixed before the first run. The interior/boundary split was added after the first run; both are reported.

- **Pause frame:** `||s[t+1]-s[t]|| < 0.05 × dataset median step`.
- **Pause segment:** at least 0.5 s of consecutive pause frames.
  - *Boundary* segments touch the first or last transition (the robot idle before or after teleoperation).
  - *Interior* segments are all the others. Our injected pauses are interior (onset 20–80% of the episode).
- **Jitter:** the share of FFT action power above fps/4, per episode. The table reports the share of episodes with jitter above 0.10.
- **Near-duplicate:** a pair of episodes whose z-scored actions, resampled to 64 points, have RMS difference below 0.05, with lengths within 5%.
- **Exact duplicate:** identical action and state bytes.

## Results

| dataset | revision | format | episodes | length s, p50 (p5–p95) | pause rate (any) | pause rate (interior) | interior pause s, p50 (p5–p95) | jitter > 0.10 | exact / near dups | timestamp problems | tasks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `lerobot/svla_so101_pickplace` | `f641879e22` | v3.0 | 50 | 7.7 (6.7–9.9) | 0.96 | 0.06 | 0.62 (0.57–1.96) | 0.00 | 0 / 0 | 0 | 1 |
| `lerobot/svla_so100_sorting` | `13870ca969` | v3.0 | 52 | 23.1 (19.1–27.0) | 0.88 | 0.50 | 0.92 (0.53–1.46) | 0.00 | 0 / 0 | 0 | 1 |
| `jchandru08/so100_press_red_button` | `b8544ee7cb` | v2.1 | 29 | 19.7 (19.6–19.8) | 0.97 | 0.90 | 0.72 (0.50–2.14) | 0.00 | 0 / 0 | 0 | 1 |
| `SO-101-robot-learning/so101_pick_place` | `c56fed5bc3` | v3.0 | 50 | 11.8 (10.0–14.1) | 1.00 | 0.30 | 1.20 (0.57–2.37) | 0.00 | 0 / 0 | 0 | 1 |
| `kmg0620/so101_bomb_pick_place_v1` | `309fb95630` | v3.0 | 150 | 17.7 (12.9–33.4) | 0.85 | 0.83 | 0.87 (0.50–2.66) | 0.15 | 0 / 0 | 22 | 1 |

## Comparison with our injection (configs/mixtures/task_a_v0.yaml, DRAFT)

| defect | ours (task_a_v0 draft) | observed in these 5 datasets | reading |
|---|---|---|---|
| pause | 6 of 54 episodes (11%), 25–75 frames at 50 fps = 0.5–1.5 s, interior | interior pauses in 6–90% of episodes (median across datasets 50%); median interior segment 0.6–1.2 s, p95 up to 2.7 s | Our pause **lengths** are realistic. Our **rate** is at the low end. Real teleop also has near-universal boundary idle, which we do not inject. |
| duplicate | 4 of 54 (7%) exact copies | 0 exact, 0 near-duplicate pairs in 331 episodes | Exact duplicates were not observed. Our rate is a stress test, not a realistic rate. |
| wrong_task | 4 of 54 (7%) | every dataset has exactly 1 task string | Mislabeled task strings cannot be measured in single-task datasets (no signal either way). |
| jitter (future, replay mode) | not in the draft | 0% of episodes above 0.10 in 4 datasets; 15% in `kmg0620/so101_bomb_pick_place_v1` (p95 share 0.41) | High-frequency action noise exists in some community data, concentrated in one collector's dataset. |
| timestamps | not injected | 22 of 150 episodes with gaps or non-monotone timestamps in `kmg0620/so101_bomb_pick_place_v1`; 0 elsewhere | A real defect class we do not model (dropped frames). |

Caveats:
- These are 5 datasets, all single-task. That is enough to sanity-check orders of magnitude, not to estimate population rates.
- The ALOHA sim human demonstrations we mix are not SO-100/101 teleop.
- Pause detection is threshold-based, so slow, deliberate motion near the threshold counts as a pause.
