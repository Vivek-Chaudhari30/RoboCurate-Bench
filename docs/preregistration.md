# RoboCurate-Bench pre-registration (D-032)

Status: **draft for review, not tagged.** Vivek tags it `prereg-v1` after review. Written 2026-11-19, before any
entrant result exists (no result-mode run has been executed; M0 calibration has not run). After the tag, every
change is appended to the deviations log at the end of this file with a number, a date and a reason. Earlier text
is never edited silently.

Sources of truth: `DECISIONS.md` (D-001 to D-032), `configs/protocol.yaml`, the task, grid and mixture configs
named below, and `CLAUDE.md`. If this file and those disagree, the decision log and the configs win, and the
mismatch is logged here as a deviation.

## 1. Question

Do published robot-data cleaners (episode filters and per-episode weighting methods) raise the closed-loop success
of an imitation-learning policy trained on a mixed-quality dataset, compared with removing (or re-weighting) the
same amount of data at random?

The comparison is against a placebo, not against "no cleaning". A cleaner that keeps 50% of the episodes is
compared with random 50% subsets trained under the identical protocol. The full dataset and a label oracle are
reported as references.

## 2. Hypotheses

For each enabled entrant e, each task, and the launch budget b = 0.5 (D-006):

- **H0(e):** the success rate of policies trained on e's selection equals that of policies trained on a
  size-matched random subset (selection track), or on e's weights randomly permuted across episodes (weighting
  track, D-020, D-028).
- **H1(e):** they differ (two-sided). The direction is reported. "Helps" means a positive difference.

There is one hypothesis per (entrant, task). No other comparison is confirmatory.

## 3. Primary test (D-026)

- **Unit of analysis:** one training run, scored by its success rate over 200 evaluation rollouts of the final
  checkpoint (CLAUDE.md rules 2, 3 and 5). Rollouts are never pooled across runs.
- **Statistic:** the exact balanced prediction t-test.
  t = (mean_e − mean_p) / sqrt((1 + 1/K) · MS_between / n), with df = K − 1.
  - mean_e is the mean over e's n runs.
  - mean_p is the mean over the K placebo subset means, each over n runs.
  - MS_between is n times the variance of the K placebo subset means.
  - With R mixture replicates: SE² = (1 + 1/K) · s²_pooled / R and df = R(K − 1).
- **Multiplicity:** Holm across the entrants of one track within one task, α = 0.05.
- **95% CI (primary):** diff ± t(df, 0.975) · SE, from the same statistic. The run-level bootstrap CI and the
  D-016 variance-components and Welch tests are reported as labelled secondary diagnostics only.
- **Calibration evidence:** null family-wise false-positive rate is 0.039–0.044 in all 32 allowed design cells
  (sigma_sub 0–0.06, sigma_seed 0.03–0.06, K ∈ {8, 12}, n ∈ {3, 5}; 200,000 simulated grids each, upper 95% MC
  bound ≤ 0.0451). See PROGRESS.md, "D-026 calibration".

## 4. Design rules

- **Balanced seeds:** n_c = n_p = n ≥ 3 training seeds per entrant and per placebo subset (D-026). The same n
  applies to the full-data and oracle references.
- **Placebo size:** K ≥ 8 random subsets per budget (selection track), or K ≥ 8 weight permutations (weighting
  track). Subset seeds come from `random_subset_seeds` and permutation seeds from `weight_permutation_seeds`, both
  in `configs/protocol.yaml` (2000–2011 and 3000–3011).
- **Eligibility labels:** designs that miss these rules are labelled NOT RESULT-ELIGIBLE everywhere (report,
  leaderboard, site, CLI), and result mode refuses to run them.
- **Checkpoint:** the final checkpoint at a fixed step count, identical for every condition (D-003, D-004). There
  is no best-checkpoint selection.
- **Training config:** identical training configuration across conditions. Weighting entrants change only the
  per-sample weights, through LeRobot's own RA-BC code on both arms (D-020, D-028).
- **Normalization stats:** computed from each condition's own training subset (D-013).
- **Seed blocks:**
  - Eval env seeds: `eval_seed_start` = 100000 + i for episode i.
  - Calibration evals: `calibration_seed_start` = 300000 (D-017), never eval seeds.
  - Cleaners that roll out policies: curation seeds 500000–599999, capped at 100 rollouts per (task, budget,
    replicate) (D-007, D-022). GPU-phase mixture sources take curation seeds from the top of that range down.
- **Information barrier:** cleaners never receive ground-truth labels, eval code, eval seeds or eval results
  (CLAUDE.md rule 7). Selection ties are broken by a salted content hash (D-021).
- **Frozen inputs:** each selection is computed once per (task, budget, replicate) and cached by input hash
  (D-022).
- **Freeze:** `src/robocurate/eval/`, `configs/protocol.yaml` and `configs/frozen/` freeze at eval-v1. Eval-seed
  performance is not observed before then (D-017).

## 5. Reference conditions (not entrants, not tested against Holm)

- **Full data:** all episodes. It shows the cost or benefit of halving the data at random.
- **Label oracle** (D-012): keeps the episodes with the best hidden ground-truth quality labels. It shows whether
  curation can help on this task at all. If the oracle does not beat the placebo, entrant nulls on that task are
  uninformative and are labelled so.

## 6. Tasks and mixtures

| | Task A (D-009, D-031) | Task B (D-030) |
|---|---|---|
| env | ALOHA sim transfer-cube (`aloha`, `AlohaTransferCube-v0`) | PushT (`pusht`, `PushT-v0`) |
| policy | ACT | Diffusion Policy |
| task config | `configs/tasks/aloha_transfer_cube.yaml` | `configs/tasks/pusht_diffusion.yaml` |
| launch grid | `configs/grids/task_a_v1.yaml` (K = 8, every enabled entrant, RA-BC with 8 permutations) | `configs/grids/pusht_diffusion_v0.yaml` (K = 8, every enabled selection entrant) |
| mixture | `configs/mixtures/task_a_v1.yaml` | `configs/mixtures/pusht_diffusion_v1.yaml` (proposal) |
| mixture content | 50 scripted (10 of them replay-mode perturbed: 5 jitter, 5 failed grasp), 50 human, 100 policy rollouts from full-data ACT checkpoints at 10/30/60/100% of training (25 each, successes and failures kept) | all 206 human demos + 100 policy rollouts (same checkpoint fractions) |
| post-hoc defects | interior pauses on 50% of episodes, 0.62–1.20 s; duplicates at rate 0 (observed medians, docs/defect_validation.md); no wrong-task strings | same rates; pauses 6–12 frames at 10 fps |
| weighting track | RA-BC (LeRobot) vs 8 progress-curve permutations (D-028) | none: LeRobot RA-BC needs `chunk_size`, which Diffusion Policy lacks |

Task B holds because all three D-030 checks passed on CPU (PROGRESS.md, "Task B setup").

Episode order in every mixture is shuffled by the recipe seed (D-021). Recipes may change only before any entrant
result exists, and they freeze with eval-v1.

## 7. Entrants

Status at the time of writing (`robocurate.cleaners.registry`). Versions are the entrant interface versions; the
wrapped tool versions are pinned in `requirements.lock`.

| entrant | version | track | implementation | status |
|---|---|---|---|---|
| lerobot_doctor | 1 | selection | wraps lerobot-doctor 0.2.0 | enabled |
| kinema | 1 | selection | wraps kinema 0.1.0 | enabled |
| roboaudit | 1 | selection | wraps roboaudit 0.3.1 | enabled |
| rda | 1 | selection | wraps robot-data-audit @47c69459 | enabled |
| robometer_filter | 1 | selection | wraps LeRobot's Robometer reward | enabled |
| rcb_score | 1 | selection | original, frozen (D-023) | enabled |
| demo_score | 1 | selection | reimplementation (arXiv 2503.03707) | enabled |
| deminf | 1 | selection | reimplementation (arXiv 2502.08623) | enabled |
| length_heuristic | 1 | selection | built-in baseline | enabled |
| rabc | 1 | weighting | wraps LeRobot RA-BC | enabled (Task A only) |
| cupid | 1 | selection | reimplementation (arXiv 2506.19121) | **disabled**: failed toybed sanity with ACT-style and diffusion policies (D-029) |

An entrant that fails its toybed sanity check with published defaults is disabled. It is never tuned to pass, and
it is not listed under its name.

## 8. Outcome labels (per entrant × task)

Each entrant gets one label, computed from the primary test and the achieved minimum detectable effect (MDE).
The MDE is the effect with 80% power at the measured variance components (Bonferroni α/m, conservative for Holm).

- **Positive:** Holm-adjusted p < 0.05 and diff > 0.
- **Negative (harms):** Holm-adjusted p < 0.05 and diff < 0.
- **No detectable effect:** not significant, and the MDE is ≤ 10 percentage points. Effects at or above the MDE
  are unlikely.
- **Inconclusive:** the primary 95% CI crosses 0 and the MDE is > 10 percentage points.

The MDE is reported with every label. The report computes the CI, the MDE and the label for every entrant
(`robocurate report`: `ci95`, `mde` and `label` in leaderboard.json, plus a `label` column in report.md). Negative and null results are reported exactly like positive ones (CLAUDE.md
rule 9).

## 9. Values fixed from M0, before any entrant run

The following are **TBD** and are set only from M0 calibration (full data × 3 seeds plus K = 4 random subsets × 2
seeds per task, calibration seeds), never from entrant or eval-seed results:

- steps S;
- n (seeds per condition);
- K (8 or 12);
- R (mixture replicates, 1–3);
- batch sizes and per-run GPU-h.

**Rule (fixed now):**

1. S is LeRobot's default step count for the policy, capped so that one run takes at most 2.5 GPU-h. If the cap
   gives less than 25% of the default, M0 stops and the task is re-planned.
2. From M0, estimate s_sub and s_seed.
3. Over the design grid n ∈ {3, 5}, K ∈ {8, 12}, R ∈ {1, 2, 3}, compute the D-026 MDE (80% power, α = 0.05,
   Bonferroni over the track's entrants) and the projected GPU-h.
4. Choose the cheapest design with MDE ≤ 12 percentage points that fits the approved compute.
5. If no design meets 12 pp within compute, choose the design with the smallest MDE that fits, and report its
   achieved MDE as a limitation.
6. Ties go to more subsets K before more seeds n (subset variance dominates, D-016).

These values are recorded as a deviation-log entry (with the M0 numbers) before the first result-mode run.

## 10. Analysis plan

1. Run the frozen grid per task: full, K random subsets, oracle, every enabled entrant, and the weighting track
   where available, with n seeds each and R replicates.
2. Per task and track, run the primary D-026 test for every entrant vs its placebo, with Holm correction.
3. Report diff, SE, df, p, Holm p, primary 95% CI, MDE and the outcome label.
4. Report the oracle and full references vs the placebo (unadjusted, labelled reference).
5. Report curation cost (GPU-h of the prepare stage) and the precision/recall of each entrant's selection against
   the hidden labels (diagnostic only, D-008).
6. Exploratory analyses, if any, are labelled exploratory and kept out of the confirmatory table.

## 11. Deviations log

None yet.

<!-- Append entries as: "### Deviation N (YYYY-MM-DD): what changed / why / effect on the confirmatory analysis". -->
