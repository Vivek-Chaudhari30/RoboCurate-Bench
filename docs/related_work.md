# Related work: CuraEval and RoboCurate-Bench

**CuraEval**: Xu, Chen, Ye, Shi, "CuraEval: Evaluating Demonstration Curation on Mixed-Skill Robot Data", CoRL 2026
Workshop "Oops, I Erred", arXiv 2610.38221v1 (read 2026-11-20). Code: github.com/jiuyixu25/CuraEval (MIT).

Section numbers below refer to arXiv v1. Everything is paraphrased; nothing is quoted. Where the paper doesn't
describe something, the table says "not described" rather than guessing.

## Protocol comparison

| aspect | CuraEval (arXiv 2610.38221v1) | RoboCurate-Bench (this repo) |
|---|---|---|
| tasks / simulators | Three robomimic tasks, state-based: can, square, transport (§4.1, §3.1). | ALOHA sim transfer-cube (Task A, gym-aloha) and PushT (Task B, gym-pusht), image-based, through LeRobot (D-009, D-030). |
| data | Real mixed-skill human demonstrations: 300 per task from six operators at three skill levels (§4.1). | Constructed mixtures (D-031): scripted + human + policy rollouts, plus defects injected at measured community rates (pauses; replay-mode jitter and failed grasps), with hidden ground-truth labels (D-008). |
| policies | Diffusion Policy (main), plus BC-RNN-GMM for some comparisons (§4.1). | ACT (Task A) and Diffusion Policy (Task B); one policy class per task. |
| curation methods | PSD, SAL, length, velocity-PSD and jerk heuristics; adapted geometric-entropy and signature-entropy selectors (LOO or greedy); a CUPID implementation (§3.2, §4.3). | Four wrapped checkers, Robometer filter, RA-BC (weighting track), DemInf, Demo-SCORE, rcb-score, a length baseline, gradsim-return; CUPID disabled with evidence (D-029, D-036). |
| placebo / baselines | Size-matched random subsets: one per retention level, three at square 25%. Full data. Matched on demonstration count (§3.1, §4.1, §4.3). | K >= 8 size-matched random subsets per budget, or K >= 8 weight permutations for weighting methods (D-020, D-026, D-028). Full data and a label oracle as references (D-012). |
| seeds per condition | 3 or 6 depending on the condition; training run = unit of analysis (§4.1, Tables 1-2). | n >= 3, balanced: the same n for the entrant and every placebo subset (D-026). Training run = unit of analysis (D-002). |
| rollouts per checkpoint | 200 (square, transport) and 100 (can) (§4.1). | At least 200 for every evaluated checkpoint (CLAUDE.md rule 2). |
| checkpoint selection | Updated protocol: final checkpoint. The original reported the best checkpoint over 50-rollout evaluations; both are compared (§4.1, §4.5). | Final checkpoint at a fixed step count only, from the start (D-003, D-004). |
| eval-seed handling | One common evaluator seed for all conditions. The protocol change moved checkpoint rule, rollout count and eval seed together (§4.1, §4.5). Separation from other seeds: not described. | Fixed eval-seed block (100000 + i, batch-invariant for ACT; fixed batch for diffusion, D-035), never observed before the eval-v1 freeze. Disjoint calibration (300000+), curation (500000-599999) and data-generation (700000-799999) blocks (D-017, D-007, D-034). Cleaners never see eval seeds, eval code or labels (CLAUDE.md rule 7). |
| statistical test | Two-sided Welch t-tests, pointwise 95% CIs, Holm within families of 8-28 comparisons, separately vs random and vs full. Transport is exploratory and unadjusted (§4.1, Tables 1-2). | Exact balanced prediction t-test, df = K − 1, which accounts for subset-to-subset variance; Holm across a track's entrants; t-based 95% CI; MDE and pre-registered labels (positive / negative / no detectable effect / inconclusive) (D-026, docs/preregistration.md). Null FWER checked by simulation: 0.039-0.044 in all 32 allowed cells. |
| pre-registration | None formal. Holm families were defined retrospectively, and the study is described as exploratory, with evaluation revised and runs added after initial results (§4.1). | docs/preregistration.md, written before any entrant result, to be tagged prereg-v1 by the maintainer. n/K/R/S are fixed from calibration by a rule written in advance. Later changes go to a numbered deviations log (D-032). |
| reproducibility | Code on GitHub (abstract). Released configs and seed lists: not described. | Every results JSON records git hash, config sha256, LeRobot version, exact commands, all seeds, wall-clock, peak GPU memory and host (CLAUDE.md rule 8). `robocurate reproduce` re-resolves a leaderboard cell. Pinned lock file. |

## Where each protocol stands

**Stricter here:**
- Placebo size: K >= 8 random subsets vs 1-3.
- Balanced seed design, plus a test that models subset variance (Welch on run means treats one subset draw as
  representative).
- Final-checkpoint-only from the outset.
- Disjoint, frozen seed blocks with an information barrier for cleaners.
- A written pre-registration with a deviations log.
- Simulation-checked null calibration of the primary test.
- Full provenance in every result.

**Equal:**
- Training run as the unit of analysis.
- Size-matched random subsets plus a full-data reference.
- Holm correction.
- 200 rollouts per checkpoint on the main tasks.
- Diffusion Policy among the policies.
- Treating CUPID-style influence as a method to test rather than a given.

**Covered by CuraEval and not here:**
- Real mixed-skill operator data with operator-skill labels.
- robomimic tasks, including bimanual transport.
- BC-RNN.
- Analysis of what curation scores actually measure (e.g. PSD collapsing to episode length, §4.2).
- Analysis of which demonstrations each selector keeps (§4.4).
- An explicit study of how conclusions move when the evaluation protocol changes (§4.5).

RoboCurate-Bench deliberately avoids inspecting what entrants select beyond diagnostics against hidden labels. It
fixes the evaluation protocol in advance rather than varying it.

The two are complementary. CuraEval shows on real mixed-skill data that apparent curation gains are fragile to
protocol choices. RoboCurate-Bench turns that lesson into a fixed, pre-registered, placebo-controlled protocol on
simulated mixtures with known defects.
