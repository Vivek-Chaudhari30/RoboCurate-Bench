# Decision log

D-001 Build on LeRobot, exact-pinned (0.6.x). Why: required; mature train/eval/reward APIs. Rejected: robomimic codebase (better comparability with prior work, but reimplements LeRobot).
D-002 Unit of analysis = training run. Primary test: Welch t-test, cleaner runs vs pooled placebo runs (all subsets x seeds), same budget; Holm correction across cleaners within a task; bootstrap CI on the difference as secondary. Rejected: pooled-rollout binomial tests (pseudoreplication).
D-003 Results use only the final checkpoint at a fixed step count. Rejected: best-of-N checkpoint (selection bias; the failure mode reported by prior curation studies).
D-004 Fixed gradient-step count across all budgets and conditions. Rejected: fixed epochs (confounds compute with data size).
D-005 Seed count set by power analysis on M0-measured sigma, before any cleaner result. Floor 3. Rejected: fixed 3 seeds (MDE likely ~20pp, above most published effects).
D-006 Launch budget: keep 50% of episodes. 25% is a stretch goal if compute allows. Rejected: budget sweep (compute).
D-007 Eval seeds live only in configs/protocol.yaml. Cleaners that need rollouts (CUPID, Demo-SCORE) get a disjoint curation_seed_range and a declared rollout cap (100 per task, matching CUPID's simulation setting). Rejected: shared seed pool (eval leakage).
D-008 Ground-truth quality labels live in sidecar files that cleaners never receive. Used only for diagnostic precision/recall in reports.
D-009 Task A = ALOHA sim transfer-cube + ACT, human demos for M0 calibration. Rejected for M0: PushT + Diffusion (likely costlier training; revisit as Task B).
D-010 Package robocurate-bench, import name robocurate, CLI robocurate (pending PyPI check).
