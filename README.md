# RoboCurate-Bench

**Does cleaning your robot data actually help?**

Many tools now promise to find the "bad" demonstrations in a robot dataset. RoboCurate-Bench tests whether
training on the episodes a cleaner keeps beats training on **size-matched random subsets of the same data**, the
placebo. Policies are trained under one fixed protocol and evaluated in simulation. The protocol is the product. A
fast wrong number is worse than no number.

> **Results: pending.** The software core and the launch entrants are built and tested on CPU. GPU calibration (M0)
> has not run yet. Every number you can produce today comes from the **fake backend** and is labeled FAKE.

## Protocol in plain language

- **Placebo, not just full data.** Every selection-track entrant is compared with K ≥ 8 random subsets of the same
  size, at the same episode budget (launch budget: keep 50%), and with full data. A cleaner that only "helps"
  because smaller data trains differently is caught by this comparison (D-002, D-019).
- **Training runs are the unit.** Each training run contributes one number: the success rate of its final
  checkpoint over ≥ 200 fixed eval rollouts. Rollouts are never pooled across runs (D-002, D-003).
- **Subset luck is part of the error bar.** A deterministic cleaner picks one subset, and subsets differ in quality.
  The test's standard error includes the subset-to-subset variance estimated from the placebo subsets, plus the
  seed variance, with df = K − 1. Holm correction across entrants (D-016, D-019).
- **Two tracks.**
  - *Selection* entrants choose episodes.
  - *Weighting* entrants (e.g. RA-BC) reweight training samples on the full data. Their placebo is the same
    weights shuffled across episodes (D-020).
- **Same training for everyone.** Same steps, hyperparameters and seeds for every condition. Normalization stats
  come from each condition's own training data (D-004, D-013).
- **No peeking.**
  - Cleaners never see ground-truth labels, eval code, eval seeds or eval results.
  - Cleaners that need rollouts get only curation seeds, under a cap of 100.
  - Selections are computed once, cached, and costed as "curation GPU-h" (D-007, D-008, D-022).
- **References, not entrants.**
  - A label oracle (uses the hidden labels) shows whether curation can help on a task at all (D-012).
  - Our own scorer, rcb-score v1, is frozen before any result (D-023).

Full rules: [CLAUDE.md](CLAUDE.md). Decisions: [DECISIONS.md](DECISIONS.md). Status: [PROGRESS.md](PROGRESS.md).

## Architecture

```
configs/          protocol.yaml (seeds, budgets, test), tasks/, grids/, mixtures/, calibration/
src/robocurate/
  data/           Hub loader, label-free DatasetView, mixtures (+ policy-rollout source), defects, validate
  cleaners/       Cleaner API + registry, conformance (`submit`), import guard, entrants/
  curation/       prepare cache, rollout service (curation seeds only), reference policy, progress models
  train/ eval/    LeRobot command builders (lerobot-train / lerobot-eval; eval freezes at eval-v1)
  runs/           planner, locked idempotent executor, fake + local backends, oracle, reproduce
  report/         statistics (D-016/D-019), report.md, plot, static leaderboard site
```

Entrants:

| entrant | track | kind |
|---|---|---|
| `lerobot_doctor` | selection | wrapped checker |
| `kinema` | selection | wrapped checker |
| `roboaudit` | selection | wrapped checker |
| `rda` | selection | wrapped checker |
| `robometer_filter` | selection | wrapped (LeRobot Robometer reward model) |
| `rabc` | weighting | wrapped (LeRobot RA-BC) |
| `demo_score` | selection | reimplementation |
| `deminf` | selection | reimplementation |
| `cupid` | selection | reimplementation — **disabled (failed toybed sanity on ACT-style and diffusion toybeds, D-029, D-036)**; refused in grids, excluded from the leaderboard |
| `gradsim_return` | selection | original (D-036; CUPID objective + TracIn similarity) — enabled for diffusion only (toybed AUROC 1.000; ACT-style 0.383) |
| `rcb_score` | selection | ours, v1 frozen |

Baselines are random subsets, full data and the oracle.

## Quickstart (CPU, fake backend)

```bash
conda create -n rcb python=3.12 && conda activate rcb
pip install -r requirements.lock && pip install -e ".[dev]" --no-deps
cp scripts/env.example.sh scripts/env.sh && source scripts/env.sh
robocurate plan configs/grids/task_a_v0.yaml
robocurate prepare configs/grids/task_a_v0.yaml --backend fake
robocurate run configs/grids/task_a_v0.yaml --backend fake --fake-params configs/fake/e2e_test.yaml
robocurate report "$ROBOCURATE_SCRATCH/results/task_a_v0-fake" --site out/site
robocurate reproduce <cell-id> --results-dir "$ROBOCURATE_SCRATCH/results/task_a_v0-fake" --backend fake \
  --selections-dir "$ROBOCURATE_SCRATCH/selections/task_a_v0-fake"
robocurate submit examples/cleaner_template/src/rcb_template_cleaner/cleaner.py
```

To submit your own cleaner, see [docs/submit.md](docs/submit.md).

## Related work

CuraEval (Xu et al., arXiv 2610.38221, CoRL 2026 workshop) compares demonstration-curation methods with
size-matched random subsets and full data on real mixed-skill robomimic data, and finds that apparent gains shrink
or vanish when checkpoint reporting, rollout count and evaluation seeds change. RoboCurate-Bench shares its core
controls (training run as the unit of analysis, size-matched random subsets, full-data reference, Holm correction,
200 rollouts) and is stricter on design: K >= 8 placebo subsets with a balanced test that models subset variance,
final checkpoints only, frozen disjoint seed blocks and a written pre-registration. CuraEval covers what we do not:
real operator-skill variation, robomimic tasks, BC-RNN, and analyses of what curation scores measure; see
[docs/related_work.md](docs/related_work.md).

## Limitations

- **Simulation only.** Real-robot curation effects may differ.
- **One policy class per task** (ACT on ALOHA transfer-cube for Task A). A cleaner can help one architecture and not
  another.
- **Single-mixture subset-variance floor.** A cleaner applied to one mixture yields one subset. More seeds cannot
  remove that variance; only more random subsets (K) and mixture replicates (R) shrink it. The MDE at the launch
  design may be large (Part 1 simulations: ≤ 0.32 power for a 15 pp effect once the subset SD reaches 0.04).
- **Balanced designs only.** The primary test (D-026) is exact only when entrant and placebo runs use the same seed
  count n ≥ 3 and K ≥ 8 random subsets; other designs are labelled NOT RESULT-ELIGIBLE.
- **Reward-model coverage.** Robometer-based entrants depend on one reward model and its revision. The mock used
  in tests says nothing about it.
- **Checker coverage.**
  - Wrapped checkers pass conformance and the toybed check on CPU; they have not run on a benchmark mixture yet.
  - trajlens and retake are not adapted (dependency conflicts / external API use).
  - lerobot-doctor reports at most 20 flagged episodes.
- **Reimplementations.** CUPID, DemInf and Demo-SCORE are reimplementations with declared adaptations.
- **Influence-based curation with unimodal policies (D-029).** With an ACT-style unimodal (MSE) policy, first-order
  influence did not separate planted wrong-direction demos from clean ones on our toybed (AUROC ≈ 0.5, even with an
  exact Hessian and 4,000 rollouts), although those demos halve closed-loop success. This is a limitation of the
  method family on unimodal policies, not a verdict on CUPID. CUPID was also tested in its native setting, a toy
  diffusion policy with the official "square" output, 64 timesteps, d = 4000 and lambda = 0: AUROC 0.51–0.52 over
  3 seeds, below the 0.8 pass bar, with a near-singular TRAK Gram matrix. It therefore stays disabled on every task.
  Evidence: [docs/findings_cupid.md](docs/findings_cupid.md). The pre-registered influence entrant is
  gradsim-return (D-036): CUPID's return-weighted objective with plain gradient similarity (TracIn); toybed AUROC
  1.000 on diffusion (enabled for diffusion tasks) and 0.383 on the ACT-style toybed (not enabled for ACT).

## License

Apache-2.0 (D-038; [LICENSE](LICENSE), the canonical text from apache.org), matching LeRobot. Wrapped optional
tools (the `[checkers]` extra) keep their own licenses and are never vendored.

## Credits

Verified on 2026-11-19 against the arXiv API, the venue pages, Crossref or PyPI. BibTeX:
[docs/references.bib](docs/references.bib).

- **LeRobot**: Cadene et al., "LeRobot: An Open-Source Library for End-to-End Robot Learning", ICLR 2026, arXiv
  2602.22818. Pinned at 0.6.1; it provides the datasets, ACT, Diffusion Policy, training/eval, Robometer and RA-BC.
  https://github.com/huggingface/lerobot
- **CUPID**: Agia et al., "CUPID: Curating Data your Robot Loves with Influence Functions", CoRL 2025, arXiv
  2506.19121. Reference code: github.com/agiachris/cupid (MIT).
- **DemInf**: Hejna et al., "Robot Data Curation with Mutual Information Estimators", RSS 2025, arXiv 2502.08623.
  Reference code: github.com/jhejna/demonstration-information (MIT).
- **Demo-SCORE**: Chen, Lessing, Liu, Finn, "Curating Demonstrations using Online Experience", RSS XXI (2025),
  DOI 10.15607/RSS.2025.XXI.071, arXiv 2503.03707. Reference code: github.com/alessing/demo-score (no license
  file; not redistributed).
- **QoQ**: Lee et al., "Quality over Quantity: Demonstration Curation via Influence Functions for Data-Centric Robot
  Learning", ICRA 2026 (accepted, per arXiv), arXiv 2603.09056. Not implemented. TODO: unverified official code
  link (none found).
- **DataMIL**: Dass et al., "DataMIL: Selecting Data for Robot Imitation Learning with Datamodels", ICLR 2026,
  arXiv 2505.09603. Not implemented. Code: github.com/UT-Austin-RobIn/datamil (MIT).
- **CuraEval**: Xu, Chen, Ye, Shi, "CuraEval: Evaluating Demonstration Curation on Mixed-Skill Robot Data", CoRL
  2026 Workshop "Oops, I Erred", arXiv 2610.38221. Code: github.com/jiuyixu25/CuraEval. Closest related work:
  it compares curation methods with size-matched random subsets and full-data training on mixed-skill robomimic
  data with Diffusion Policy and BC-RNN, and reports that apparent gains shrink when checkpoint reporting, rollout
  count and evaluation seeds change together.
- **TracIn**: Pruthi, Liu, Sundararajan, Kale, "Estimating Training Data Influence by Tracing Gradient Descent",
  NeurIPS 2020, arXiv 2002.08484. gradsim-return (D-036) uses its gradient-similarity comparison with CUPID's
  return-weighted rollout objective.
- **RA-BC / SARM**: Chen et al., "SARM: Stage-Aware Reward Modeling for Long Horizon Robot Manipulation", arXiv
  2509.25358 (Eq. 8-9 as implemented in lerobot `rewards/sarm/rabc.py`).
- **Robometer**: Liang et al., "Robometer: Scaling General-Purpose Robotic Reward Models via Trajectory
  Comparisons", arXiv 2603.02115. Used through lerobot `rewards/robometer` (`lerobot/Robometer-4B`).
- **Checkers** (versions and licenses from PyPI):
  - lerobot-doctor 0.2.0 (Apache-2.0).
  - kinema 0.1.0 (Apache-2.0; 0.1.1 exists).
  - roboaudit 0.3.1 (MIT).
  - Robot Data Audit (RDA), github.com/liesliy/rda @47c69459 = PyPI robot-data-audit 0.12.5 (MIT).
- **KSG estimator**: Kraskov, Stögbauer, Grassberger, "Estimating mutual information", Phys. Rev. E 69, 066138
  (2004), DOI 10.1103/PhysRevE.69.066138.
