# PROGRESS

## Status 2026-11-19 (D-026 task, branch m1-core)

Commits on m1-core (pushed): `0003aa3` D-026 test + refusals + tables; `2fec523` D-025 placebo; `cb3a8d2` CUPID
outcome (c); `33c196f` checkers installed; `d8e593e` eligibility labels; `896b22e` test-order fix; plus this
summary. m0-calibration untouched (Step 3 skipped). GPU-h 0 (CPU only, Mac).

- **Step 1 — D-026: done.** All 32 result-eligible FPR cells pass (max upper 95% 0.0451, MC SE <= 0.0005); fake
  e2e K = 8, n = 3: oracle +0.197 (p = 0.0068) significant, length_heuristic −0.003 (p_holm 0.957) not. Refusals
  for n_c != n_p, n < 3, K < 8 (subsets and permutations) unit-tested; protocol loader floor 3.
- **Step 2 — scratch: outcome (c), B-006.** The lab wiki's scratch convention is local `/disk/u/<user>`; hokkaido
  has no `/disk`. No directory created, no D-027. Candidates under B-006.
- **Step 3 — skipped** (Step 2 unresolved). rcb-m0 does not exist. Hokkaido 15:10 UTC (read-only): GPU 82% /
  27.6 GB used by another user's ~7-day process; no tmux server for this user.
- **Step 4 — D-025: done (transform + tests + prepare artifacts); training path blocked, B-005.** LeRobot 0.6.1
  takes RA-BC progress, not weights; the local backend refuses weight_placebo runs until a path is decided.
- **Step 5 — checkers: done.** 4 installed + pinned (+ lock), all PASS `robocurate submit` and toybed sanity;
  trajlens and retake skipped (reasons in the table).
- **Step 6 — CUPID: outcome (c).** Disabled ("disabled — failed sanity"), refused in grids, excluded from the
  leaderboard; LOO test confirms the influence formula; reference "square" output does not port to an MLP policy.
- **Step 7 — labels: done.** "NOT RESULT-ELIGIBLE (design below protocol minimum)" on json/md/site/CLI.
- Tests: default 227 passed + 1 xfailed (CUPID); slow 15 passed (incl. the hard D-026 null calibration, checker conformance, DemInf sanity, help-flag test); ruff clean.

**Decisions needed:** (1) B-006 scratch path (or provision `/disk` on hokkaido); (2) B-005 how the D-025 placebo
reaches lerobot-train; (3) confirm the protocol.yaml edits made for D-026 (`test: d026_balanced`; seed-floor
comments 6 -> 3); (4) `random_subset_seeds` lists 3 seeds: K >= 8 result runs need 8 (e.g. 2000–2007) and the M0
grid's K = 4 needs 2000–2003; (5) CUPID: build the diffusion-policy toybed variant (outcome b) or leave disabled.

## Non-GPU completion log (overnight 2, branch m1-core, CPU only, no MPS, no hokkaido; GPU-h 0)

Start 2026-11-19T16:32Z (`caffeinate -dims -t 43200`). Env: rcb (py3.12, lerobot 0.6.1), `source scripts/env.sh`,
`CUDA_VISIBLE_DEVICES=""`, `PYTORCH_ENABLE_MPS_FALLBACK=0`, device=cpu forced in every new code path.

| UTC | item | what changed | tests | outcome |
|---|---|---|---|---|
| 2026-11-19T16:34Z | 1 decisions | D-027..D-032 appended verbatim to DECISIONS.md; protocol.yaml random_subset_seeds 2000–2011, weight_permutation_seeds 3000–3011 (approved); test:d026_balanced and floor-3 comments kept; two tests updated for the new lists | default 227 passed + 1 xfailed; ruff clean | done |
| 2026-11-19T16:37Z | 2 D-028 | weights.py: permute_progress_curves replaces permute_weight_curves/write_weights_parquet (no rescale); prepare writes perm{i}_progress.parquet (RA-BC schema, rabc.py:101-126) and records placebo_progress_parquets, weighting_placebo D-028; local backend placebo refusal removed (permutation subset_index -> its parquet); CLI print updated. Tests: shapes, identity exact, seeded, resample/no-rescale, golden rabc vs placebo cmd differ only in progress path, LeRobot make_sample_weighter/RABCWeights (utils/sample_weighting.py:103-185, rabc.py:175-233) accept placebo files with finite weights. B-005 and B-006 closed. | default 227 passed + 1 xfailed; test_curation 23 passed; ruff clean | done |
<!-- ng-log -->

## Previous status (B-003/M0-launch task, STOPPED at Step 1, B-004; superseded by D-026)

Branch `m1-core`. D-024 is recorded and implemented (result mode refuses n_p < 2). The hard FPR test, run on the
result-eligible designs of the M0 power table (K in {8, 12}, n_p = 2, n_c in {3, 5}), **fails 6 eligible cells, all
with n_c = 5** (upper 95% up to 0.0773 > 0.06). Stop condition: "If any allowed cell fails the bound, STOP and report
the table. Do not adjust the test." Steps 2–7 were not started: m0-calibration was not fast-forwarded, nothing was
launched on hokkaido, no checker was installed. GPU-h 0.

Hokkaido read-only check (13:51 UTC) found two further Step 3 blockers, independent of B-004:
- `/disk/u/vivek` does not exist (there is no `/disk` mount). Candidates, NOT chosen: `/share/datasets` (NFS, 42 T
  free), `/share/NFS/...` outside `$HOME` (NFS, 32 T free; `$HOME` = `/share/NFS/u/vivek`), local NVMe `/` (98% full,
  52 G free).
- GPU busy: one process of another user, 27 556 MiB, 58% utilization, running for ~7 days. The 2 h poll limit would
  very likely expire.

Tests (Mac, CPU, `dba6618`): default incl. integration 203 passed + 1 xfailed (CUPID); slow 10 passed, 1 failed
(`test_primary_test_null_calibration`, B-004, intended; `test_flags_exist_in_help_or_dataclasses` needs the rcb env's
bin on PATH and passes with it); ruff clean.

See B-004 under "## Blocked" for the table, the cause, and the options.

## Morning summary (overnight m1-core)

Branch `m1-core` (from `m0-calibration` @ `0218b8c`), pushed after every item. Overnight run 2026-11-19
06:19–07:30 UTC, about 1.2 h of the 10 h budget; CPU only; GPU-h 0. m0-calibration and main were not touched.

**Items: 10 of 10 done, 0 blocked.** Two acceptance gaps are inherited or depend on a permission, listed below:
the slow suite has one failing test (B-003), and the checker adapters have never run because the tools were never installed.

| item | commit |
|---|---|
| 1 decisions D-020..D-023, protocol additions, weighting-track config | `c97155c` |
| 2 D-021 tie-break; mixture shuffle verified | `4fad1bd` |
| 3 curation infra (prepare cache, rollout service, reference policy, RA-BC wiring) | `bcf182f` |
| 4 toybed | `909d60c`, `63f3eb4` |
| 5 entrants | `1052270` |
| 6 policy-rollout mixture source | `d1f983c` |
| 7 validate-defects + docs/defect_validation.md | `90fec7f` |
| 8 D-019 report, `report --site`, `reproduce`, CI | `c5fa733` |
| 9 cleaner template, docs/submit.md | `c8a5f06`, `02567e5` |
| 10 README draft, docs/task_b_readiness.md | `7614d81` |

**Tests (Mac, CPU):**
- default (`not gpu and not slow and not integration`, = CI): **197 passed, 1 xfailed**
- integration: **3 passed**
- slow: **10 passed, 1 failed** (12 min). The failure is `test_primary_test_null_calibration`: B-003, the K=8, n_p=1 cell, inherited and intended. It includes the DemInf sanity + conformance tests, the template install test and the help-flag test.
- ruff: clean
- import-graph tests: cleaners/ and curation/ cannot reach eval/, labels or oracle (pass)

**Entrants:**

| entrant | track | impl. | source | resources | toybed sanity (published defaults) |
|---|---|---|---|---|---|
| lerobot_doctor | selection | wrapped | PyPI 0.2.0 | dataset_root | not run (tool not installed); mapping unit-tested |
| kinema | selection | wrapped | PyPI 0.1.0 | dataset_root | not run (tool not installed); mapping unit-tested |
| roboaudit | selection | wrapped | PyPI 0.3.1 | dataset_root | not run (tool not installed); mapping unit-tested |
| rda | selection | wrapped | liesliy/rda@47c6945 | dataset_root | not run (tool not installed); mapping unit-tested |
| robometer_filter | selection | wrapped (LeRobot) | rewards/robometer | gpu | pass (mock progress) |
| rabc | weighting | wrapped (LeRobot RA-BC) | rewards/sarm/rabc.py | gpu | pass (mock progress) |
| rcb_score v1 | selection | original, frozen (digest test) | D-023 | gpu | pass (mock progress) |
| demo_score | selection | reimplementation | arXiv 2503.03707 §IV | 100 rollouts, ref policy | pass: keeps 2/8 wrong-direction vs 18/30 clean |
| deminf | selection | reimplementation | arXiv 2502.08623 §V | none | pass (slow: 222 s at 50k VAE steps): keeps 1/8 wrong, 0/6 jitter, 18/30 clean, and **6/6 pause** |
| cupid | selection | reimplementation | arXiv 2506.19121 Def. 1, Eq. 2-5 | 100 rollouts, ref policy | **FAILS** with defaults (keeps 5/8 wrong); strict xfail; plumbing tests pass |

`submit` conformance passes for robometer_filter, rabc, rcb_score, demo_score and cupid (default suite), and for
deminf (slow). The checkers cannot pass `submit` until installed and given a dataset_root.

**Checker availability:**

| tool | where | license | installs? | adapter |
|---|---|---|---|---|
| lerobot-doctor | PyPI 0.2.0 | Apache-2.0 | dry-run OK (+rich, markdown-it-py, mdurl) | yes. Reports at most 20 flagged episodes, lowest indices first: an index-order bias |
| kinema | PyPI 0.1.0 | Apache-2.0 | dry-run OK | yes (FAIL→2, WARN→1, count of WARN+) |
| roboaudit | PyPI 0.3.1 | MIT | dry-run OK, no new deps | yes (exclude/review/keep) |
| RDA | GitHub liesliy/rda@47c6945 | MIT | dry-run OK, no new deps | yes (EXCLUDE/REVIEW/UNVERIFIABLE/PASS, combined_score) |
| trajlens | PyPI 0.4.0 | Apache-2.0 | conflicts: needs pyarrow<25 (lock 26.0.1) | no |
| retake | GitHub TheNandinee/retake (PyPI name is an unrelated package) | Apache-2.0 | would upgrade numpy/pandas/huggingface_hub/pillow; calls external LLM APIs | no |

Common mapping: flagged episodes rank last, then the tool's own score, ties per D-021.

**Defect validation** (5 pinned SO-100/101 datasets, 331 episodes, parquet only):
- Interior pauses appear in 6–90% of episodes, median segment 0.6–1.2 s. Ours: 11% of episodes at 0.5–1.5 s.
  Our lengths are realistic; our rate is low.
- Duplicates: 0 exact and 0 near-duplicates observed. Ours: 7%.
- Jitter above 0.10 occurs only in one dataset (15% of its episodes).
- Timestamp gaps in 22/150 episodes of one dataset.
- All five datasets are single-task.

**hokkaido (read-only, start and end):** no tmux session for this user, so `rcb-m0` does not exist. M0 was never
launched: the previous task stopped at B-003. There is no M0 log path. The GPU was busy with another user's job
(start: 66% / 27.6 GB; end: 100% / 27.6 GB).

**Needs Vivek's decision:**
1. **B-003 (still open):** require n_p ≥ 2 for result runs, or build the parametric-bootstrap fallback. Until then
   the slow suite has one failing test by design. M0 is still not launched.
2. **Checker install:** installing `.[checkers]` into the shared env was blocked by the permission system. Run
   `pip install -e ".[checkers]" --no-deps` plus the 3 small deps, or approve it, then run the adapters once on a real
   dataset.
3. **Weighting placebo interpretation (D-020):** RA-BC weights are per frame, so we permute each episode's
   progress *curve* across episodes, time-resampled. Confirm.
4. **Weighting-track Holm family** is separate from the selection track. Confirm.
5. **rcb-score v1 uses final-frame Robometer progress, not success probability.** Lerobot 0.6.1 exposes only a
   0.5-thresholded success. It is frozen as v1 in code. If you want success probability, that is v2 (D-023).
6. **CUPID fails its toy sanity check with defaults.** On the toy, 786 training samples < d = 4000, so H = ΦᵀΦ is
   singular. Influence also comes from 100 stochastic rollouts. Keep it as an entrant as-is (reported), or decide on
   an investigation (no tuning)?
7. **Declared adaptations** for Demo-SCORE (one reference checkpoint, 80/20 rollout split, top-k instead of a
   threshold, no rollout augmentation) and CUPID (stochastic rollout policy with σ = RMS BC residual). Accept?
8. **The report now uses D-019 as primary** (D-019 is accepted; its n_p = 1 calibration gap is B-003). OK?

**Next step:** decide B-003. Then launch M0 (Part 2–4 of the M0 task) when hokkaido's GPU is free.

**Current milestone: M0 — test calibrated (D-026, all 32 eligible cells pass); BLOCKED on the hokkaido scratch path (B-006); GPU busy; GPU work not started**

## D-026 calibration and fake end-to-end (Step 1 of the D-026 task)

Primary test (D-026): t = (mean_c − mean_p) / sqrt((1 + 1/K) · var(subset means)), df = K − 1, balanced
n_c = n_p = n >= 3, K >= 8, Holm across the entrants of a track (`stats.balanced_test`, `balanced_test_arrays`,
`compare_entrants_d026`; power/MDE `balanced_power`, `balanced_mde`: exact noncentral t with
nc = effect / sqrt((1 + 1/K)(sigma_sub^2 + sigma_run^2/n)/R), df = R(K − 1), Bonferroni alpha/m). The D-016/D-019
variance-components test and Welch are attached to every comparison as labeled secondary diagnostics.
Result mode refuses: K < 8 random subsets (`check_result_subsets`), and in `check_result_design` any unbalanced
design (seeds per unit differ), n < 3, or < 8 weight permutations. Protocol loader floor: 3 seeds (D-015's 6
retired); `protocol.test` must be `d026_balanced`.

Null calibration: 7 null entrants, 200 rollouts, base 0.5, 200 000 grids per cell (`python scripts/null_grid.py`,
57 s; hard slow test `test_primary_test_null_calibration`). All 32 result-eligible cells pass (max upper 95% bound
0.0451; FWER 0.039–0.044 in every cell, as expected for an exact test with Holm over 7 correlated tests). MC power (Holm) and
analytic power (Bonferroni, conservative) agree to <= 0.010. The unit test `test_balanced_test_is_exact_under_normal_model` checks the single
test's size is 0.05 within 4 MC SE (200 000 grids, no clipping or binomial).

Result-eligible (D-026 test):

| sigma_sub | sigma_seed | K, n_p, n_c | test | eligibility | grids | FWER | MC SE | upper 95% | criterion | power 15pp (MC) | power 15pp (analytic) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.03 | 8, 3, 3 | d026 | result-eligible | 200000 | 0.0401 | 0.0004 | 0.0410 | upper95 <= 0.06 ok | 0.884 | 0.880 |
| 0.00 | 0.03 | 8, 5, 5 | d026 | result-eligible | 200000 | 0.0397 | 0.0004 | 0.0406 | upper95 <= 0.06 ok | 0.987 | 0.987 |
| 0.00 | 0.03 | 12, 3, 3 | d026 | result-eligible | 200000 | 0.0432 | 0.0005 | 0.0440 | upper95 <= 0.06 ok | 0.965 | 0.961 |
| 0.00 | 0.03 | 12, 5, 5 | d026 | result-eligible | 200000 | 0.0428 | 0.0005 | 0.0437 | upper95 <= 0.06 ok | 0.999 | 0.999 |
| 0.00 | 0.06 | 8, 3, 3 | d026 | result-eligible | 200000 | 0.0398 | 0.0004 | 0.0406 | upper95 <= 0.06 ok | 0.483 | 0.477 |
| 0.00 | 0.06 | 8, 5, 5 | d026 | result-eligible | 200000 | 0.0391 | 0.0004 | 0.0399 | upper95 <= 0.06 ok | 0.747 | 0.747 |
| 0.00 | 0.06 | 12, 3, 3 | d026 | result-eligible | 200000 | 0.0424 | 0.0005 | 0.0433 | upper95 <= 0.06 ok | 0.624 | 0.620 |
| 0.00 | 0.06 | 12, 5, 5 | d026 | result-eligible | 200000 | 0.0430 | 0.0005 | 0.0439 | upper95 <= 0.06 ok | 0.876 | 0.876 |
| 0.02 | 0.03 | 8, 3, 3 | d026 | result-eligible | 200000 | 0.0394 | 0.0004 | 0.0403 | upper95 <= 0.06 ok | 0.683 | 0.673 |
| 0.02 | 0.03 | 8, 5, 5 | d026 | result-eligible | 200000 | 0.0402 | 0.0004 | 0.0410 | upper95 <= 0.06 ok | 0.825 | 0.821 |
| 0.02 | 0.03 | 12, 3, 3 | d026 | result-eligible | 200000 | 0.0429 | 0.0005 | 0.0438 | upper95 <= 0.06 ok | 0.819 | 0.815 |
| 0.02 | 0.03 | 12, 5, 5 | d026 | result-eligible | 200000 | 0.0436 | 0.0005 | 0.0445 | upper95 <= 0.06 ok | 0.929 | 0.927 |
| 0.02 | 0.06 | 8, 3, 3 | d026 | result-eligible | 200000 | 0.0401 | 0.0004 | 0.0409 | upper95 <= 0.06 ok | 0.382 | 0.374 |
| 0.02 | 0.06 | 8, 5, 5 | d026 | result-eligible | 200000 | 0.0399 | 0.0004 | 0.0408 | upper95 <= 0.06 ok | 0.569 | 0.563 |
| 0.02 | 0.06 | 12, 3, 3 | d026 | result-eligible | 200000 | 0.0431 | 0.0005 | 0.0440 | upper95 <= 0.06 ok | 0.501 | 0.499 |
| 0.02 | 0.06 | 12, 5, 5 | d026 | result-eligible | 200000 | 0.0441 | 0.0005 | 0.0450 | upper95 <= 0.06 ok | 0.717 | 0.711 |
| 0.04 | 0.03 | 8, 3, 3 | d026 | result-eligible | 200000 | 0.0400 | 0.0004 | 0.0408 | upper95 <= 0.06 ok | 0.319 | 0.318 |
| 0.04 | 0.03 | 8, 5, 5 | d026 | result-eligible | 200000 | 0.0393 | 0.0004 | 0.0402 | upper95 <= 0.06 ok | 0.374 | 0.371 |
| 0.04 | 0.03 | 12, 3, 3 | d026 | result-eligible | 200000 | 0.0434 | 0.0005 | 0.0443 | upper95 <= 0.06 ok | 0.437 | 0.429 |
| 0.04 | 0.03 | 12, 5, 5 | d026 | result-eligible | 200000 | 0.0426 | 0.0005 | 0.0434 | upper95 <= 0.06 ok | 0.497 | 0.495 |
| 0.04 | 0.06 | 8, 3, 3 | d026 | result-eligible | 200000 | 0.0396 | 0.0004 | 0.0405 | upper95 <= 0.06 ok | 0.216 | 0.213 |
| 0.04 | 0.06 | 8, 5, 5 | d026 | result-eligible | 200000 | 0.0396 | 0.0004 | 0.0404 | upper95 <= 0.06 ok | 0.277 | 0.280 |
| 0.04 | 0.06 | 12, 3, 3 | d026 | result-eligible | 200000 | 0.0437 | 0.0005 | 0.0446 | upper95 <= 0.06 ok | 0.294 | 0.290 |
| 0.04 | 0.06 | 12, 5, 5 | d026 | result-eligible | 200000 | 0.0442 | 0.0005 | 0.0451 | upper95 <= 0.06 ok | 0.382 | 0.380 |
| 0.06 | 0.03 | 8, 3, 3 | d026 | result-eligible | 200000 | 0.0399 | 0.0004 | 0.0408 | upper95 <= 0.06 ok | 0.148 | 0.147 |
| 0.06 | 0.03 | 8, 5, 5 | d026 | result-eligible | 200000 | 0.0388 | 0.0004 | 0.0396 | upper95 <= 0.06 ok | 0.161 | 0.160 |
| 0.06 | 0.03 | 12, 3, 3 | d026 | result-eligible | 200000 | 0.0431 | 0.0005 | 0.0440 | upper95 <= 0.06 ok | 0.204 | 0.200 |
| 0.06 | 0.03 | 12, 5, 5 | d026 | result-eligible | 200000 | 0.0435 | 0.0005 | 0.0444 | upper95 <= 0.06 ok | 0.220 | 0.218 |
| 0.06 | 0.06 | 8, 3, 3 | d026 | result-eligible | 200000 | 0.0397 | 0.0004 | 0.0405 | upper95 <= 0.06 ok | 0.120 | 0.116 |
| 0.06 | 0.06 | 8, 5, 5 | d026 | result-eligible | 200000 | 0.0395 | 0.0004 | 0.0404 | upper95 <= 0.06 ok | 0.135 | 0.137 |
| 0.06 | 0.06 | 12, 3, 3 | d026 | result-eligible | 200000 | 0.0435 | 0.0005 | 0.0444 | upper95 <= 0.06 ok | 0.158 | 0.157 |
| 0.06 | 0.06 | 12, 5, 5 | d026 | result-eligible | 200000 | 0.0436 | 0.0005 | 0.0445 | upper95 <= 0.06 ok | 0.188 | 0.186 |

Informational, not result-eligible (old D-016/D-019 test, unbalanced):

| sigma_sub | sigma_seed | K, n_p, n_c | test | eligibility | grids | FWER | MC SE | upper 95% | criterion | power 15pp (MC) | power 15pp (analytic) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.03 | 8, 1, 3 | d016 | not result-eligible (informational) | 200000 | 0.0027 | 0.0001 | 0.0029 | — (no pass/fail) | 0.591 | 0.797 |
| 0.00 | 0.03 | 8, 2, 3 | d016 | not result-eligible (informational) | 200000 | 0.0024 | 0.0001 | 0.0026 | — (no pass/fail) | 0.767 | 0.860 |
| 0.00 | 0.03 | 12, 2, 3 | d016 | not result-eligible (informational) | 200000 | 0.0070 | 0.0002 | 0.0074 | — (no pass/fail) | 0.917 | 0.954 |
| 0.00 | 0.03 | 8, 2, 5 | d016 | not result-eligible (informational) | 200000 | 0.0016 | 0.0001 | 0.0018 | — (no pass/fail) | 0.896 | 0.970 |
| 0.00 | 0.03 | 12, 2, 5 | d016 | not result-eligible (informational) | 200000 | 0.0057 | 0.0002 | 0.0061 | — (no pass/fail) | 0.976 | 0.997 |
| 0.00 | 0.06 | 8, 1, 3 | d016 | not result-eligible (informational) | 200000 | 0.0026 | 0.0001 | 0.0028 | — (no pass/fail) | 0.183 | 0.383 |
| 0.00 | 0.06 | 8, 2, 3 | d016 | not result-eligible (informational) | 200000 | 0.0023 | 0.0001 | 0.0025 | — (no pass/fail) | 0.269 | 0.451 |
| 0.00 | 0.06 | 12, 2, 3 | d016 | not result-eligible (informational) | 200000 | 0.0067 | 0.0002 | 0.0070 | — (no pass/fail) | 0.456 | 0.599 |
| 0.00 | 0.06 | 8, 2, 5 | d016 | not result-eligible (informational) | 200000 | 0.0016 | 0.0001 | 0.0018 | — (no pass/fail) | 0.476 | 0.666 |
| 0.00 | 0.06 | 12, 2, 5 | d016 | not result-eligible (informational) | 200000 | 0.0055 | 0.0002 | 0.0058 | — (no pass/fail) | 0.690 | 0.830 |
| 0.02 | 0.03 | 8, 1, 3 | d016 | not result-eligible (informational) | 200000 | 0.0105 | 0.0002 | 0.0109 | — (no pass/fail) | 0.492 | 0.601 |
| 0.02 | 0.03 | 8, 2, 3 | d016 | not result-eligible (informational) | 200000 | 0.0100 | 0.0002 | 0.0104 | — (no pass/fail) | 0.610 | 0.654 |
| 0.02 | 0.03 | 12, 2, 3 | d016 | not result-eligible (informational) | 200000 | 0.0221 | 0.0003 | 0.0228 | — (no pass/fail) | 0.781 | 0.804 |
| 0.02 | 0.03 | 8, 2, 5 | d016 | not result-eligible (informational) | 200000 | 0.0157 | 0.0003 | 0.0163 | — (no pass/fail) | 0.734 | 0.783 |
| 0.02 | 0.03 | 12, 2, 5 | d016 | not result-eligible (informational) | 200000 | 0.0367 | 0.0004 | 0.0375 | — (no pass/fail) | 0.866 | 0.910 |
| 0.02 | 0.06 | 8, 1, 3 | d016 | not result-eligible (informational) | 200000 | 0.0054 | 0.0002 | 0.0058 | — (no pass/fail) | 0.174 | 0.308 |
| 0.02 | 0.06 | 8, 2, 3 | d016 | not result-eligible (informational) | 200000 | 0.0052 | 0.0002 | 0.0055 | — (no pass/fail) | 0.240 | 0.356 |
| 0.02 | 0.06 | 12, 2, 3 | d016 | not result-eligible (informational) | 200000 | 0.0133 | 0.0003 | 0.0138 | — (no pass/fail) | 0.399 | 0.483 |
| 0.02 | 0.06 | 8, 2, 5 | d016 | not result-eligible (informational) | 200000 | 0.0061 | 0.0002 | 0.0065 | — (no pass/fail) | 0.413 | 0.505 |
| 0.02 | 0.06 | 12, 2, 5 | d016 | not result-eligible (informational) | 200000 | 0.0180 | 0.0003 | 0.0186 | — (no pass/fail) | 0.590 | 0.668 |
| 0.04 | 0.03 | 8, 1, 3 | d016 | not result-eligible (informational) | 200000 | 0.0433 | 0.0005 | 0.0442 | — (no pass/fail) | 0.312 | 0.293 |
| 0.04 | 0.03 | 8, 2, 3 | d016 | not result-eligible (informational) | 200000 | 0.0362 | 0.0004 | 0.0370 | — (no pass/fail) | 0.324 | 0.311 |
| 0.04 | 0.03 | 12, 2, 3 | d016 | not result-eligible (informational) | 200000 | 0.0507 | 0.0005 | 0.0517 | — (no pass/fail) | 0.436 | 0.423 |
| 0.04 | 0.03 | 8, 2, 5 | d016 | not result-eligible (informational) | 200000 | 0.0612 | 0.0005 | 0.0622 | — (no pass/fail) | 0.390 | 0.357 |
| 0.04 | 0.03 | 12, 2, 5 | d016 | not result-eligible (informational) | 200000 | 0.0761 | 0.0006 | 0.0773 | — (no pass/fail) | 0.502 | 0.483 |
| 0.04 | 0.06 | 8, 1, 3 | d016 | not result-eligible (informational) | 200000 | 0.0178 | 0.0003 | 0.0184 | — (no pass/fail) | 0.159 | 0.186 |
| 0.04 | 0.06 | 8, 2, 3 | d016 | not result-eligible (informational) | 200000 | 0.0174 | 0.0003 | 0.0179 | — (no pass/fail) | 0.182 | 0.205 |
| 0.04 | 0.06 | 12, 2, 3 | d016 | not result-eligible (informational) | 200000 | 0.0341 | 0.0004 | 0.0349 | — (no pass/fail) | 0.276 | 0.283 |
| 0.04 | 0.06 | 8, 2, 5 | d016 | not result-eligible (informational) | 200000 | 0.0326 | 0.0004 | 0.0334 | — (no pass/fail) | 0.278 | 0.260 |
| 0.04 | 0.06 | 12, 2, 5 | d016 | not result-eligible (informational) | 200000 | 0.0586 | 0.0005 | 0.0597 | — (no pass/fail) | 0.383 | 0.362 |
| 0.06 | 0.03 | 8, 1, 3 | d016 | not result-eligible (informational) | 200000 | 0.0699 | 0.0006 | 0.0711 | — (no pass/fail) | 0.179 | 0.141 |
| 0.06 | 0.03 | 8, 2, 3 | d016 | not result-eligible (informational) | 200000 | 0.0478 | 0.0005 | 0.0487 | — (no pass/fail) | 0.157 | 0.146 |
| 0.06 | 0.03 | 12, 2, 3 | d016 | not result-eligible (informational) | 200000 | 0.0520 | 0.0005 | 0.0529 | — (no pass/fail) | 0.209 | 0.198 |
| 0.06 | 0.03 | 8, 2, 5 | d016 | not result-eligible (informational) | 200000 | 0.0646 | 0.0005 | 0.0657 | — (no pass/fail) | 0.182 | 0.157 |
| 0.06 | 0.03 | 12, 2, 5 | d016 | not result-eligible (informational) | 200000 | 0.0643 | 0.0005 | 0.0654 | — (no pass/fail) | 0.235 | 0.215 |
| 0.06 | 0.06 | 8, 1, 3 | d016 | not result-eligible (informational) | 200000 | 0.0428 | 0.0005 | 0.0437 | — (no pass/fail) | 0.126 | 0.107 |
| 0.06 | 0.06 | 8, 2, 3 | d016 | not result-eligible (informational) | 200000 | 0.0365 | 0.0004 | 0.0373 | — (no pass/fail) | 0.122 | 0.114 |
| 0.06 | 0.06 | 12, 2, 3 | d016 | not result-eligible (informational) | 200000 | 0.0496 | 0.0005 | 0.0506 | — (no pass/fail) | 0.167 | 0.155 |
| 0.06 | 0.06 | 8, 2, 5 | d016 | not result-eligible (informational) | 200000 | 0.0618 | 0.0005 | 0.0628 | — (no pass/fail) | 0.169 | 0.131 |
| 0.06 | 0.06 | 12, 2, 5 | d016 | not result-eligible (informational) | 200000 | 0.0745 | 0.0006 | 0.0756 | — (no pass/fail) | 0.215 | 0.180 |

Fake end-to-end, K = 8, n = 3, oracle +0.25, length_heuristic 0 (configs in the scratchpad: protocol copy with 8
`random_subset_seeds` 2000–2007, grid copy with n_subsets 8, fake params with train_seeds [0, 1, 2], sleep 0; the
repo protocol still lists 3 subset seeds, see decisions):
```
rabc v1 [weighting] budget 1.0: progress + 8 permutation parquets; computed in 0.21s, GPU-h 0.000, rollouts 0, key a54930fab8b0
length_heuristic v1 [selection] budget 0.5: 25 episodes; computed in 0.00s, GPU-h 0.000, rollouts 0, key ad56eba12505
selections -> /Users/vivek/robocurate-scratch/selections/m1-e2e-d026
[done]    bc5c5f6fb94af264  oracle b=0.5 seed=2
done=60 skipped_existing=0 skipped_locked=0 reclaimed_stale_locks=0
site: /Users/vivek/robocurate-scratch/site-fake-d026/index.html
*** FAKE RESULTS (synthetic backend) ***
task aloha_transfer_cube budget 0.5 (selection):
  full                         runs=3   mean=0.528  cell 5c055b8f910e
  random                       runs=24  mean=0.500  cell e5267b2f86c9
  oracle                       runs=3   mean=0.697  cell b8be368d553c
  cleaner:length_heuristic     runs=3   mean=0.497  cell 186cc0bd36cb
  cleaner:length_heuristic vs placebo [D-026]: diff=-0.003 SE=0.052 df=7 p_holm=0.957 not significant   (Welch secondary p_holm=0.891)
  [ref] oracle vs placebo: diff=+0.197 p=0.00684 SIGNIFICANT
  [ref] full vs placebo: diff=+0.029 p=0.459 not significant
  s_sub=0.037 s_seed=0.056 K=8 n_p=3 n_c=3  MDE (80% power): 0.170
task aloha_transfer_cube full data (weighting):
  full                         runs=3   mean=0.528  cell 5c055b8f910e
  weighting:rabc               runs=3   mean=0.495  cell a03b0f077b04
  weight_placebo:rabc          runs=24  mean=0.478  cell 167857559e44
  weighting:rabc vs placebo [D-026]: diff=+0.017 SE=0.039 df=7 p_holm=0.664 not significant   (Welch secondary p_holm=0.838)
  [ref] full vs placebo: diff=+0.051 p=0.244 not significant
  s_sub=0.000 (truncated) s_seed=0.065 K=8 n_p=3 n_c=3  MDE (80% power): 0.126
wrote /Users/vivek/robocurate-scratch/results/m1-e2e-d026/report.md, /Users/vivek/robocurate-scratch/results/m1-e2e-d026/leaderboard.json
cell b8be368d553c: task aloha_transfer_cube budget 0.5 condition {'kind': 'oracle'} (3 runs, kind fake)
git: 35e8b78d76 (dirty)
config <scratchpad>/e2e_d026/protocol_k8.yaml: sha256 8d28b83b7b42 ok
config /Users/vivek/Developer/CODES/RoboCurate-Bench/configs/tasks/aloha_transfer_cube.yaml: sha256 37eef6888ecb ok
config <scratchpad>/e2e_d026/grid_k8.yaml: sha256 6ac27745efb4 ok
config <scratchpad>/e2e_d026/fake_n3.yaml: sha256 3e3db93deabc ok
commands:
  robocurate prepare <scratchpad>/e2e_d026/grid_k8.yaml --backend fake --selections-dir <SELECTIONS>
  robocurate run <scratchpad>/e2e_d026/grid_k8.yaml --backend fake --fake-params <scratchpad>/e2e_d026/fake_n3.yaml --results-dir /Users/vivek/robocurate-scratch/results/m1-e2e-d026 --selections-dir /Users/vivek/robocurate-scratch/selections/m1-e2e-d026
re-executed: MATCH 3/3
```
Oracle (reference, D-026 test): +0.197, p = 0.0068, significant. length_heuristic: −0.003, p_holm 0.957, not
significant. rabc vs its 8 permutations: +0.017, p_holm 0.664. Full vs placebo: not significant. Reproduce MATCH 3/3.

## Report eligibility labels (Step 7 of the D-026 task)

Every selection budget and weighting panel gets `result_eligible` + `eligibility` reasons from
`report.design_eligibility` (K >= 8 placebo subsets/permutations; every condition and placebo unit with the same
run count n; n >= 3). Ineligible panels carry "NOT RESULT-ELIGIBLE (design below protocol minimum)" in
leaderboard.json, report.md, the site (a bordered banner per panel, in addition to the FAKE banner and watermark)
and the CLI output; `leaderboard.json` also has a top-level `result_eligible`. Tests:
`test_design_eligibility_rules`, `test_ineligible_designs_labeled_on_report_and_site`,
`test_balanced_k8_design_is_eligible`. Regenerated fake sites (scratch, not committed):
`$ROBOCURATE_SCRATCH/site-m1-e2e-d026/index.html` (K = 8, n = 3: both panels eligible, no label) and
`$ROBOCURATE_SCRATCH/site-m1-e2e-fake/index.html` (old K = 3 selection panel labeled "K = 3 < 8"; its weighting
panel, K = 8, n = 6 balanced, is eligible).

## Checker installs (Step 5 of the D-026 task)

Installed 2026-11-19 into the Mac rcb env with exact pins (`pip install` of the four pins below; before/after
`pip freeze` diff: + kinema, lerobot-doctor, roboaudit, robot-data-audit, rich 15.0.0, markdown-it-py 4.3.0,
mdurl 0.1.2; nothing upgraded or removed). requirements.lock regenerated (= `pip freeze --exclude-editable`, +7
lines). No sudo, system packages, Docker or GPU; each download < 1 MB. Sources checked against the PyPI JSON API
(project URLs) and installed-distribution metadata (license). Conformance = `robocurate submit <name>` (fake curation
context now writes the label-free view as a LeRobot dataset for cleaners that declare `requires_dataset_root`, via
the new labels-free `data/lerobot_write.py`). Sanity = real tool on the toybed LeRobot dataset (50 demos: 30 clean,
6 pause, 6 jitter, 8 wrong-direction), fraction kept at a 50% budget; criterion: some planted defect type kept less
than clean (tests `test_checker_toybed_sanity`, `test_checker_conformance_with_installed_tool`). Not tuned.

| checker | source | version | license | installed | conformance | sanity: kept clean / pause / jitter / wrong |
|---|---|---|---|---|---|---|
| lerobot-doctor | PyPI `lerobot-doctor` | 0.2.0 | Apache-2.0 | yes | PASS | 0.50 / 0.00 / 0.83 / 0.62 (pass) |
| kinema | PyPI `kinema` (links github.com/vedevpatel/kinema) | 0.1.0 (0.1.1 exists; adapter built for 0.1.0) | Apache-2.0 | yes | PASS | 0.63 / 0.00 / 0.00 / 0.75 (pass) |
| roboaudit | PyPI `roboaudit` | 0.3.1 | MIT | yes | PASS | 0.67 / 0.00 / 0.83 / 0.00 (pass; flags all 50 as at least "review") |
| RDA | github.com/liesliy/rda (the PyPI project `robot-data-audit` 0.12.5 links it) | commit 47c69459 (= 0.12.5) | MIT | yes | PASS | 0.63 / 0.00 / 1.00 / 0.00 (pass; ranks by score, no episode flagged) |
| trajlens | PyPI `trajlens` | 0.4.0 | Apache-2.0 | no | — | — (pyarrow<25 conflicts with the lock's 26.0.1; PyPI links a placeholder repo `yourusername/trajlens`) |
| retake | github.com/TheNandinee/retake (PyPI `retake` is an unrelated vector-DB package) | — | Apache-2.0 (overnight read) | no | — | — (upgrades core packages; calls external LLM APIs) |

## CUPID sanity investigation (Step 6 of the D-026 task) — outcome (c): unresolved, entrant disabled

Timebox 3 h; used ~1.2 h (CPU only). No hyperparameter of the entrant was changed. Diagnostics:
`scripts/diagnostics/cupid/{diag,premise,m_sweep,rank}.py` (some vary d or m as DIAGNOSTICS only; the entrant keeps
d = 4000, lambda = 0, m = 100).

- *Official code.* github.com/agiachris/cupid@2941eba9 exists but is built into the Diffusion Policy codebase on
  the `trak` library (not installed; not an approved dependency), so it was not run. Its TRAK files were read
  instead (raw GitHub, read-only): `scripts/train/train_trak.sh:309,311,315-316` (proj_dim 4000, lambda_reg 0.0,
  `loss_fn="square"` with `"ddpm"` commented out, num_timesteps 64); `diffusion_policy/data_attribution/
  modelout_functions.py:111-136` ("ddpm" = mse(pred, target); "square" = mean(pred^2) of the denoiser output;
  `get_out_to_loss_grad` = 1).
- *Divergence found.* Our docstring said the model output is "the per-sample BC loss (our reading of 'square')".
  That reading was wrong: "square" is the squared network prediction, not the loss. So our H = Phi^T Phi of LOSS
  gradients is an empirical Fisher (each sample weighted by its squared residual), not the reference's features.
  For a non-diffusion (MLP) policy the "square" analogue f = mean(mu(s)^2) ignores the action labels entirely
  (the denoiser sees the noisy action as input; an MLP mu(s) does not), so it cannot be ported faithfully to our
  toybed policy class.
- *Formula check.* New unit test `test_cupid_influence_matches_leave_one_out_refit`: with the exact Hessian, our
  influence (per-sample grads from `per_sample_projected_grads`, Eq. 2 sign, Eq. 3 "net" aggregation, factored out
  as `performance_influence`) predicts brute-force leave-one-out refits of a linear least-squares policy
  (corr > 0.99, sign agreement > 95%). The per-sample gradient test (vs autograd) already passed.
- *Premise holds on the toybed.* BC success over 300 toy seeds (3 train seeds): all data 0.39/0.43/0.40; without
  the 8 wrong-direction demos 0.75/0.83/0.77; clean only 1.0. The wrong demos are strongly harmful.
- *Influence does not find them under any variant* (50 demos, 800 frames, 100 rollouts, 41 successes; fraction of
  wrong vs clean demos kept at a 50% budget): ours (empirical Fisher, d = 4000) 0.625 vs 0.467; same, no
  projection 0.625 vs 0.5; EXACT Hessian of the training loss 0.625 vs 0.5; Gauss-Newton on mu(s) 0.625 vs 0.5;
  official "square" analogue 0.5 vs 0.533 (passes `wrong < clean` only by noise; its features ignore actions).
- *Not rollout variance.* m = 100 / 1000 / 4000 (3 reps each): AUC P(wrong ranked below clean) 0.65/0.47/0.42,
  0.60/0.64/0.53, 0.53/0.47/0.60 — no trend to 1.
- *Rank of Phi^T Phi.* N = 800 frames < d = 4000 (rank-deficient, pinv with lambda = 0). Diagnostic d = 400:
  wrong 0.375 vs clean 0.567 (AUC 0.667); 10x toybed (N ~ 7900 > d): AUC 0.49 / 0.55, but there 98–100 of 100
  rollouts succeed, so the +-1 contrast vanishes. Inconclusive.
- *Outcome (c).* Not a bug in the influence formula (LOO test), not a TRAK-approximation artifact (exact Hessian
  fails too), not rollout count; a toybed variant that satisfies CUPID's assumptions would need a diffusion policy
  (the "square" output depends on it), which does not fit the timebox. `Cupid.status = "disabled — failed sanity"`;
  `make_plan` refuses disabled entrants in any grid; `robocurate report` drops disabled entrants with an issue
  (tests `test_disabled_entrant_refused_in_grids_and_excluded_from_report`). It is not listed under CUPID's name.
- *Next step if wanted.* (b): a toy diffusion policy (DDPM loss, the reference "square" output, Q = 1) on the
  toybed, then require the sanity pass there.

## Blocked

- **B-006 (CLOSED by D-027: /share/datasets/robocurate; 2026-11-19T15:10Z, OPEN) — no scratch path on hokkaido; Step 2 outcome (c), Step 3 skipped.**
  - *Evidence (read-only).* `ssh hokkaido 'ls -ld /disk /disk/u; ls /disk; ...'`: "ls: cannot access '/disk': No such
    file or directory" (twice). Mounts: `/` = local NVMe `/dev/nvme0n1p2` (1.8 T, 98% used, 52 G free);
    `/share/NFS` (baushare, 104 T, 31 T free); `/share/datasets` (baunas24, 98 T, 42 T free). `$HOME` =
    `/share/NFS/u/vivek`. `id`: uid 1176, group `research`. `/share/projects`, `/share/u`, `/share/data/datasets` do
    not exist on hokkaido; `/share/NFS/projects` exists (`dr-xr-xr-x`, owner uid 1027: not writable for us);
    `/share/datasets` is `drwxrwxr-x root:research` (group-writable; 18 entries, other users' dirs inside).
  - *Wiki* (github.com/thebaulab/onramp.wiki @ ec6b6665, 2026-10-27). Filesystems-on-bau-clusters.md: "On some
    machines there is a local `/disk` directory ... `/disk/u/[username]` for your own personal directories";
    NFS-Storage-Administration.md "Where Should I Put My Files?": temporary/scratch files → `/disk/` (local disk);
    large datasets (>10GB) → `/share/datasets/` ("baunas24 has space; baushare is full"); shared project files →
    `/share/projects/projectname/`. Network-Architecture.md: "Each GPU machine also has local disk at `/disk/`" (not
    true for hokkaido today). Accessing-Clusters.md lists hokkaido as "Shared".
  - *Decision rule.* (a) The wiki's scratch convention is `/disk/u/<user>`, which cannot be followed (no `/disk`).
    (b) `/disk/u` does not exist. → (c): stop Step 2, no directory created, no D-027.
  - *Candidates (not chosen).* (1) Ask the lab admin to provision `/disk` on hokkaido (the wiki convention).
    (2) `/share/datasets/robocurate` on baunas24 (group-writable, the wiki's place for >10 GB data; NFS, so slower
    I/O, and it is the "datasets" share, not scratch). (3) A directory on the local system disk, e.g.
    `/tmp/vivek-robocurate` (fast, but 52 G free on a 98%-full system disk; the wiki says not to write outside
    `/share` and `/disk`). (4) `$HOME` is excluded by CLAUDE.md.

- **B-005 (2026-11-19T15:30Z; CLOSED 2026-11-19 by D-028, option (2)) — the D-025 placebo cannot be trained with LeRobot 0.6.1 as is.**
  - *Closure.* D-028 permutes whole progress curves (normalized-time resampling, no rescale); `prepare` writes
    `perm{i}_progress.parquet` in the entrant's RA-BC schema (index, episode_index, frame_index, progress_sparse;
    rabc.py:101-126) and the local backend passes permutation `subset_index` as `--sample_weighting.progress_path`.
    Tests: golden (rabc vs placebo train cmds differ only in the progress path), LeRobot's own
    `make_sample_weighter` + `RABCWeights` accept every placebo file and give finite, batch-normalized weights.
  - *Fact.* lerobot-train applies per-sample weights only via `--sample_weighting.type` in {rabc, uniform}
    (utils/sample_weighting.py:125-132, no hook for precomputed weights). RA-BC computes weights from a progress
    parquet: delta = progress[t + chunk_size] − progress[t] (last frame near the end, rewards/sarm/rabc.py:237-260),
    then Eq. 8-9 with global stats (rabc.py:262-295). D-025 defines the placebo on WEIGHT curves (permute, resample,
    rescale); an arbitrary resampled weight curve is not in general the RA-BC image of any progress curve (weights
    near episode ends are tied to the last frame's progress), so the placebo cannot be encoded as a progress parquet.
    Batch weights are renormalized to sum to the batch size (rabc.py:232-233) and the loss divides by the weight sum
    (lerobot_train.py:162), so D-025's global rescale is a no-op for LeRobot training (harmless).
  - *Done.* `permute_weight_curves` (D-025) with tests; `prepare` writes the entrant's RA-BC frame weights
    (`weights.parquet`, our mirror equal to LeRobot's RABCWeights before batch normalization) and the 8 D-025
    placebo weight parquets; the local backend refuses `weight_placebo` runs (test). The D-020 progress permutation
    is no longer written. The fake backend is unaffected.
  - *Options (decision needed).* (1) A thin launcher around lerobot-train that registers a precomputed-weights
    weighter (reads our weights parquet, same batch normalization as RABCWeights) for BOTH arms (entrant weights =
    our RA-BC mirror, placebo = D-025), leaving the training loop untouched. (2) Keep LeRobot's RA-BC for the
    entrant and define the placebo on progress curves (D-020 interpretation, rejected by D-025). (3) Contribute a
    precomputed-weights sample-weighting type upstream and pin the release that has it.

- **B-004 (RESOLVED by D-026; 2026-11-19T13:55Z) — the D-019/D-024 test misses the 0.06 bound on result-eligible designs with n_c > n_p.**
  Stop condition (B-003 task, Step 1): "If any allowed cell fails the bound, STOP and report the table. Do not
  adjust the test."
  - *What was run.* `test_primary_test_null_calibration` with the eligible cells = the M0 power-table designs the task
    prompt names (n_c in {3, 5}, K in {8, 12}, n_p = 2), chosen from the prompt, not from results. 200 000 grids per
    eligible cell; the K = 4 cell keeps its calibration-only FWER <= 0.07 check; the n_p = 1 cell is reported as
    not result-eligible with no pass/fail (D-024).
  - *Result.* (8, 2, 3) and (12, 2, 3) pass every sigma cell (max upper 95% 0.0487 and 0.0529). (8, 2, 5) fails 3 of
    8 sigma cells and (12, 2, 5) fails 3 of 8 (max upper 95% 0.0773 at sigma_sub 0.04, sigma_seed 0.03, K 12).
  - *Cause (same mechanism as B-003).* With the pooled s_seed^2 and s_sub^2 = (MSB − s_seed^2)/n_p,
    SE^2 = MSB(1 + 1/K)/n_p + s_seed^2 (1/n_c − 1/n_p) (before truncation). When n_c = n_p the s_seed^2 term
    vanishes and SE^2 is MSB times a constant: exactly chi-square with K − 1 df, so df = K − 1 is exact (up to
    clipping and the binomial). n_c < n_p adds a positive term (conservative). n_c > n_p subtracts one, which makes
    the SE noisier than K − 1 df allows (liberal): weight −1/6 at (n_p 2, n_c 3) passes; −0.3 at (2, 5) fails;
    −2/3 at (1, 3) was B-003.
  - *Diagnostic probe (not part of the test; 200 000 grids per cell).* Every design with n_c <= n_p stays far
    below the bound: max upper 95% 0.0433 (K 12, n_p 2, n_c 2). Table below the main table.
  - *Note.* Today's planner gives every condition the same `train_seeds`, so grids can only produce n_c = n_p
    designs (the exact case). Designs with n_c != n_p (the power table's n_c 5 / n_p 2) need per-condition seed
    counts that grids cannot express yet.
  - *Options (protocol decision, not made here).* (1) Require n_p >= n_c for result runs (amend D-024): valid by
    construction, matches what the planner produces today, every probed cell passes; power table restricted to
    n_c <= n_p. (2) Keep n_p >= 2 and add the parametric bootstrap (the named fallback). (3) Cap n_c at 3 when n_p = 2
    (passes, but no principle for other n_p).
  - *Done regardless.* D-024 appended; `check_result_placebo_seeds` (planner) wired into the result-mode guard; unit
    tests `test_check_result_placebo_seeds_unit`, `test_result_mode_guard_refuses_np_below_2`,
    `test_simulate_eligibility_labels`; the table labels eligibility and reports MC SE for every cell.
  - Reproduce: `pytest -q -s -m slow -k null_calibration` (~30 s).

Main table (`report/simulate.py`, D-024 labels):

| sigma_sub | sigma_seed | K, n_p, n_c | eligibility | grids | FWER | MC SE | upper 95% | criterion | power 15pp (MC) | power 15pp (analytic) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.03 | 4, 2, 3 | calibration only (K < 8) | 40000 | 0.0000 | 0.0000 | 0.0001 | FWER <= 0.07 ok | 0.060 | 0.343 |
| 0.00 | 0.03 | 8, 2, 3 | result-eligible | 200000 | 0.0024 | 0.0001 | 0.0026 | upper95 <= 0.06 ok | 0.767 | 0.860 |
| 0.00 | 0.03 | 8, 2, 5 | result-eligible | 200000 | 0.0016 | 0.0001 | 0.0018 | upper95 <= 0.06 ok | 0.896 | 0.970 |
| 0.00 | 0.03 | 12, 2, 3 | result-eligible | 200000 | 0.0070 | 0.0002 | 0.0074 | upper95 <= 0.06 ok | 0.917 | 0.954 |
| 0.00 | 0.03 | 12, 2, 5 | result-eligible | 200000 | 0.0057 | 0.0002 | 0.0061 | upper95 <= 0.06 ok | 0.976 | 0.997 |
| 0.00 | 0.03 | 8, 1, 3 | not result-eligible (n_p = 1) | 200000 | 0.0027 | 0.0001 | 0.0029 | — (no pass/fail) | 0.591 | 0.797 |
| 0.00 | 0.06 | 4, 2, 3 | calibration only (K < 8) | 40000 | 0.0000 | 0.0000 | 0.0000 | FWER <= 0.07 ok | 0.004 | 0.147 |
| 0.00 | 0.06 | 8, 2, 3 | result-eligible | 200000 | 0.0023 | 0.0001 | 0.0025 | upper95 <= 0.06 ok | 0.269 | 0.451 |
| 0.00 | 0.06 | 8, 2, 5 | result-eligible | 200000 | 0.0016 | 0.0001 | 0.0018 | upper95 <= 0.06 ok | 0.476 | 0.666 |
| 0.00 | 0.06 | 12, 2, 3 | result-eligible | 200000 | 0.0067 | 0.0002 | 0.0070 | upper95 <= 0.06 ok | 0.456 | 0.599 |
| 0.00 | 0.06 | 12, 2, 5 | result-eligible | 200000 | 0.0055 | 0.0002 | 0.0058 | upper95 <= 0.06 ok | 0.690 | 0.830 |
| 0.00 | 0.06 | 8, 1, 3 | not result-eligible (n_p = 1) | 200000 | 0.0026 | 0.0001 | 0.0028 | — (no pass/fail) | 0.183 | 0.383 |
| 0.02 | 0.03 | 4, 2, 3 | calibration only (K < 8) | 40000 | 0.0000 | 0.0000 | 0.0000 | FWER <= 0.07 ok | 0.060 | 0.226 |
| 0.02 | 0.03 | 8, 2, 3 | result-eligible | 200000 | 0.0100 | 0.0002 | 0.0104 | upper95 <= 0.06 ok | 0.610 | 0.654 |
| 0.02 | 0.03 | 8, 2, 5 | result-eligible | 200000 | 0.0157 | 0.0003 | 0.0163 | upper95 <= 0.06 ok | 0.734 | 0.783 |
| 0.02 | 0.03 | 12, 2, 3 | result-eligible | 200000 | 0.0221 | 0.0003 | 0.0228 | upper95 <= 0.06 ok | 0.781 | 0.804 |
| 0.02 | 0.03 | 12, 2, 5 | result-eligible | 200000 | 0.0367 | 0.0004 | 0.0375 | upper95 <= 0.06 ok | 0.866 | 0.910 |
| 0.02 | 0.03 | 8, 1, 3 | not result-eligible (n_p = 1) | 200000 | 0.0105 | 0.0002 | 0.0109 | — (no pass/fail) | 0.492 | 0.601 |
| 0.02 | 0.06 | 4, 2, 3 | calibration only (K < 8) | 40000 | 0.0000 | 0.0000 | 0.0000 | FWER <= 0.07 ok | 0.005 | 0.118 |
| 0.02 | 0.06 | 8, 2, 3 | result-eligible | 200000 | 0.0052 | 0.0002 | 0.0055 | upper95 <= 0.06 ok | 0.240 | 0.356 |
| 0.02 | 0.06 | 8, 2, 5 | result-eligible | 200000 | 0.0061 | 0.0002 | 0.0065 | upper95 <= 0.06 ok | 0.413 | 0.505 |
| 0.02 | 0.06 | 12, 2, 3 | result-eligible | 200000 | 0.0133 | 0.0003 | 0.0138 | upper95 <= 0.06 ok | 0.399 | 0.483 |
| 0.02 | 0.06 | 12, 2, 5 | result-eligible | 200000 | 0.0180 | 0.0003 | 0.0186 | upper95 <= 0.06 ok | 0.590 | 0.668 |
| 0.02 | 0.06 | 8, 1, 3 | not result-eligible (n_p = 1) | 200000 | 0.0054 | 0.0002 | 0.0058 | — (no pass/fail) | 0.174 | 0.308 |
| 0.04 | 0.03 | 4, 2, 3 | calibration only (K < 8) | 40000 | 0.0014 | 0.0002 | 0.0018 | FWER <= 0.07 ok | 0.053 | 0.107 |
| 0.04 | 0.03 | 8, 2, 3 | result-eligible | 200000 | 0.0362 | 0.0004 | 0.0370 | upper95 <= 0.06 ok | 0.324 | 0.311 |
| 0.04 | 0.03 | 8, 2, 5 | result-eligible | 200000 | 0.0612 | 0.0005 | 0.0622 | upper95 <= 0.06 **FAIL** | 0.390 | 0.357 |
| 0.04 | 0.03 | 12, 2, 3 | result-eligible | 200000 | 0.0507 | 0.0005 | 0.0517 | upper95 <= 0.06 ok | 0.436 | 0.423 |
| 0.04 | 0.03 | 12, 2, 5 | result-eligible | 200000 | 0.0761 | 0.0006 | 0.0773 | upper95 <= 0.06 **FAIL** | 0.502 | 0.483 |
| 0.04 | 0.03 | 8, 1, 3 | not result-eligible (n_p = 1) | 200000 | 0.0433 | 0.0005 | 0.0442 | — (no pass/fail) | 0.312 | 0.293 |
| 0.04 | 0.06 | 4, 2, 3 | calibration only (K < 8) | 40000 | 0.0002 | 0.0001 | 0.0003 | FWER <= 0.07 ok | 0.008 | 0.074 |
| 0.04 | 0.06 | 8, 2, 3 | result-eligible | 200000 | 0.0174 | 0.0003 | 0.0179 | upper95 <= 0.06 ok | 0.182 | 0.205 |
| 0.04 | 0.06 | 8, 2, 5 | result-eligible | 200000 | 0.0326 | 0.0004 | 0.0334 | upper95 <= 0.06 ok | 0.278 | 0.260 |
| 0.04 | 0.06 | 12, 2, 3 | result-eligible | 200000 | 0.0341 | 0.0004 | 0.0349 | upper95 <= 0.06 ok | 0.276 | 0.283 |
| 0.04 | 0.06 | 12, 2, 5 | result-eligible | 200000 | 0.0586 | 0.0005 | 0.0597 | upper95 <= 0.06 ok | 0.383 | 0.362 |
| 0.04 | 0.06 | 8, 1, 3 | not result-eligible (n_p = 1) | 200000 | 0.0178 | 0.0003 | 0.0184 | — (no pass/fail) | 0.159 | 0.186 |
| 0.06 | 0.03 | 4, 2, 3 | calibration only (K < 8) | 40000 | 0.0083 | 0.0005 | 0.0092 | FWER <= 0.07 ok | 0.038 | 0.057 |
| 0.06 | 0.03 | 8, 2, 3 | result-eligible | 200000 | 0.0478 | 0.0005 | 0.0487 | upper95 <= 0.06 ok | 0.157 | 0.146 |
| 0.06 | 0.03 | 8, 2, 5 | result-eligible | 200000 | 0.0646 | 0.0005 | 0.0657 | upper95 <= 0.06 **FAIL** | 0.182 | 0.157 |
| 0.06 | 0.03 | 12, 2, 3 | result-eligible | 200000 | 0.0520 | 0.0005 | 0.0529 | upper95 <= 0.06 ok | 0.209 | 0.198 |
| 0.06 | 0.03 | 12, 2, 5 | result-eligible | 200000 | 0.0643 | 0.0005 | 0.0654 | upper95 <= 0.06 **FAIL** | 0.235 | 0.215 |
| 0.06 | 0.03 | 8, 1, 3 | not result-eligible (n_p = 1) | 200000 | 0.0699 | 0.0006 | 0.0711 | — (no pass/fail) | 0.179 | 0.141 |
| 0.06 | 0.06 | 4, 2, 3 | calibration only (K < 8) | 40000 | 0.0010 | 0.0002 | 0.0013 | FWER <= 0.07 ok | 0.012 | 0.047 |
| 0.06 | 0.06 | 8, 2, 3 | result-eligible | 200000 | 0.0365 | 0.0004 | 0.0373 | upper95 <= 0.06 ok | 0.122 | 0.114 |
| 0.06 | 0.06 | 8, 2, 5 | result-eligible | 200000 | 0.0618 | 0.0005 | 0.0628 | upper95 <= 0.06 **FAIL** | 0.169 | 0.131 |
| 0.06 | 0.06 | 12, 2, 3 | result-eligible | 200000 | 0.0496 | 0.0005 | 0.0506 | upper95 <= 0.06 ok | 0.167 | 0.155 |
| 0.06 | 0.06 | 12, 2, 5 | result-eligible | 200000 | 0.0745 | 0.0006 | 0.0756 | upper95 <= 0.06 **FAIL** | 0.215 | 0.180 |
| 0.06 | 0.06 | 8, 1, 3 | not result-eligible (n_p = 1) | 200000 | 0.0428 | 0.0005 | 0.0437 | — (no pass/fail) | 0.126 | 0.107 |

Diagnostic: FWER vs n_c/n_p (weight on s_seed^2 = 1/n_c − 1/n_p):

| sigma_sub | sigma_seed | K, n_p, n_c | 1/n_c - 1/n_p | FWER | MC SE | upper95 |
|---|---|---|---|---|---|---|
| 0.00 | 0.03 | 8, 2, 2 | +0.000 | 0.0034 | 0.0001 | 0.0036 |
| 0.00 | 0.03 | 12, 2, 2 | +0.000 | 0.0085 | 0.0002 | 0.0089 |
| 0.00 | 0.03 | 8, 3, 3 | +0.000 | 0.0020 | 0.0001 | 0.0022 |
| 0.00 | 0.03 | 8, 5, 5 | +0.000 | 0.0014 | 0.0001 | 0.0016 |
| 0.00 | 0.03 | 8, 3, 2 | +0.167 | 0.0027 | 0.0001 | 0.0029 |
| 0.00 | 0.03 | 8, 5, 3 | +0.133 | 0.0021 | 0.0001 | 0.0023 |
| 0.00 | 0.03 | 12, 5, 3 | +0.133 | 0.0067 | 0.0002 | 0.0070 |
| 0.00 | 0.06 | 8, 2, 2 | +0.000 | 0.0033 | 0.0001 | 0.0035 |
| 0.00 | 0.06 | 12, 2, 2 | +0.000 | 0.0086 | 0.0002 | 0.0090 |
| 0.00 | 0.06 | 8, 3, 3 | +0.000 | 0.0022 | 0.0001 | 0.0024 |
| 0.00 | 0.06 | 8, 5, 5 | +0.000 | 0.0014 | 0.0001 | 0.0016 |
| 0.00 | 0.06 | 8, 3, 2 | +0.167 | 0.0025 | 0.0001 | 0.0027 |
| 0.00 | 0.06 | 8, 5, 3 | +0.133 | 0.0015 | 0.0001 | 0.0017 |
| 0.00 | 0.06 | 12, 5, 3 | +0.133 | 0.0067 | 0.0002 | 0.0071 |
| 0.02 | 0.03 | 8, 2, 2 | +0.000 | 0.0085 | 0.0002 | 0.0089 |
| 0.02 | 0.03 | 12, 2, 2 | +0.000 | 0.0176 | 0.0003 | 0.0182 |
| 0.02 | 0.03 | 8, 3, 3 | +0.000 | 0.0088 | 0.0002 | 0.0092 |
| 0.02 | 0.03 | 8, 5, 5 | +0.000 | 0.0141 | 0.0003 | 0.0146 |
| 0.02 | 0.03 | 8, 3, 2 | +0.167 | 0.0064 | 0.0002 | 0.0067 |
| 0.02 | 0.03 | 8, 5, 3 | +0.133 | 0.0080 | 0.0002 | 0.0084 |
| 0.02 | 0.03 | 12, 5, 3 | +0.133 | 0.0180 | 0.0003 | 0.0186 |
| 0.02 | 0.06 | 8, 2, 2 | +0.000 | 0.0052 | 0.0002 | 0.0056 |
| 0.02 | 0.06 | 12, 2, 2 | +0.000 | 0.0123 | 0.0002 | 0.0127 |
| 0.02 | 0.06 | 8, 3, 3 | +0.000 | 0.0047 | 0.0002 | 0.0050 |
| 0.02 | 0.06 | 8, 5, 5 | +0.000 | 0.0060 | 0.0002 | 0.0063 |
| 0.02 | 0.06 | 8, 3, 2 | +0.167 | 0.0042 | 0.0001 | 0.0044 |
| 0.02 | 0.06 | 8, 5, 3 | +0.133 | 0.0041 | 0.0001 | 0.0044 |
| 0.02 | 0.06 | 12, 5, 3 | +0.133 | 0.0118 | 0.0002 | 0.0123 |
| 0.04 | 0.03 | 8, 2, 2 | +0.000 | 0.0231 | 0.0003 | 0.0237 |
| 0.04 | 0.03 | 12, 2, 2 | +0.000 | 0.0352 | 0.0004 | 0.0360 |
| 0.04 | 0.03 | 8, 3, 3 | +0.000 | 0.0300 | 0.0004 | 0.0308 |
| 0.04 | 0.03 | 8, 5, 5 | +0.000 | 0.0362 | 0.0004 | 0.0370 |
| 0.04 | 0.03 | 8, 3, 2 | +0.167 | 0.0182 | 0.0003 | 0.0188 |
| 0.04 | 0.03 | 8, 5, 3 | +0.133 | 0.0219 | 0.0003 | 0.0226 |
| 0.04 | 0.03 | 12, 5, 3 | +0.133 | 0.0312 | 0.0004 | 0.0319 |
| 0.04 | 0.06 | 8, 2, 2 | +0.000 | 0.0124 | 0.0002 | 0.0129 |
| 0.04 | 0.06 | 12, 2, 2 | +0.000 | 0.0233 | 0.0003 | 0.0239 |
| 0.04 | 0.06 | 8, 3, 3 | +0.000 | 0.0155 | 0.0003 | 0.0161 |
| 0.04 | 0.06 | 8, 5, 5 | +0.000 | 0.0242 | 0.0003 | 0.0248 |
| 0.04 | 0.06 | 8, 3, 2 | +0.167 | 0.0096 | 0.0002 | 0.0100 |
| 0.04 | 0.06 | 8, 5, 3 | +0.133 | 0.0123 | 0.0002 | 0.0128 |
| 0.04 | 0.06 | 12, 5, 3 | +0.133 | 0.0243 | 0.0003 | 0.0250 |
| 0.06 | 0.03 | 8, 2, 2 | +0.000 | 0.0341 | 0.0004 | 0.0349 |
| 0.06 | 0.03 | 12, 2, 2 | +0.000 | 0.0424 | 0.0005 | 0.0433 |
| 0.06 | 0.03 | 8, 3, 3 | +0.000 | 0.0384 | 0.0004 | 0.0392 |
| 0.06 | 0.03 | 8, 5, 5 | +0.000 | 0.0386 | 0.0004 | 0.0395 |
| 0.06 | 0.03 | 8, 3, 2 | +0.167 | 0.0276 | 0.0004 | 0.0284 |
| 0.06 | 0.03 | 8, 5, 3 | +0.133 | 0.0302 | 0.0004 | 0.0309 |
| 0.06 | 0.03 | 12, 5, 3 | +0.133 | 0.0366 | 0.0004 | 0.0375 |
| 0.06 | 0.06 | 8, 2, 2 | +0.000 | 0.0222 | 0.0003 | 0.0229 |
| 0.06 | 0.06 | 12, 2, 2 | +0.000 | 0.0353 | 0.0004 | 0.0362 |
| 0.06 | 0.06 | 8, 3, 3 | +0.000 | 0.0296 | 0.0004 | 0.0303 |
| 0.06 | 0.06 | 8, 5, 5 | +0.000 | 0.0363 | 0.0004 | 0.0371 |
| 0.06 | 0.06 | 8, 3, 2 | +0.167 | 0.0181 | 0.0003 | 0.0187 |
| 0.06 | 0.06 | 8, 5, 3 | +0.133 | 0.0218 | 0.0003 | 0.0225 |
| 0.06 | 0.06 | 12, 5, 3 | +0.133 | 0.0314 | 0.0004 | 0.0322 |

- **B-003 (RESOLVED by D-024, which excludes n_p = 1 from result runs; superseded by B-004; 2026-11-19T06:18Z) — the D-019 test (df = K−1) misses the K >= 8 tier at n_p = 1.**
  Stop condition from the B-002 decision: "If any K >= 8 cell misses the 0.06 bound, STOP and report the table.
  Do not weaken the test. Next step in that case would be the parametric bootstrap."
  - *Symptom.* `test_primary_test_null_calibration` (slow, hard, two tiers) fails one cell:
    (sigma_sub 0.06, sigma_seed 0.03, K 8, n_p 1, n_c 3): FWER 0.0699, MC SE 0.0006, upper 95% 0.0711 > 0.06.
    Tier 1 (FWER <= 0.07 everywhere) passes; every other K >= 8 cell has upper 95% <= 0.049. Table below.
  - *Cause.* With n_p = 1 the placebo has no within-subset replicates. SE^2 = MSB(1 + 1/K) + s_seed^2(1/n_c − 1/n_p),
    so the s_seed^2 term enters with weight −2/3 (−1/6 at n_p = 2). The negative weight makes the SE estimate
    noisier than a chi-square with K−1 df: the Satterthwaite-equivalent df is ~4.1, not the assumed 7.
  - *Outside-grid probe (200k grids, sigma_seed 0.03).* n_p = 1: K=8 0.070/0.069/0.061 and K=12
    0.080/0.069/0.060 at sigma_sub 0.06/0.08/0.10, so raising K does not fix it. n_p = 2: K=8 0.048/0.047/0.045 and
    K=12 0.052/0.049/0.046, max upper 95% 0.053.
  - *Options (protocol decision, not made here).* (1) Parametric bootstrap reference distribution (the
    fallback D-019 names). (2) Require n_p >= 2 for result runs: passes every simulated cell; it costs K extra
    runs per budget vs n_p = 1. (3) df = min(K−1, Satterthwaite). Not simulated.
  - *Done regardless of the choice.* D-019 recorded; df = K−1 in the test and in power/MDE; result mode refuses
    K < 8 (`check_result_subsets`, tests `test_result_mode_refuses_k_below_8`, `test_check_result_subsets_unit`).
  - *Not done (waits on this decision).* Step 4 (report switch + K=8 fake e2e), Parts 2-4 (hokkaido, GPU).
    `robocurate report` still uses Welch as primary.
  - Reproduce: `python scripts/d016_null_grid.py` (~12 s) or `pytest -q -s -m slow -k null_calibration`.

| sigma_sub | sigma_seed | K, n_p, n_c | grids | FWER | MC SE | upper 95% | bound | power 15pp (MC) | power 15pp (analytic) |
|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.03 | 4, 2, 3 | 40000 | 0.0000 | 0.0000 | 0.0001 | FWER <= 0.07 ok | 0.060 | 0.343 |
| 0.00 | 0.03 | 8, 2, 3 | 200000 | 0.0024 | 0.0001 | 0.0026 | upper95 <= 0.06 ok | 0.767 | 0.860 |
| 0.00 | 0.03 | 8, 1, 3 | 200000 | 0.0027 | 0.0001 | 0.0029 | upper95 <= 0.06 ok | 0.591 | 0.797 |
| 0.00 | 0.06 | 4, 2, 3 | 40000 | 0.0000 | 0.0000 | 0.0000 | FWER <= 0.07 ok | 0.004 | 0.147 |
| 0.00 | 0.06 | 8, 2, 3 | 200000 | 0.0023 | 0.0001 | 0.0025 | upper95 <= 0.06 ok | 0.269 | 0.451 |
| 0.00 | 0.06 | 8, 1, 3 | 200000 | 0.0026 | 0.0001 | 0.0028 | upper95 <= 0.06 ok | 0.183 | 0.383 |
| 0.02 | 0.03 | 4, 2, 3 | 40000 | 0.0000 | 0.0000 | 0.0000 | FWER <= 0.07 ok | 0.060 | 0.226 |
| 0.02 | 0.03 | 8, 2, 3 | 200000 | 0.0100 | 0.0002 | 0.0104 | upper95 <= 0.06 ok | 0.610 | 0.654 |
| 0.02 | 0.03 | 8, 1, 3 | 200000 | 0.0105 | 0.0002 | 0.0109 | upper95 <= 0.06 ok | 0.492 | 0.601 |
| 0.02 | 0.06 | 4, 2, 3 | 40000 | 0.0000 | 0.0000 | 0.0000 | FWER <= 0.07 ok | 0.005 | 0.118 |
| 0.02 | 0.06 | 8, 2, 3 | 200000 | 0.0052 | 0.0002 | 0.0055 | upper95 <= 0.06 ok | 0.240 | 0.356 |
| 0.02 | 0.06 | 8, 1, 3 | 200000 | 0.0054 | 0.0002 | 0.0058 | upper95 <= 0.06 ok | 0.174 | 0.308 |
| 0.04 | 0.03 | 4, 2, 3 | 40000 | 0.0014 | 0.0002 | 0.0018 | FWER <= 0.07 ok | 0.053 | 0.107 |
| 0.04 | 0.03 | 8, 2, 3 | 200000 | 0.0362 | 0.0004 | 0.0370 | upper95 <= 0.06 ok | 0.324 | 0.311 |
| 0.04 | 0.03 | 8, 1, 3 | 200000 | 0.0433 | 0.0005 | 0.0442 | upper95 <= 0.06 ok | 0.312 | 0.293 |
| 0.04 | 0.06 | 4, 2, 3 | 40000 | 0.0002 | 0.0001 | 0.0003 | FWER <= 0.07 ok | 0.008 | 0.074 |
| 0.04 | 0.06 | 8, 2, 3 | 200000 | 0.0174 | 0.0003 | 0.0179 | upper95 <= 0.06 ok | 0.182 | 0.205 |
| 0.04 | 0.06 | 8, 1, 3 | 200000 | 0.0178 | 0.0003 | 0.0184 | upper95 <= 0.06 ok | 0.159 | 0.186 |
| 0.06 | 0.03 | 4, 2, 3 | 40000 | 0.0083 | 0.0005 | 0.0092 | FWER <= 0.07 ok | 0.038 | 0.057 |
| 0.06 | 0.03 | 8, 2, 3 | 200000 | 0.0478 | 0.0005 | 0.0487 | upper95 <= 0.06 ok | 0.157 | 0.146 |
| 0.06 | 0.03 | 8, 1, 3 | 200000 | 0.0699 | 0.0006 | 0.0711 | upper95 <= 0.06 **FAIL** | 0.179 | 0.141 |
| 0.06 | 0.06 | 4, 2, 3 | 40000 | 0.0010 | 0.0002 | 0.0013 | FWER <= 0.07 ok | 0.012 | 0.047 |
| 0.06 | 0.06 | 8, 2, 3 | 200000 | 0.0365 | 0.0004 | 0.0373 | upper95 <= 0.06 ok | 0.122 | 0.114 |
| 0.06 | 0.06 | 8, 1, 3 | 200000 | 0.0428 | 0.0005 | 0.0437 | upper95 <= 0.06 ok | 0.126 | 0.107 |

- **B-002 (RESOLVED by D-019; 2026-11-19T05:55Z, OPEN) — the D-016 test as specified fails the Part 1 null calibration at K=4.**
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

## Overnight m1-core log (branch m1-core, CPU only, GPU-h 0)

Started 2026-11-19T06:19:12Z. Hokkaido read-only check at start: no tmux session for this user (`error connecting to /tmp//tmux-1176/default`); M0 was never launched (B-003), so there is no m0 log path; GPU in use by another user (66% util, 27584 MiB).

| UTC | Item | Commit | Tests | Outcome |
|---|---|---|---|---|
| 2026-11-19T06:21Z | 1 decisions + protocol | (next) | default 152 passed; ruff clean | D-020..D-023 verbatim; protocol: tie_break_salt (secrets.token_hex(16)), weight_permutation_seeds 3000-3007, max_curation_rollouts 100 (validated); weighting condition kind + weight_placebo spec/planner; task_a_v0 weighting example (rabc, 8 permutations) |
| 2026-11-19T06:23Z | 2 ordering + ties | (next) | default 155 passed; ruff clean | Mixture builder already shuffles by recipe seed (mixture.py:117 rng.permutation) -> test only. ScoreCleaner tie-break now sha256(salt||content hash) (D-021); old index tie test replaced. Fake e2e rerun (task_a_v0, 90 runs): length_heuristic +0.014 p_holm 0.643 not sig; oracle +0.256 p 4.7e-06 sig; MDE 0.089 (numbers differ from Phase 8 because the protocol hash, hence run_id/fake rng, changed). Weighting group shown but not yet analysed (item 8). |
| 2026-11-19T06:36Z | 3 curation infra | (next) | default 172 passed; integration 3 passed; ruff clean | Cleaner resource declarations + validate_resources (D-022); WeightingCleaner (D-020); curation/{context,rollouts,progress,weights,prepare,policy}.py; `robocurate prepare` (fake backend; local refuses: needs mixture, Robometer weights, M0 checkpoint); runs consume selections only from the cache (fake + local); rollout service rejects eval/calibration/out-of-range/repeated seeds and enforces the cap; reference policy chmod read-only + tree-hash tamper check; LeRobot RA-BC weighting wired for weighting-track conditions only (golden test: non-weighting train cmds byte-identical); our RA-BC weight mirror equals lerobot RABCWeights on the same parquet. Bug found+fixed with regression test: cache key used the protocol salt instead of the view's. Import-graph fix: prepare no longer imports runs/backends/local (reached runs.oracle). |
| 2026-11-19T06:37Z | 4 toybed | (next) | default 175 passed; ruff clean | tests/toybed: PointReachEnv (2D reach, obs [x,y,gx,gy]), expert controller, executed defects clean/pause/jitter/wrong (wrong = drives to the goal mirrored through the start), shuffled dataset with hidden labels, goal-distance progress; MLPPolicy with loss(batch) in curation/policy.py. Sanity: BC on clean 1.00 vs all 0.53 success (100 toy seeds), < 1 s. |
| 2026-11-19T06:57Z | 5 entrants | (next) | default 195 passed + 1 xfailed (cupid sanity); entrant non-slow 19 passed; ruff clean | 10 entrants registered (lazy). Wrapped: lerobot_doctor, kinema, roboaudit, rda (adapters; tools NOT installed: install into shared env was denied by the permission system), robometer_filter, rabc (LeRobot reward/RA-BC). Reimplemented: demo_score, deminf, cupid (official code not pip-installable / no license). Original: rcb_score v1 (frozen, digest test). Toybed sanity with published defaults: robometer_filter/rcb_score/rabc (mock progress) pass; demo_score passes after replacing an unsourced epoch count (200) with the reference code's default 3000; deminf passes (slow: 222 s at the paper's 50k VAE steps) after fixing a KSG zero-radius NaN bug (regression test); cupid FAILS with defaults (xfail strict, reported). trajlens (pyarrow<25 vs 26.0.1) and retake (core dep upgrades + external LLM APIs) not adapted. |
| 2026-11-19T07:00Z | 6 policy-rollout source | (next) | default 197 passed + 1 xfailed; integration 3 passed; ruff clean | data/mixture.py: PolicySource / collect_policy_episodes / write_policy_dataset: rolls out policies (seeds from the TOP of curation_seed_range downward, disjoint from cleaner rollouts which take seeds bottom-up; eval seeds refused) with replay-mode Perturbations executed in the env; LeRobot dataset (no video) + sidecar labels (source policy, checkpoint_step, success, perturbations; order shuffled per D-021). EpisodeLabel gained optional checkpoint_step/success/perturbations (schema-compatible). Tested end to end on the toybed with fake policies; not run on real checkpoints. |
| 2026-11-19T07:03Z | 7 validate-defects | (next) | default 198 passed + 1 xfailed; ruff clean | robocurate validate-defects (data/validate.py; v2.1 + v3.0 parquet; actions/state/timestamps only; meta/** + data/** download, 0.5-45.6 MB each). Ran on 5 pinned SO-100/101 datasets (331 episodes): interior-pause episode rate 0.06-0.90 (median segment 0.6-1.2 s) vs our 11% at 0.5-1.5 s; 0 exact/near duplicates vs our 7%; jitter>0.10 in 0% of episodes except kmg0620 (15%); timestamp gaps in 22/150 episodes of one dataset; all single-task. docs/defect_validation.md + stats JSON. M0 mixture output not available (never built). Interior/boundary pause split added after the first run (both reported). |
| 2026-11-19T07:09Z | 8 reproduce, site, CI | (next) | default (CI selection) 197 passed + 1 xfailed; ruff clean; CI YAML valid, its commands run locally | report switched to the D-019 primary test (Welch = labeled secondary), weighting-track analysis (Holm across weighting entrants, full data as reference), variance components + MDE, curation cost, cell ids. robocurate report --site (single self-contained HTML + leaderboard.json, FAKE watermark). robocurate reproduce <cell> (resolves git hash, configs+sha256, selection keys; prints commands; fake re-execution MATCH 6/6). .github/workflows/ci.yml (ubuntu-24.04, Python 3.12.16, exact lock with CPU torch wheels, ruff + default suite minus integration). Bugs fixed with regression tests: second report on the same dir loaded leaderboard.json as a run; in-process CLI calls recorded the caller's sys.argv as the command line. Outputs pasted above (m1-core end-to-end section). |
| 2026-11-19T07:10Z | 9 submission path | (next) | template install test (slow) passed; ruff clean | examples/cleaner_template (installable package, robocurate.cleaners entry point, README); docs/submit.md (tracks, access rules, resource declarations, submit checks, how results are produced); tests/test_submission_template.py installs the template into a throwaway venv (--system-site-packages, --no-deps --no-index) and passes robocurate submit. |
| 2026-11-19T07:11Z | 10 README + Task B readiness | (next) | ruff clean; README quickstart submit command run | README draft (pitch, protocol D-002/D-016/D-019/D-020 in plain language, architecture, entrants, fake quickstart, Results: pending, limitations, credits with TODO / VERIFY CITATION markers: CuraEval not found, QoQ code not found). docs/task_b_readiness.md: pusht env registered (configs.py:191, extra gym-pusht not installed), metaworld registered (configs.py:454, metaworld==3.0.0 not installed; scripted experts via metaworld.policies, envs/metaworld.py:24,67-73,107), no robomimic env in lerobot; Hub datasets + formats + revisions; robomimic mh only as official hdf5. |
| 2026-11-19T07:24Z | 11 morning summary | (this commit) | default 197+1 xfailed; integration 3; slow 10 passed, 1 failed (B-003, inherited); ruff clean | Morning summary at the top of PROGRESS.md; LeRobot touchpoints + method sources sections; parking lot (QoQ, DataMIL, CUPID ACT adapter, trajlens). Hokkaido end check: no rcb-m0 session (never launched), GPU 100% / 27.6 GB used by another user. |
<!-- m1-log-end -->

## m1-core end-to-end (fake backend): prepare + run + report --site + reproduce, task_a_v0 incl. weighting track

Commands (fake params = configs/fake/e2e_test.yaml with sleep_s 0):
```
robocurate prepare configs/grids/task_a_v0.yaml --backend fake --selections-dir $ROBOCURATE_SCRATCH/selections/m1-e2e-fake
  rabc v1 [weighting] budget 1.0: progress + 8 permutation parquets; computed in 0.24s, GPU-h 0.000, rollouts 0, key a54930fab8b0
  length_heuristic v1 [selection] budget 0.5: 25 episodes; computed in 0.00s, GPU-h 0.000, rollouts 0, key ad56eba12505
robocurate run configs/grids/task_a_v0.yaml --backend fake --fake-params <fake> --results-dir $ROBOCURATE_SCRATCH/results/m1-e2e-fake --selections-dir ...
  done=90 skipped_existing=0 skipped_locked=0 reclaimed_stale_locks=0
robocurate report $ROBOCURATE_SCRATCH/results/m1-e2e-fake --n-boot 2000 --site $ROBOCURATE_SCRATCH/site-fake
site: /Users/vivek/robocurate-scratch/site-fake/index.html
*** FAKE RESULTS (synthetic backend) ***
task aloha_transfer_cube budget 0.5 (selection):
  full                         runs=6   mean=0.562  cell 5c055b8f910e
  random                       runs=18  mean=0.489  cell e5267b2f86c9
  oracle                       runs=6   mean=0.745  cell b8be368d553c
  cleaner:length_heuristic     runs=6   mean=0.503  cell 186cc0bd36cb
  cleaner:length_heuristic vs placebo [D-019]: diff=+0.014 SE=0.028 df=2 p_holm=0.662 not significant   (Welch secondary p_holm=0.643)
  [ref] oracle vs placebo: diff=+0.256 p=0.0117 SIGNIFICANT
  [ref] full vs placebo: diff=+0.073 p=0.12 not significant
  s_sub=0.000 (truncated) s_seed=0.059 K=3 n_p=6 n_c=6  MDE (80% power): 0.158
task aloha_transfer_cube full data (weighting):
  full                         runs=6   mean=0.562  cell 5c055b8f910e
  weighting:rabc               runs=6   mean=0.514  cell a03b0f077b04
  weight_placebo:rabc          runs=48  mean=0.512  cell 167857559e44
  weighting:rabc vs placebo [D-019]: diff=+0.002 SE=0.034 df=7 p_holm=0.958 not significant   (Welch secondary p_holm=0.938)
  [ref] full vs placebo: diff=+0.050 p=0.143 not significant
  s_sub=0.016 s_seed=0.069 K=8 n_p=6 n_c=6  MDE (80% power): 0.113
wrote /Users/vivek/robocurate-scratch/results/m1-e2e-fake/report.md, /Users/vivek/robocurate-scratch/results/m1-e2e-fake/leaderboard.json
robocurate reproduce a03b0f077b04 --results-dir ... --backend fake --selections-dir ...
cell a03b0f077b04: task aloha_transfer_cube budget 1.0 condition {'kind': 'weighting', 'name': 'rabc', 'version': '1'} (6 runs, kind fake)
git: 90fec7f70f (dirty)
config /Users/vivek/Developer/CODES/RoboCurate-Bench/configs/protocol.yaml: sha256 412a70e4c962 ok
config /Users/vivek/Developer/CODES/RoboCurate-Bench/configs/tasks/aloha_transfer_cube.yaml: sha256 37eef6888ecb ok
config configs/grids/task_a_v0.yaml: sha256 40d7e2343354 ok
config <scratchpad>/fake_nosleep.yaml: sha256 d620cc130ba5 ok
selection cache keys: a54930fab8b0
commands:
  robocurate prepare configs/grids/task_a_v0.yaml --backend fake --selections-dir <SELECTIONS>
  robocurate run configs/grids/task_a_v0.yaml --backend fake --fake-params <scratchpad>/fake_nosleep.yaml --results-dir /Users/vivek/robocurate-scratch/results/m1-e2e-fake --selections-dir /Users/vivek/robocurate-scratch/selections/m1-e2e-fake
re-executed: MATCH 6/6
```
Known injected effects: oracle +0.25 (recovered: +0.256, p = 0.012, reference), length_heuristic 0 (not significant),
rabc 0 (not significant vs its 8 permutation placebos). With D-019 the null full-vs-placebo difference (+0.073) is no
longer significant (Welch gave p = 0.02 on the same data). The site is one self-contained HTML file (no scripts, no
external requests) + leaderboard.json, FAKE-watermarked.

## M0 log (branch m0-calibration)

| UTC | Step | Commit | Outcome | GPU-h |
|---|---|---|---|---|
| 2026-11-19T05:55Z | Part 0: D-016/017/018, protocol `calibration_seed_start: 300000`, CLAUDE.md rule 10 | `63a9a5a` | committed | 0 |
| 2026-11-19T05:55Z | Part 1: D-016 variance components, Satterthwaite test, Design/power/MDE, grid script | `ba55b62` | blocked, B-002 | 0 |
| 2026-11-19T06:18Z | D-019: df = K−1, result-mode K >= 8 guard, hard two-tier calibration test, `report/simulate.py` | (this commit) | **blocked, B-003** | 0 |

D-019 step: default 145 passed; integration 3 passed; slow 7 passed + 1 FAILED (`test_primary_test_null_calibration`, B-003, intended); ruff clean.
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

Added in the D-026 task: `utils/sample_weighting.py:125-132` (types rabc/uniform only), `:154-155` (RA-BC
chunk_size from policy.config); `rewards/sarm/rabc.py:232-233` (batch weights normalized to sum to batch size),
`:237-260` (delta = progress[t + chunk_size] − progress[t], last frame near the end); `scripts/lerobot_train.py:162`
(loss = sum(w l) / sum(w)); `policies/act/configuration_act.py:85` (ACT chunk_size 100); `policies/factory.py:102`
(`make_policy_config`); `utils/constants.py:69` (HF_LEROBOT_HOME read at import). Other sources: lab wiki
github.com/thebaulab/onramp.wiki @ ec6b6665 (Filesystems-on-bau-clusters, NFS-Storage-Administration "Where Should
I Put My Files?", Network-Architecture "Local Storage"); CUPID github.com/agiachris/cupid@2941eba9
(`scripts/train/train_trak.sh:309-316`, `diffusion_policy/data_attribution/modelout_functions.py:111-136`); PyPI
JSON API for checker project metadata.
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

### New touchpoints (m1-core)
- **Sample weighting (weighting track):**
  - `TrainPipelineConfig.sample_weighting: SampleWeightingConfig | None` (`configs/train.py:139`; legacy
    `use_rabc` migration `:61-73`).
  - `SampleWeightingConfig` fields: type, progress_path, head_mode, kappa 0.01, epsilon 1e-6
    (`utils/sample_weighting.py:76-100`).
  - `make_sample_weighter` supports only `rabc` and `uniform` (`:103-132`; no plugin hook). Its RABC builder reads
    `policy.config.chunk_size` (`:155`).
  - Created in training at `scripts/lerobot_train.py:406-418`. Weighted loss is `sum(w*l)/(sum(w)+eps)` (`:162`).
- **RABCWeights** (`rewards/sarm/rabc.py:59`):
  - Parquet columns: index, episode_index, `progress_<head_mode>` (`:101-126`).
  - Global delta stats `:143-173`; delta over chunk_size, clamped at the episode end `:237-260`.
  - SARM Eq. 8-9 `:262-295`; per-batch normalization to sum = batch size `:230-233`.
- **Robometer:**
  - `compute_robometer_progress(dataset_repo_id, reward_model_path, output_path, device, ...)`
    (`rewards/robometer/compute_rabc_weights.py:103`).
  - It loads the dataset by repo_id only (`:133`) and writes the RA-BC schema with `progress_sparse` (`:193-201`).
  - `compute_reward` returns clamped progress, or a 0.5-thresholded success (`modeling_robometer.py:249-283`).
  - Config: `reward_output` "progress"/"success", `success_threshold` 0.5, default `pretrained_path`
    `lerobot/Robometer-4B` (`configuration_robometer.py:53,59-60`).
- **Env rollouts (curation/rollouts.py, GPU path only):**
  - `make_env_config` (`envs/factory.py:26`), `make_env_pre_post_processors` (`:37`), `make_env` (`:58`).
  - `lerobot_eval.rollout(..., seeds, return_observations)` (`scripts/lerobot_eval.py:167`; observations
    `(b, seq+1, *)` `:189-191, 402-406`). Success reduction mirrors `:534-560`; setup mirrors `:753-783`.
  - `OBS_STATE` (`utils/constants.py:23`), `PreTrainedConfig.from_pretrained` (`configs/policies.py:172`),
    `make_pre_post_processors` (`policies/factory.py:150`), `make_policy` (`:240`).
- **Task B facts:**
  - pusht env `envs/configs.py:191`; metaworld env `:454`, `envs/metaworld.py:286`.
  - MetaWorld scripted experts: `envs/metaworld.py:24, 67-73, 107`.
  - Extras: `lerobot-0.6.1.dist-info/METADATA:260-271`.

## Method sources (m1-core entrants)
- **CUPID** (reimplementation): Agia et al., arXiv 2506.19121v2.
  - Paper: Def. 1, Eq. 2-5, App. A.1-A.2 (TRAK, d = 4000, Gauss-Newton, no damping), Sec. 6 / App. B.4 (m = 100).
  - Reference code github.com/agiachris/cupid@2941eba9427a3398d3cbe26b10c01e3ece56bd18 (MIT):
    `scripts/train/train_trak.sh:309,311,315`; `diffusion_policy/common/error_util.py:17-19, 311-330`
    ("net", "sum_of_sum"); `notebooks/data_curation.ipynb` cell 5.
- **DemInf** (reimplementation): Hejna et al., arXiv 2502.08623v3.
  - Paper: Sec. V-A/B/C, App. D, Table I (RoboMimic state: z_s 12, z_a 6).
  - Reference code github.com/jhejna/demonstration-information@232151defb258b5ceb0a6b62644574b57d981b58 (MIT):
    `scripts/quality/quality_estimators.py:39-55, 89-90, 165, 171`; `estimate_quality.py:39`;
    `configs/quality/vae_robomimic_state.py:32-35, 91, 94, 104`.
- **Demo-SCORE** (reimplementation; repo has no license): Chen, Lessing et al., arXiv 2503.03707v2, Sec. IV-A/B,
  App. A-B. Reference code github.com/alessing/demo-score@b82a5ece97b8975643b14703734afa8f24ab7b0a:
  - `demo_score/demo_score/train_stepwise_classifier.py:73-74` (num_eps 3000, hidden [8,8], dropout 0.3, lr 1e-4,
    wd 0.1, batch 1024), `:172` (code-only class weighting, not used), `:187`.
  - `filters/classifier_filter_lerobot.py:65-72`.
- **RA-BC**: lerobot `rewards/sarm/rabc.py` (SARM arXiv 2509.25358, Eq. 8-9). **Robometer**: lerobot
  `rewards/robometer`.
- **KSG**: Kraskov et al. 2004.
- **Checkers** (inspected read-only from published artifacts; not installed):
  - lerobot-doctor 0.2.0 (PyPI, Apache-2.0).
  - kinema 0.1.0 (PyPI, Apache-2.0).
  - roboaudit 0.3.1 (PyPI, MIT).
  - robot-data-audit 0.12.5 = github.com/liesliy/rda@47c69459e201cdc1f509f84b027c2407ab949e30 (MIT).

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
- QoQ (Lee et al., arXiv 2603.09056, ICRA 2026; no official code found) and DataMIL (Dass et al., arXiv 2505.09603,
  github.com/UT-Austin-RobIn/datamil, MIT) entrants: not implemented (m1-core scope).
- A loss(batch) adapter so CUPID can run on LeRobot PreTrainedPolicy (ACT); a reserved seed block for the
  policy-rollout mixture source instead of the top of the curation range.
- trajlens adapter via an isolated venv CLI (its pyarrow<25 pin conflicts with the core lock).
- Label diagnostics in the report (precision/recall of each cleaner's kept set vs labels, D-008).
- Replay-mode defect generation in sim (jitter/gripper drop) to build `bad`-tier episodes.
- A subset-variance-aware null simulation as a slow test, once open question 1 is decided.
