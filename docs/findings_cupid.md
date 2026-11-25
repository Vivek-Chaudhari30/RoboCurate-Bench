# Finding: CUPID's official configuration does not recover planted harmful demos on our toybeds

Status: **CUPID is disabled** for every policy family (D-029, D-036). This page reports the evidence. It is a
finding about one configuration of an influence method on two controlled toybeds. It is not a verdict on the
CUPID paper's results.

## What was tested

**Method.** CUPID (Agia et al., CoRL 2025, arXiv 2506.19121), reimplemented in
`src/robocurate/cleaners/entrants/cupid.py`. The official settings were read from agiachris/cupid@2941eba:
- `scripts/train/train_trak.sh:309,311,315-316`: projection dimension 4000, lambda_reg 0.0, model output "square",
  64 timesteps;
- `diffusion_policy/data_attribution/modelout_functions.py:79-82, 116-119, 128-136`: fresh noise; square output
  = mean of the squared denoiser prediction; Q = 1;
- `gradient_computers.py:110-128`: per-sample gradients averaged over timesteps;
- `error_util.py:17-19, 299-330`: sum_of_sum aggregation; "net" = successes minus failures;
- TRAK at agiachris/trak@ff2fdc84 `trak/score_computers.py:157-166`: float32 inverse of X^T X + lambda I;
- m = 100 rollouts (paper Sec. 6).

**Pass rule** (task prompt, fixed before the runs): the AUROC of ranking the planted wrong-direction demos as the
most harmful must be at least 0.8.

## Toybed 1: ACT-style unimodal policy (MLP)

- *Setup.* A 2D reach environment; 30 clean, 6 pause, 6 jitter and 8 wrong-direction demos; an MLP BC policy.
- *Premise.* The wrong demos are harmful. Success is 0.39–0.43 with all data and 0.75–0.83 without them (3 train
  seeds, 300 toy seeds).
- *Result.* CUPID does not separate them. At a 50% budget it keeps 62.5% of wrong demos and 46.7% of clean ones.
  No variant recovers them: the exact Hessian, Gauss-Newton, no projection, the "square" analogue, and m = 1000 or
  4000 rollouts all give AUROC about 0.5.
- *Formula check.* With the exact Hessian, our influence predicts brute-force leave-one-out refits of a linear
  least-squares policy (correlation > 0.99, sign agreement > 95%;
  `tests/test_entrants.py::test_cupid_influence_matches_leave_one_out_refit`). The implementation of the formula is
  not the problem.

## Toybed 2: diffusion policy (CUPID's native setting)

- *Setup.* 240 clean + 60 wrong-direction demos (a distinct mode), 4230 frames, more than d = 4000. A DDPM toy
  policy (`tests/toybed/diffusion.py`; LeRobot's Diffusion defaults: 100 timesteps, squaredcos_cap_v2, epsilon,
  clip 1). The setup was fixed before the first CUPID run.
- *Premise.* Success is 0.07 with all data and 1.00 without the wrong mode (3 seeds, 200 eval seeds).
- *Result.* AUROC **0.514 / 0.518 / 0.518** (policy seeds 0/1/2) with the official configuration. With TRAK's exact
  float32 inverse it is **0.482 / 0.500**. FAIL.
- *Diagnosis.* The Gram matrix Phi^T Phi is numerically singular even though N > d: its minimum eigenvalue is about
  −3e−5 against a maximum of 7e3. With lambda = 0, its inverse amplifies noise directions.

**Diagnostic only, not adopted.** Changing these knobs would tune the method on the toybed, and D-036 rejects
that.

| variant | AUROC |
|---|---|
| "ddpm" output instead of "square" | 0.512 |
| d = 400 | 0.586 |
| 4x data (N = 16,939) | 0.470 |
| damping lambda = 1 | 0.578 |
| damping lambda = 10 | 0.714 / 0.779 / 0.724 |
| damping lambda = 100 | 0.819 |
| damping lambda = 1e6 (H ≈ lambda·I, i.e. plain gradient similarity) | 1.000 |

The "square" features carry the signal; the undamped inverse removes it.

## What we did instead (D-036)

The pre-registered influence entrant is **gradsim-return**
(`src/robocurate/cleaners/entrants/gradsim_return.py`). It keeps CUPID's return-weighted rollout objective and
CUPID's model-output gradients, compares them with a plain dot product (TracIn, Pruthi et al., NeurIPS 2020, arXiv
2002.08484), and has no Hessian, no Gram inverse, no projection and no parameter of its own.

| toybed | gradsim-return AUROC | enabled (D-036: >= 0.8) |
|---|---|---|
| diffusion | **1.000** | yes, for policy family `diffusion` |
| MLP / ACT-style | **0.383** | no, for `act` |

## Limitations of this finding

- **Toybeds only.** These are two 2D toybeds with planted defects. The real tasks (ALOHA, PushT) were not used for
  this, because doing so would tune on benchmark data.
- **Our reimplementation.** The official code is built into the Diffusion Policy codebase and was read, not run.
  Our diffusion toybed policy is an MLP denoiser, not the official 1D U-Net.
- **Unimodal policies.** On the ACT-style toybed, neither CUPID nor gradsim-return recovers the wrong demos. This is
  consistent with first-order influence being unable to see a wrong-direction demo in a unimodal policy, and we
  report it as a limitation of influence-based curation with such policies (D-029).

Raw numbers and diagnostics scripts: PROGRESS.md ("CUPID sanity investigation", "CUPID diffusion toybed",
overnight 3 item 6).
