# Submitting a cleaner to RoboCurate-Bench

A cleaner is a Python class that picks training episodes, or assigns training weights, from a robot dataset. The
benchmark trains the same policy, with identical settings, on your choice and on size-matched random choices (the
placebo), and tests whether yours does better. The unit of analysis is a training run.

## 1. Write it

Start from [examples/cleaner_template](../examples/cleaner_template). There are two tracks (D-020).

**Selection track.** Subclass `robocurate.cleaners.base.Cleaner` or `ScoreCleaner`:

```python
class MyCleaner(ScoreCleaner):
    name, version = "my_cleaner", "1"
    def score(self, episode: EpisodeData) -> float:   # higher = keep
        ...
```

- `select(view, budget, rng)` must return exactly `budget` unique episode indices of `view`.
- It must be deterministic for a given `rng` state. The benchmark seeds `rng`.
- Ties in `ScoreCleaner` are broken by a salted content hash, never by episode index (D-021).

**Weighting track.** Subclass `WeightingCleaner` and implement `frame_progress(view, rng)`: one value in [0, 1]
per frame of every episode. Training uses LeRobot's RA-BC sample weighting on your progress. Your placebo is
your own progress curves randomly permuted across episodes (at least 8 permutations). A weighting entrant may change
only the weighting; every other training setting stays identical.

## 2. What a cleaner may and may not access

| May | May not |
|---|---|
| The label-free `DatasetView`: per episode, actions, state, task string and timestamps (read-only) | Ground-truth quality labels (`*.labels.json`) |
| `self.context.rollouts`: env rollouts on **curation seeds only**, within its declared cap | Eval code (`robocurate.eval`), eval seeds or eval results |
| `self.context.reference_policy`: a read-only full-data checkpoint, if declared | The oracle (`robocurate.runs.oracle`) |
| `self.context.progress_model`: the reward-model progress (Robometer), cached | Network calls to change its behaviour after results are seen |

`robocurate submit` enforces the forbidden imports statically, before importing your code, including transitively
through robocurate's own modules. It also flags `exec`, `eval`, `__import__` and `importlib`. The rollout service
rejects eval and calibration seeds and enforces the cap. Reference-policy files are made read-only and hash-checked.

## 3. Declare resources (D-022)

```python
requires_rollouts = 0             # int, <= 100 (protocol.max_curation_rollouts), curation seeds only
requires_reference_policy = False # True: receive the full-data checkpoint of the first train seed
requires_gpu = False              # True: the prepare stage runs on the GPU host; its GPU-h are reported
implementation = "original"       # original | wrapped | reimplementation
source = "paper / repo@commit"
```

Selections and weights are computed **once** per (task, budget, mixture replicate) by `robocurate prepare` and
cached by input hashes. Training runs only read that cache. The prepare stage's GPU-hours are shown on the
leaderboard as curation cost.

## 4. Validate

```bash
pip install -e path/to/your_package --no-deps
robocurate submit my_cleaner            # or: robocurate submit path/to/cleaner.py
```

The checks are:
- **static:** no forbidden imports; no dynamic import, `exec` or `eval`.
- **loads**
- **resources:** valid declarations.
- **count / unique / valid:** for selection-track cleaners.
- **progress:** shape and range, for weighting-track cleaners.
- **determinism:** two identically seeded runs give the same output.
- **runtime:** within the time limit (`--time-limit`).
- **no-leak:** no forbidden module loaded at runtime.

Conformance runs on a synthetic dataset with a fake context: synthetic rollouts, a small BC reference policy and
a mock progress model. It checks the contract, not your method's quality.

## 5. Propose it

1. **Open an issue** with the "Cleaner submission" form (`.github/ISSUE_TEMPLATE/cleaner_submission.yml`). It asks
   for:
   - the name, version, track and implementation type;
   - the citation (paper sections, repo@commit, license);
   - the declared resources;
   - your `robocurate submit` output;
   - three rule confirmations: no labels or eval access, no tuning on results, a license that allows the benchmark
     to run the code.
2. **Open a pull request** that adds your package as its own directory under `submissions/<name>/`. It must be an
   installable package whose `pyproject.toml` declares your cleaner in the `robocurate.cleaners` entry-point group,
   like `examples/cleaner_template`.
3. **CI runs conformance.** The `submit` workflow (`.github/workflows/submit.yml`) runs on pull requests that touch
   `submissions/**` or `examples/**`. It:
   - finds every changed package that declares `robocurate.cleaners` entry points;
   - installs each one with `pip install -e --no-deps` into the locked benchmark environment, on CPU;
   - runs `robocurate submit <name>` for each declared cleaner;
   - posts a PASS/FAIL table with the full output as the job summary.

   It has no GPU and no secrets, and its token is read-only. You can run the same check locally:

   ```bash
   python scripts/ci_submit.py --changed submissions/<name>/pyproject.toml --install
   ```

4. **Maintainers review** the method, the citation and the license, then add the entrant to the next grid version.
   An entrant must also pass its toybed sanity check with published defaults before it is enabled. A failing
   entrant is disabled and never tuned (see CUPID, D-029).

Do not tune anything on benchmark results (CLAUDE.md rule 9). Changes after results ship as a new version.

## 6. How results are produced

1. `robocurate prepare <grid>` caches your selection or weights.
2. `robocurate run <grid>` trains every condition, with identical training settings, on fixed seeds:
   - your entrant;
   - K >= 8 size-matched random subsets (or weight permutations);
   - full data;
   - a label oracle as a reference.

   Every condition uses the same number n >= 3 of training seeds (D-026).
3. Each final checkpoint is evaluated with at least 200 rollouts on protocol eval seeds.
4. `robocurate report --site` applies the primary test, the exact balanced prediction t-test (D-026,
   df = K − 1), Holm-corrected across the entrants of a track. It also reports a 95% CI, the minimum detectable
   effect and a pre-registered label: positive, negative, no detectable effect or inconclusive
   (docs/preregistration.md). Negative results are reported exactly like positive ones.
