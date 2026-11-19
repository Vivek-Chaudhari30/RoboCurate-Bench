# rcb-template-cleaner

A minimal RoboCurate-Bench cleaner: it keeps the `budget` episodes with the smoothest actions.
It is installable on its own and registers itself through the `robocurate.cleaners` entry point.

```bash
pip install -e examples/cleaner_template --no-deps   # inside the benchmark environment
robocurate submit template_smoothness                 # conformance checks; writes a selection JSON
```

To start your own cleaner:

1. Copy this directory and rename the package.
2. Change `name`, `version` and `score()` in `cleaner.py`.
3. Keep the entry-point key in `pyproject.toml` equal to `name`.

See [docs/submit.md](../../docs/submit.md) for the rules.
