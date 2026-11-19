#!/usr/bin/env bash
# Cache the GPU entrants' scores (Robometer progress for robometer_filter, rcb_score and the RA-BC weighting track)
# once per (task, budget, replicate) before the result grid (D-022). Usage:
#   scripts/prepare_gpu_scores.sh <grid.yaml> <local|fake> [--selections-dir DIR]
set -euo pipefail
cd "$(dirname "$0")/.."
[[ $# -ge 2 ]] || { echo "usage: $0 <grid.yaml> <local|fake> [extra prepare args]" >&2; exit 2; }
GRID=$1; BACKEND=$2; shift 2
[[ -n "${ROBOCURATE_SCRATCH:-}" && -n "${HF_LEROBOT_HOME:-}" ]] || { echo "source scripts/env.sh first" >&2; exit 2; }
ONLY=$(python - "$GRID" <<'PY'
import sys
from robocurate.cleaners import registry
from robocurate.config import load_grid
g = load_grid(sys.argv[1])
names = [c.name for c in g.condition_specs() if c.kind in ("cleaner", "weighting") and registry.get(c.name).requires_gpu]
print(",".join(names))
PY
)
[[ -n "$ONLY" ]] || { echo "no GPU entrants in $GRID"; exit 0; }
echo "GPU entrants: $ONLY"
exec robocurate prepare "$GRID" --backend "$BACKEND" --only "$ONLY" "$@"
