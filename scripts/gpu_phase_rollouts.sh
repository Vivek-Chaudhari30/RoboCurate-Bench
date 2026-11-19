#!/usr/bin/env bash
# GPU-phase mixture sources (D-031): policy rollouts from the M0 full-data seed-1000 checkpoints at 10/30/60/100% of
# training, and the replay-mode scripted episodes. Curation seeds only (top of curation_seed_range, offset per source).
#   scripts/gpu_phase_rollouts.sh --recipe configs/mixtures/task_a_v1.yaml --out $ROBOCURATE_SCRATCH/sources/task_a_v1 \
#       --backend local --m0-run-dir $ROBOCURATE_SCRATCH/runs/<rid> --steps <S>
#   dry run: --backend fake (stub env/policies; checks seeds, labels and the LeRobot writer)
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -n "${ROBOCURATE_SCRATCH:-}" && -n "${HF_LEROBOT_HOME:-}" ]] || { echo "source scripts/env.sh first" >&2; exit 2; }
exec python -m robocurate.launch.rollouts "$@"
