#!/usr/bin/env bash
# GPU-phase policy-rollout mixture source (D-031): the M0 full-data seed-1000 checkpoints at 10/30/60/100% of S
# (D-033, kept by launch-kit stage e) rolled out on generation seeds (D-034) with camera frames recorded.
#   scripts/gpu_phase_rollouts.sh --recipe configs/mixtures/task_a_v1.yaml --task configs/tasks/aloha_transfer_cube.yaml \
#       --out $ROBOCURATE_SCRATCH/sources/task_a_v1 --m0-run-dir $ROBOCURATE_SCRATCH/runs/<seed-1000 full rid> --steps <S>
#   dry run (CPU): add --fake-checkpoints --max-episodes 1 --episode-length 8 (writes tiny random checkpoints first)
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -n "${ROBOCURATE_SCRATCH:-}" && -n "${HF_LEROBOT_HOME:-}" ]] || { echo "source scripts/env.sh first" >&2; exit 2; }
if [[ "$(uname)" == "Darwin" ]]; then export MUJOCO_GL=disable; fi  # CPU software renderer (robocurate.sim.cpu_render)
exec python -m robocurate.launch.rollouts "$@"
