#!/usr/bin/env bash
# GPU-phase mixture sources (D-031): policy rollouts from the M0 full-data seed-1000 checkpoints at 10/30/60/100% of
# training, and the replay-mode scripted episodes. Curation seeds only (top of curation_seed_range, offset per source).
#   scripts/gpu_phase_rollouts.sh --recipe configs/mixtures/task_a_v1.yaml --out $ROBOCURATE_SCRATCH/sources/task_a_v1 \
#       --backend local --env-type aloha --env-task AlohaTransferCube-v0 --device cuda \
#       --m0-run-dir $ROBOCURATE_SCRATCH/runs/<rid> --steps <S>
#   dry run: --backend fake (stub env/policies; checks seeds, labels and the LeRobot writer; stays on the cpu default)
# --backend local adds --device cuda when the caller did not already pass --device.
# CUDA_VISIBLE_DEVICES does not select the device; an empty value only refuses an explicit cuda request.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -n "${ROBOCURATE_SCRATCH:-}" && -n "${HF_LEROBOT_HOME:-}" ]] || { echo "source scripts/env.sh first" >&2; exit 2; }

backend_local=0
device_set=0
prev=""
for arg in "$@"; do
  if [[ "$prev" == "--backend" && "$arg" == "local" ]]; then
    backend_local=1
  fi
  case "$arg" in
    --backend=local) backend_local=1 ;;
    --device=*) device_set=1 ;;
  esac
  if [[ "$prev" == "--device" ]]; then
    device_set=1
  fi
  prev=$arg
done

if [[ "$backend_local" == 1 && "$device_set" == 0 ]]; then
  exec python -m robocurate.launch.rollouts "$@" --device cuda
fi
exec python -m robocurate.launch.rollouts "$@"
