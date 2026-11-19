# Copy to scripts/env.sh (gitignored), edit, then: source scripts/env.sh
# Storage rule (CLAUDE.md): scratch must be outside the repo; on hokkaido, on the scratch disk (not $HOME).
export ROBOCURATE_SCRATCH="${ROBOCURATE_SCRATCH:-$HOME/robocurate-scratch}"
export HF_HOME="$ROBOCURATE_SCRATCH/hf"
export HF_LEROBOT_HOME="$ROBOCURATE_SCRATCH/lerobot"   # read by lerobot/utils/constants.py:69
export MUJOCO_GL=egl
export PYTHONNOUSERSITE=1   # keep ~/.local site-packages out of the env
# This task is CPU-only. Remove only when a task prompt explicitly allows GPU use.
export CUDA_VISIBLE_DEVICES=""
