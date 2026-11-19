#!/usr/bin/env bash
# M0 calibration pipeline, stages a-f (src/robocurate/launch/m0.py; docs/runbook_gpu.md).
#
#   Real run (hokkaido, inside tmux "rcb-m0"; CLAUDE.md: anything > 10 min runs in tmux):
#     source scripts/env.sh            # BEFORE python: lerobot reads HF_LEROBOT_HOME at import (constants.py:69)
#     scripts/m0_pipeline.sh --task a  # Task A (ALOHA transfer-cube + ACT)
#     scripts/m0_pipeline.sh --task b  # Task B (PushT + Diffusion Policy, D-030)
#   Dry run (any CPU machine; fake backend, temp scratch, nothing committed):
#     scripts/m0_pipeline.sh --task a --dry-run --fake-params configs/fake/m0_dryrun.yaml --state-dir /tmp/x
#
# Requires ROBOCURATE_SCRATCH=/share/datasets/robocurate (D-027) for real runs.
set -euo pipefail
cd "$(dirname "$0")/.."

DRY=0
for a in "$@"; do [[ "$a" == "--dry-run" ]] && DRY=1; done

die() { echo "m0_pipeline: $*" >&2; exit 2; }
[[ -n "${ROBOCURATE_SCRATCH:-}" && -n "${HF_HOME:-}" && -n "${HF_LEROBOT_HOME:-}" ]] \
  || die "source scripts/env.sh first (ROBOCURATE_SCRATCH / HF_HOME / HF_LEROBOT_HOME unset)"

if [[ $DRY -eq 0 ]]; then
  [[ "$(realpath "$ROBOCURATE_SCRATCH")" == "/share/datasets/robocurate" ]] \
    || die "ROBOCURATE_SCRATCH must be /share/datasets/robocurate (D-027), got $ROBOCURATE_SCRATCH"
  [[ -n "${TMUX:-}" ]] || die "run inside tmux session rcb-m0 (CLAUDE.md: > 10 min runs in tmux)"
  [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || die "CUDA_VISIBLE_DEVICES is empty (env.sh disables the GPU on CPU-only tasks)"
  [[ -z "$(git status --porcelain)" ]] || die "git tree not clean (results record the git hash of a clean tree)"
  exec python -m robocurate.launch.m0 --backend local "$@"
else
  export CUDA_VISIBLE_DEVICES="" PYTORCH_ENABLE_MPS_FALLBACK=0
  exec python -m robocurate.launch.m0 --backend fake "$@"
fi
