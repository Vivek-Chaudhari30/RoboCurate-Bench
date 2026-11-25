#!/usr/bin/env bash
# Replay-mode scripted episodes for task_a_v1 (D-031, D-037), two stages (robocurate.sim.scripted_aloha):
#   stage 1: end-effector rollouts under ACT's pinned simulator (mujoco 2.3.3 / dm_control 1.0.9; ACT's own code
#            succeeds there and fails under mujoco 3.8.1); needs a separate Python env, ACT_SIM_PYTHON.
#   stage 2: joint replay + recording in the benchmark env (gym-aloha 0.1.5, mujoco 3.8.1), CPU software renderer.
#   ACT_SIM_PYTHON=$ROBOCURATE_SCRATCH/venvs/mj233/bin/python scripts/generate_scripted_replay.sh <out_dir>
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?usage: $0 <out_dir>}
[[ -n "${ACT_SIM_PYTHON:-}" ]] || { echo "set ACT_SIM_PYTHON to a python with mujoco==2.3.3 dm_control==1.0.9 gym-aloha==0.1.5 (--no-deps)" >&2; exit 2; }
SEEDS=$(python -c "
from robocurate.config import load_protocol; from robocurate.data.mixture import policy_rollout_seeds
print(' '.join(map(str, policy_rollout_seeds(load_protocol('configs/protocol.yaml'), 10, 0))))")
MUJOCO_GL=disable PYTHONPATH=src "$ACT_SIM_PYTHON" -m robocurate.sim.scripted_aloha ee --seeds $SEEDS --out "$OUT/ee_replay"
export MUJOCO_GL=disable
exec python -m robocurate.sim.generate_replay --ee-dir "$OUT/ee_replay" --spec configs/mixtures/task_a_v1_scripted_replay.yaml --out "$OUT/scripted_replay"
