# ruff: noqa
"""validate-defects on the CPU-buildable part of task_a_v1 (actions/state/timestamps only; offline HF cache).
The video build is blocked tonight (huggingface.co TLS), so the plan's post-hoc defects are applied to the cached
parquet episodes exactly as build_mixture applies them (data/defects.inject_pause on frames), timestamps
regenerated at the dataset fps (as LeRobot's add_frame does)."""
import json, glob
import numpy as np
from robocurate.data import mixture as M
from robocurate.data.defects import inject_pause
from robocurate.data.validate import Episode, analyze, load_parquet_dataset
REV = {"lerobot/aloha_sim_transfer_cube_scripted": "c5467ba9059e4c65f27775ced212076980a173fb",
       "lerobot/aloha_sim_transfer_cube_human": "6a43d500f101255823a9d2b9dc244eeb01a2cd31"}
import os
H = os.environ["HF_HOME"]; L = os.environ["HF_LEROBOT_HOME"]
roots = {"lerobot/aloha_sim_transfer_cube_scripted": glob.glob(f"{H}/hub/datasets--lerobot--aloha_sim_transfer_cube_scripted/snapshots/{REV['lerobot/aloha_sim_transfer_cube_scripted']}")[0],
         "lerobot/aloha_sim_transfer_cube_human": glob.glob(f"{L}/hub/datasets--lerobot--aloha_sim_transfer_cube_human/snapshots/*")[0]}
src = {k: load_parquet_dataset(v) for k, v in roots.items()}
recipe = M.Recipe.load("configs/mixtures/task_a_v1.yaml")
plan = M.plan_mixture(recipe, {k: [len(e.actions) for e in eps] for k, (info, eps) in src.items()})
fps = float(src["lerobot/aloha_sim_transfer_cube_human"][0]["fps"])
out, labels = [], []
for p in plan:
    info, eps = src[p.source_dataset]
    e = next(x for x in eps if x.index == p.source_episode)
    frames = [{"s": s, "a": a} for s, a in zip(e.state, e.actions)]
    if "pause" in p.defects:
        frames = inject_pause(frames, p.pause_t, p.pause_k)
    n = len(frames)
    out.append(Episode(p.out_index, e.task, np.arange(n) / fps, np.stack([f["s"] for f in frames]), np.stack([f["a"] for f in frames])))
    labels.append(p.defects)
mix = analyze({"fps": fps, "codebase_version": "v3.0", "robot_type": "aloha"}, out)
base = {k: analyze(info, eps) for k, (info, eps) in src.items()}
row = lambda a: {"n": a["n_episodes"], "pause_any": round(a["pause"]["episode_rate"], 3), "pause_interior": round(a["pause"]["interior_episode_rate"], 3),
                 "interior_s_p50": round(a["pause"]["interior_segment_seconds"].get("p50", float("nan")), 2),
                 "jitter>0.10": a["jitter"]["episodes_above_0.10"], "hf_share_p50": float("%.2e" % a["jitter"]["hf_power_share"]["p50"]),
                 "exact_dups": a["duplicates"]["exact_duplicate_episodes"], "near_dup_pairs": a["duplicates"]["near_duplicate_pairs"]}
print(json.dumps({"planned": {"episodes": len(plan), "paused": sum("pause" in d for d in labels), "duplicates": sum("duplicate" in d for d in labels)},
                  "task_a_v1 CPU part": row(mix), **{k.split('/')[1]: row(v) for k, v in base.items()}}, indent=1))
