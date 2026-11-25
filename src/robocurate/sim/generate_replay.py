"""Stage 2 of the scripted replay-mode generation (scripts/generate_scripted_replay.sh): replay stage-1 end-effector
joint trajectories in the benchmark env with the spec's perturbations (D-011, D-031, D-037) on generation seeds
(D-034), and write the LeRobot dataset + labels sidecar (robocurate.sim.scripted_aloha.write_dataset).
`--validate-dir` replays clean stage-1 trajectories and reports the success rate and path deviations only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def validate(ee_dir: Path) -> dict:
    import numpy as np

    from robocurate.sim import scripted_aloha as SA

    rows = []
    for f in sorted(ee_dir.glob("ee_*.npz")):
        ee, ver = SA.load_ee(f)
        ep = SA.replay(ee, record_images=False)
        rows.append({"seed": ee.seed, "stage1_mujoco": ver, "ee_success": ee.ee_success, "success": ep.success,
                     **{k: v for k, v in ep.meta.items() if k.startswith("path")}})
    return {"n": len(rows), "success_rate": float(np.mean([r["success"] for r in rows])) if rows else None,
            "episodes": rows}


def generate(ee_dir: Path, spec_path: str, out: Path, protocol_path: str, shuffle_seed: int, seed_offset: int = 0):
    import yaml

    from robocurate.config import load_protocol
    from robocurate.data import mixture as M
    from robocurate.sim import scripted_aloha as SA

    spec = M._check_deferred(yaml.safe_load(Path(spec_path).read_text()), spec_path)
    proto = load_protocol(protocol_path)
    srcs = M.replay_policy_sources(spec, proto, seed_offset)
    seeds = M.policy_rollout_seeds(proto, len(srcs), seed_offset)
    eps = []
    for src, seed in zip(srcs, seeds, strict=True):
        ee, ver = SA.load_ee(ee_dir / f"ee_{seed}.npz")
        if ver != "2.3.3":
            raise SystemExit(f"stage 1 for seed {seed} ran under mujoco {ver}; ACT's procedure needs 2.3.3")
        ep = SA.replay(ee, perturbations=[dict(p) for p in src.perturbations])
        ep.meta["stage1_mujoco"] = ver
        eps.append(ep)
    root = SA.write_dataset(eps, "robocurate/task_a_v1_scripted_replay", out, shuffle_seed=shuffle_seed)
    return root, [{"seed": e.seed, "perturbations": e.perturbations, "success": e.success,
                   **{k: v for k, v in e.meta.items() if k.startswith("path")}} for e in eps]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="generate_replay")
    ap.add_argument("--ee-dir", required=True)
    ap.add_argument("--spec", help="replay spec (configs/mixtures/task_a_v1_scripted_replay.yaml)")
    ap.add_argument("--out", help="dataset root to create")
    ap.add_argument("--validate", action="store_true", help="only replay clean trajectories and report success")
    ap.add_argument("--protocol", default="configs/protocol.yaml")
    ap.add_argument("--shuffle-seed", type=int, default=31, help="task_a_v1 recipe seed (D-021)")
    a = ap.parse_args(argv)
    from robocurate.sim.scripted_aloha import ensure_cpu_renderer

    ensure_cpu_renderer()
    if a.validate:
        print(json.dumps(validate(Path(a.ee_dir)), indent=1))
        return 0
    if not (a.spec and a.out):
        raise SystemExit("--spec and --out are required unless --validate")
    root, rows = generate(Path(a.ee_dir), a.spec, Path(a.out), a.protocol, a.shuffle_seed)
    print(json.dumps({"root": str(root), "episodes": rows}, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
