"""Policy-rollout mixture source with full observations, camera frames included (D-031, D-033, D-034).

Loads a LeRobot checkpoint (pretrained_model dir) exactly as `lerobot-eval` does (scripts/lerobot_eval.py:740-781:
make_env -> make_policy(cfg.policy, env_cfg) -> policy.eval() -> make_pre_post_processors(pretrained_path,
device override) -> make_env_pre_post_processors) and runs LeRobot's own `rollout()` with return_observations=True
(lerobot_eval.py:167-182, :275, :391-405) on DATA-GENERATION seeds (protocol.generation_seed_range, D-034).
Observations come back preprocessed (envs/utils.py:78-102: images float CHW in [0, 1]); they are converted back to
HWC uint8 and written with LeRobot's dataset API, one episode per seed up to its first `done` (the rollout runs
every env of the batch until all are done; lerobot_eval.py:198-200). Optional replay-mode perturbations (D-011)
wrap each sub-env, so the EXECUTED action is what is recorded and what the next observation follows.

Labels sidecar (beside the root): source "policy", checkpoint path + step, success, perturbation types/parameters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


class PerturbActionWrapper:
    """Wraps one gymnasium env: applies Perturbations to the action before env.step and logs executed actions."""

    def __init__(self, env, perturbations: list[dict], seed: int):
        from robocurate.data import defects as D
        from robocurate.data.mixture import make_perturbations

        self.env, self._specs, self._seed = env, [dict(p) for p in perturbations], int(seed)
        self._pp = D.PerturbedPolicy(lambda obs: self._a, make_perturbations(tuple(dict(p) for p in self._specs)),
                                     seed=self._seed)
        self.episodes: list[list[np.ndarray]] = []  # one list per reset (vector envs autoreset at episode end)
        self._a = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def executed(self) -> list[np.ndarray]:
        """Executed actions of the first episode since construction (the recorded one)."""
        return self.episodes[0] if self.episodes else []

    def reset(self, *args, **kwargs):
        self._pp.reset()
        self.episodes.append([])
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        self._a = np.asarray(action, dtype=float)
        a = np.asarray(self._pp(None), dtype=np.float32)
        self.episodes[-1].append(a.copy())
        return self.env.step(a)


@dataclass
class RecordedEpisode:
    seed: int
    images: dict[str, np.ndarray]  # key -> (T, H, W, 3) uint8
    states: np.ndarray
    actions: np.ndarray
    success: bool
    checkpoint: str
    checkpoint_step: int
    perturbations: list[dict] = field(default_factory=list)


def load_checkpoint(pretrained_dir: str | Path, env_cfg: Any, device: str = "cpu"):
    """Policy + processors as lerobot-eval builds them (lerobot_eval.py:760-781)."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(str(pretrained_dir))
    cfg.pretrained_path = str(pretrained_dir)
    cfg.device = device
    policy = make_policy(cfg=cfg, env_cfg=env_cfg)
    policy.eval()
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(pretrained_dir),
                                         preprocessor_overrides={"device_processor": {"device": device}})
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)
    return policy, pre, post, env_pre, env_post


def record(pretrained_dir: str | Path, checkpoint_step: int, env_type: str, env_task: str, seeds: list[int],
           *, protocol: Any, episode_length: int | None = None, perturbations: list[dict] | None = None,
           device: str = "cpu", batch_size: int = 1) -> list[RecordedEpisode]:
    import torch
    from lerobot.envs.factory import make_env, make_env_config
    from lerobot.scripts.lerobot_eval import rollout

    lo, hi = protocol.generation_seed_range  # D-034: data-generation rollouts only
    bad = [s for s in seeds if not lo <= s <= hi]
    if bad:
        raise ValueError(f"seeds {bad[:3]} outside generation_seed_range [{lo}, {hi}] (D-034)")
    kw = {"task": env_task}
    if episode_length is not None:
        kw["episode_length"] = int(episode_length)
    env_cfg = make_env_config(env_type, **kw)
    policy, pre, post, env_pre, env_post = load_checkpoint(pretrained_dir, env_cfg, device)
    out: list[RecordedEpisode] = []
    for b0 in range(0, len(seeds), batch_size):
        bs = [int(s) for s in seeds[b0:b0 + batch_size]]
        (_, d), = make_env(env_cfg, n_envs=len(bs)).items()
        vec = d[0]
        wraps = None
        if perturbations:  # replace the sub-envs of the innermost SyncVectorEnv (make_env may wrap the vector env)
            inner = getattr(vec, "unwrapped", vec)
            wraps = [PerturbActionWrapper(e, perturbations, s) for e, s in zip(inner.envs, bs, strict=True)]
            inner.envs = wraps
        with torch.no_grad():
            r = rollout(vec, policy, env_pre, env_post, pre, post, seeds=bs, return_observations=True)
        obs = r["observation"]
        done = r["done"].cpu().numpy()
        succ = r["success"].cpu().numpy()
        for i, s in enumerate(bs):
            T = int(np.argmax(done[i])) + 1 if done[i].any() else done.shape[1]
            imgs = {k: (obs[k][i, :T].permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
                    for k in obs if k.startswith("observation.image")}
            acts = (np.stack(wraps[i].executed[:T]) if wraps else r["action"][i, :T].cpu().numpy()).astype(np.float32)
            out.append(RecordedEpisode(s, imgs, obs["observation.state"][i, :T].cpu().numpy().astype(np.float32),
                                       acts, bool(succ[i, :T].any()), str(pretrained_dir), int(checkpoint_step),
                                       [dict(p) for p in (perturbations or [])]))
        vec.close()
    return out


def write_rollout_dataset(episodes: list[RecordedEpisode], repo_id: str, root: str | Path, fps: int, task: str,
                          shuffle_seed: int, robot_type: str | None = None) -> Path:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from robocurate.data.labels import TIERS, EpisodeLabel, LabelsSidecar, sidecar_path, write_labels
    from robocurate.data.mixture import DEFECT_TIER

    root = Path(root)
    if root.exists():
        raise FileExistsError(f"{root} exists")
    e0 = episodes[0]
    features = {k: {"dtype": "video", "shape": tuple(v.shape[1:]), "names": ["height", "width", "channel"]}
                for k, v in e0.images.items()}
    features["observation.state"] = {"dtype": "float32", "shape": (e0.states.shape[1],), "names": None}
    features["action"] = {"dtype": "float32", "shape": (e0.actions.shape[1],), "names": None}
    ds = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=features, root=root, robot_type=robot_type,
                               use_videos=True)
    order = np.random.default_rng(shuffle_seed).permutation(len(episodes))
    labels = []
    try:
        for out_i, j in enumerate(order):
            ep = episodes[int(j)]
            for t in range(len(ep.actions)):
                frame = {"task": task, "observation.state": ep.states[t], "action": ep.actions[t]}
                frame.update({k: v[t] for k, v in ep.images.items()})
                ds.add_frame(frame)
            ds.save_episode()
            kinds = tuple(sorted({p["type"] for p in ep.perturbations}))
            labels.append(EpisodeLabel(out_i, "policy", ep.checkpoint, ep.seed, defects=kinds,
                                       quality_tier=max([DEFECT_TIER[d] for d in kinds] or ["clean"], key=TIERS.index),
                                       checkpoint_step=ep.checkpoint_step, success=ep.success, perturbations=kinds,
                                       perturbation_params=json.dumps(ep.perturbations, sort_keys=True)))
    finally:
        ds.finalize()
    write_labels(LabelsSidecar(dataset=repo_id, recipe_sha256="policy-rollouts", seed=shuffle_seed, episodes=labels),
                 sidecar_path(root))
    return root
