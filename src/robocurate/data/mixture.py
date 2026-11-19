"""Deterministic mixed-quality dataset recipes.

A recipe (YAML) names source datasets (scripted demos, human demos, policy rollouts), how many episodes to
draw from each, and which post-hoc defects to inject (D-011: pause, duplicate, wrong_task only). The recipe
plus its seed fully determine the output episode order and the labels sidecar (`plan_mixture` is pure and
does not touch LeRobot; equal inputs give byte-identical labels JSON).

`build_mixture` materializes the plan as a NEW LeRobot-format dataset through LeRobot's own writing API:
LeRobotDataset.create (datasets/lerobot_dataset.py:684) -> add_frame (:418) -> save_episode (:434) ->
finalize (:474). The labels sidecar is written BESIDE the dataset root (data/labels.py), never inside it.

Quality tier per output episode = worst tier over its defects:
    pause -> degraded, duplicate -> degraded, wrong_task -> degraded; no defect -> clean.
(Execution-time defects -- jitter, gripper_drop -> bad; hesitation -> degraded -- come from replay-mode
sources and are not produced by this module.)
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from robocurate.config import ConfigError, load_yaml
from robocurate.data import defects as D
from robocurate.data.labels import TIERS, EpisodeLabel, LabelsSidecar, sidecar_path, write_labels

DEFECT_TIER = {
    "pause": "degraded",
    "duplicate": "degraded",
    "wrong_task": "degraded",
    "jitter": "bad",
    "gripper_drop": "bad",
    "hesitation": "degraded",
}
DEFAULT_FEATURE_KEYS = ("timestamp", "frame_index", "episode_index", "index", "task_index")  # constants.py:81


@dataclass(frozen=True)
class SourceSpec:
    dataset: str
    source: str  # scripted | human | policy
    n_episodes: int
    revision: str | None = None


@dataclass(frozen=True)
class Recipe:
    name: str
    repo_id: str
    seed: int
    sources: tuple[SourceSpec, ...]
    wrong_task_strings: tuple[str, ...] = ()
    pause: dict[str, int] = field(default_factory=dict)  # n_episodes, k_min, k_max
    duplicate: dict[str, int] = field(default_factory=dict)  # n_episodes
    wrong_task: dict[str, int] = field(default_factory=dict)  # n_episodes
    sha256: str = ""

    @classmethod
    def load(cls, path: str | Path) -> Recipe:
        raw = Path(path).read_bytes()
        d = load_yaml(path)
        allowed = {"name", "repo_id", "seed", "sources", "wrong_task_strings", "defects"}
        if set(d) - allowed:
            raise ConfigError(f"{path}: unknown keys {sorted(set(d) - allowed)}")
        dd = d.get("defects", {}) or {}
        if set(dd) - {"pause", "duplicate", "wrong_task"}:
            raise ConfigError(
                f"{path}: only pause/duplicate/wrong_task can be injected post hoc (D-011); got {sorted(dd)}"
            )
        return cls(
            name=d["name"],
            repo_id=d["repo_id"],
            seed=int(d["seed"]),
            sources=tuple(SourceSpec(**s) for s in d["sources"]),
            wrong_task_strings=tuple(d.get("wrong_task_strings", ())),
            pause=dict(dd.get("pause", {})),
            duplicate=dict(dd.get("duplicate", {})),
            wrong_task=dict(dd.get("wrong_task", {})),
            sha256=hashlib.sha256(raw).hexdigest(),
        )


@dataclass(frozen=True)
class EpisodePlan:
    out_index: int
    source: str
    source_dataset: str
    source_episode: int
    defects: tuple[str, ...]
    pause_t: int | None = None
    pause_k: int | None = None
    wrong_task: str | None = None
    duplicate_of: int | None = None  # out_index of the original

    @property
    def quality_tier(self) -> str:
        tiers = [DEFECT_TIER[d] for d in self.defects] or ["clean"]
        return max(tiers, key=TIERS.index)


def plan_mixture(recipe: Recipe, source_lengths: dict[str, list[int]]) -> list[EpisodePlan]:
    """Pure, deterministic plan. `source_lengths[dataset]` = episode lengths of that source."""
    rng = np.random.default_rng(recipe.seed)
    base: list[tuple[SourceSpec, int]] = []
    for s in recipe.sources:
        total = len(source_lengths[s.dataset])
        if s.n_episodes > total:
            raise ConfigError(f"{s.dataset}: asks for {s.n_episodes} of {total} episodes")
        picks = np.sort(rng.choice(total, size=s.n_episodes, replace=False))
        base += [(s, int(e)) for e in picks]
    order = rng.permutation(len(base))
    items = [base[i] for i in order]
    n = len(items)

    n_pause = int(recipe.pause.get("n_episodes", 0))
    n_wrong = int(recipe.wrong_task.get("n_episodes", 0))
    n_dup = int(recipe.duplicate.get("n_episodes", 0))
    if n_pause + n_wrong + n_dup > n:
        raise ConfigError(f"{n_pause + n_wrong + n_dup} defect episodes requested from {n} base episodes")
    if n_wrong and not recipe.wrong_task_strings:
        raise ConfigError("wrong_task defects need wrong_task_strings")
    targets = rng.permutation(n)
    pause_ids = set(int(i) for i in targets[:n_pause])
    wrong_ids = set(int(i) for i in targets[n_pause : n_pause + n_wrong])
    dup_src = [int(i) for i in targets[n_pause + n_wrong : n_pause + n_wrong + n_dup]]  # clean originals

    plans: list[dict[str, Any]] = []
    for i, (s, e) in enumerate(items):
        p: dict[str, Any] = dict(source=s.source, source_dataset=s.dataset, source_episode=e, defects=())
        if i in pause_ids:
            L = source_lengths[s.dataset][e]
            p.update(
                defects=("pause",),
                pause_t=int(rng.integers(int(0.2 * L), int(0.8 * L))),
                pause_k=int(rng.integers(int(recipe.pause["k_min"]), int(recipe.pause["k_max"]) + 1)),
            )
        if i in wrong_ids:
            p.update(
                defects=("wrong_task",),
                wrong_task=str(recipe.wrong_task_strings[int(rng.integers(len(recipe.wrong_task_strings)))]),
            )
        p["_orig"] = i
        plans.append(p)
    # insert duplicates at random positions (after their original is fine either way)
    for src in dup_src:
        copy = dict(plans[[q["_orig"] for q in plans].index(src)])
        copy.update(defects=("duplicate",), _dup_of_orig=src, _orig=None)
        pos = int(rng.integers(0, len(plans) + 1))
        plans.insert(pos, copy)
    orig_to_out = {q["_orig"]: k for k, q in enumerate(plans) if q.get("_orig") is not None}
    out = []
    for k, q in enumerate(plans):
        dup_of = orig_to_out[q["_dup_of_orig"]] if "_dup_of_orig" in q else None
        q = {kk: v for kk, v in q.items() if not kk.startswith("_")}
        out.append(EpisodePlan(out_index=k, duplicate_of=dup_of, **q))
    return out


def labels_for(recipe: Recipe, plan: list[EpisodePlan]) -> LabelsSidecar:
    return LabelsSidecar(
        dataset=recipe.repo_id,
        recipe_sha256=recipe.sha256,
        seed=recipe.seed,
        episodes=[
            EpisodeLabel(p.out_index, p.source, p.source_dataset, p.source_episode, p.defects, p.quality_tier)
            for p in plan
        ],
    )


def plan_json(plan: list[EpisodePlan]) -> str:
    return json.dumps([p.__dict__ for p in plan], sort_keys=True, indent=2)


# ---------------------------------------------------------------------------- materialize (LeRobot I/O)


def _to_hwc_uint8(img) -> np.ndarray:
    a = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8)
    return a


def _episode_frames(ds, meta, episode: int, user_keys: list[str], video_keys: list[str]) -> list[dict]:
    from robocurate.data.hub import episode_frame_range

    lo, hi = episode_frame_range(meta, episode)
    a2r = ds.absolute_to_relative_idx
    frames = []
    for absolute in range(lo, hi):
        item = ds[a2r[absolute] if a2r is not None else absolute]
        if int(item["episode_index"]) != episode:
            raise RuntimeError(f"frame {absolute}: expected episode {episode}, got {int(item['episode_index'])}")
        f: dict[str, Any] = {"task": item["task"]}
        for k in user_keys:
            v = item[k]
            if k in video_keys:
                f[k] = _to_hwc_uint8(v)
            else:
                arr = v.numpy() if hasattr(v, "numpy") else np.asarray(v)
                f[k] = arr.reshape(meta.features[k]["shape"]).astype(meta.features[k]["dtype"])
        frames.append(f)
    return frames


def build_mixture(recipe_path: str | Path, root: str | Path | None = None, overwrite: bool = False) -> Path:
    """Materialize a recipe as a LeRobot dataset + labels sidecar. Returns the dataset root."""
    from robocurate.data.hub import episode_lengths, load_dataset, load_metadata
    from robocurate.paths import scratch_dir

    recipe = Recipe.load(recipe_path)
    metas = {s.dataset: load_metadata(s.dataset, revision=s.revision) for s in recipe.sources}
    plan = plan_mixture(recipe, {k: episode_lengths(m) for k, m in metas.items()})

    ref = metas[recipe.sources[0].dataset]
    for k, m in metas.items():
        if m.fps != ref.fps or {f: v["shape"] for f, v in m.features.items()} != {
            f: v["shape"] for f, v in ref.features.items()
        }:
            raise ConfigError(f"source {k} is incompatible with {recipe.sources[0].dataset} (fps/features)")
    features = {k: v for k, v in ref.features.items() if k not in DEFAULT_FEATURE_KEYS}
    user_keys = sorted(features)
    video_keys = [k for k, v in features.items() if v["dtype"] == "video"]

    root = Path(root) if root else scratch_dir("datasets") / recipe.repo_id
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"{root} exists (pass overwrite=True)")
        shutil.rmtree(root)
    root.parent.mkdir(parents=True, exist_ok=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = LeRobotDataset.create(
        repo_id=recipe.repo_id,
        fps=ref.fps,
        features=features,
        root=root,
        robot_type=ref.robot_type,
        use_videos=bool(video_keys),
    )
    needed: dict[str, list[int]] = {}
    for p in plan:
        needed.setdefault(p.source_dataset, []).append(p.source_episode)
    sources = {
        k: load_dataset(k, episodes=sorted(set(v)), revision=next(
            s.revision for s in recipe.sources if s.dataset == k
        ))
        for k, v in needed.items()
    }
    try:
        for p in plan:
            frames = _episode_frames(sources[p.source_dataset], metas[p.source_dataset], p.source_episode,
                                     user_keys, video_keys)
            if "pause" in p.defects:
                frames = D.inject_pause(frames, p.pause_t, p.pause_k)
            if "wrong_task" in p.defects:
                frames = D.inject_wrong_task(frames, p.wrong_task)
            if "duplicate" in p.defects:
                frames = D.duplicate_episode(frames)
            for f in frames:
                out.add_frame(dict(f))
            out.save_episode()
    finally:
        out.finalize()
    write_labels(labels_for(recipe, plan), sidecar_path(root))
    return root


# ---------------------------------------------------------------------------- policy-rollout source (item 6)
#
# Rolls out given policy checkpoints (e.g. full-data checkpoints at different training stages) on CURATION seeds,
# optionally with replay-mode Perturbations executed in the env (D-011: observations follow the executed
# actions), and records the episodes as a LeRobot dataset + labels sidecar (source "policy", checkpoint step,
# success, perturbations). Seeds are taken from the TOP of protocol.curation_seed_range downward, so they never
# collide with cleaner rollouts, which take curation seeds from the bottom up (curation/rollouts.next_seeds).
# NOT run on real checkpoints in this milestone (toybed only).

PERTURBATIONS: dict[str, type] = {"jitter": D.Jitter, "gripper_drop": D.GripperDrop, "hesitation": D.Hesitation}


@dataclass(frozen=True)
class PolicySource:
    name: str
    checkpoint: str
    checkpoint_step: int
    n_episodes: int
    perturbations: tuple[dict, ...] = ()  # e.g. ({"type": "jitter", "sigma": 0.3, "onset": 5, "duration": 10},)


@dataclass
class PolicyEpisode:
    source: PolicySource
    seed: int
    states: np.ndarray
    actions: np.ndarray
    success: bool

    @property
    def perturbation_names(self) -> tuple[str, ...]:
        return tuple(sorted({p["type"] for p in self.source.perturbations}))


def make_perturbations(specs: tuple[dict, ...]) -> list:
    out = []
    for sp in specs:
        sp = dict(sp)
        t = sp.pop("type")
        if t not in PERTURBATIONS:
            raise ConfigError(f"unknown perturbation {t!r}; allowed {sorted(PERTURBATIONS)}")
        out.append(PERTURBATIONS[t](**sp))
    return out


def policy_rollout_seeds(protocol: Any, n: int) -> list[int]:
    lo, hi = protocol.curation_seed_range
    if n > hi - lo + 1:
        raise ConfigError("not enough curation seeds")
    seeds = list(range(hi, hi - n, -1))
    e0, e1 = protocol.eval_seed_range()
    if any(e0 <= s < e1 for s in seeds):
        raise ConfigError("policy-rollout seeds overlap eval seeds")
    return seeds


def collect_policy_episodes(sources: list[PolicySource], load_policy, env_factory, protocol: Any,
                            max_steps: int) -> list[PolicyEpisode]:
    """load_policy(source) -> callable obs -> action; env_factory() -> gym-style env (info['is_success'])."""
    seeds = policy_rollout_seeds(protocol, sum(s.n_episodes for s in sources))
    out, k = [], 0
    for src in sources:
        base = load_policy(src)
        for _ in range(src.n_episodes):
            seed = seeds[k]
            k += 1
            pol = D.PerturbedPolicy(base, make_perturbations(src.perturbations), seed=seed)
            env = env_factory()
            obs, _ = env.reset(seed=seed)
            states, actions, success = [], [], False
            for _ in range(max_steps):
                a = np.asarray(pol(obs), dtype=np.float32)
                states.append(np.asarray(obs, dtype=np.float32))
                actions.append(a)
                obs, _, term, trunc, info = env.step(a)
                success = success or bool(info.get("is_success", False))
                if term or trunc:
                    break
            out.append(PolicyEpisode(src, seed, np.array(states), np.array(actions), success))
    return out


def write_policy_dataset(episodes: list[PolicyEpisode], repo_id: str, root: str | Path, fps: int, task: str,
                         shuffle_seed: int) -> Path:
    """LeRobot dataset (no video) + labels sidecar. Episode order shuffled by shuffle_seed (D-021)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    if root.exists():
        raise FileExistsError(f"{root} exists")
    sd, ad = episodes[0].states.shape[1], episodes[0].actions.shape[1]
    features = {"observation.state": {"dtype": "float32", "shape": (sd,), "names": None},
                "action": {"dtype": "float32", "shape": (ad,), "names": None}}
    ds = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=features, root=root, use_videos=False)
    order = np.random.default_rng(shuffle_seed).permutation(len(episodes))
    labels = []
    try:
        for out_i, j in enumerate(order):
            ep = episodes[int(j)]
            for s, a in zip(ep.states, ep.actions, strict=True):
                ds.add_frame({"task": task, "observation.state": s, "action": a})
            ds.save_episode()
            names = ep.perturbation_names
            labels.append(EpisodeLabel(out_i, "policy", ep.source.checkpoint, ep.seed, defects=names,
                                       quality_tier=max([DEFECT_TIER[d] for d in names] or ["clean"], key=TIERS.index),
                                       checkpoint_step=ep.source.checkpoint_step, success=ep.success,
                                       perturbations=names))
    finally:
        ds.finalize()
    write_labels(LabelsSidecar(dataset=repo_id, recipe_sha256="policy-source", seed=shuffle_seed, episodes=labels),
                 sidecar_path(root))
    return root
