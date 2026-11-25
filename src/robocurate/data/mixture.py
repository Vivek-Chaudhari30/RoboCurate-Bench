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
from collections.abc import Iterator
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
    deferred_sources: tuple[dict, ...] = ()  # generated in the GPU phase; not part of the CPU-buildable plan
    sha256: str = ""

    @classmethod
    def load(cls, path: str | Path) -> Recipe:
        raw = Path(path).read_bytes()
        d = load_yaml(path)
        allowed = {"name", "repo_id", "seed", "sources", "wrong_task_strings", "defects", "deferred_sources"}
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
            deferred_sources=tuple(_check_deferred(x, path) for x in d.get("deferred_sources", ()) or ()),
            sha256=hashlib.sha256(raw).hexdigest(),
        )._checked(path)

    def _checked(self, path) -> Recipe:
        for name, spec in (("pause", self.pause), ("duplicate", self.duplicate), ("wrong_task", self.wrong_task)):
            if spec and ("n_episodes" in spec) == ("rate" in spec):
                raise ConfigError(f"{path}: defects.{name} needs exactly one of n_episodes / rate")
            if "rate" in spec and not 0.0 <= float(spec["rate"]) <= 1.0:
                raise ConfigError(f"{path}: defects.{name}.rate must be in [0, 1]")
        return self

    @property
    def n_deferred_episodes(self) -> int:
        return sum(int(x["n_episodes"]) for x in self.deferred_sources)


DEFERRED_KINDS = ("policy", "scripted_replay")


def _check_deferred(x: dict, path) -> dict:
    """A source generated in the GPU phase (D-031): declared in the recipe so it is hashed and pre-registered, but
    never built on CPU. policy: rollouts of full-data checkpoints at training fractions; scripted_replay: the scripted
    policy executed in sim with Perturbations (D-011 replay mode)."""
    need = {"name", "source", "n_episodes", "generated_in"}
    if not need <= set(x):
        raise ConfigError(f"{path}: deferred source needs {sorted(need)}; got {sorted(x)}")
    if x["source"] not in DEFERRED_KINDS or x["generated_in"] != "gpu_phase":
        raise ConfigError(f"{path}: deferred source kind must be one of {DEFERRED_KINDS}, generated_in gpu_phase")
    n = int(x["n_episodes"])
    parts = x.get("checkpoints") if x["source"] == "policy" else x.get("perturbations")
    if not parts or sum(int(p["n_episodes"]) for p in parts) != n:
        raise ConfigError(f"{path}: deferred source {x['name']}: per-part n_episodes must sum to {n}")
    if x["source"] == "scripted_replay":
        for p in parts:
            if p["type"] not in PERTURBATIONS:
                raise ConfigError(f"{path}: unknown perturbation {p['type']!r}; allowed {sorted(PERTURBATIONS)}")
    else:
        for p in parts:
            if not 0 < float(p["train_fraction"]) <= 1:
                raise ConfigError(f"{path}: checkpoint train_fraction must be in (0, 1]")
    return dict(x)


def defect_count(spec: dict, n_base: int) -> int:
    """n_episodes, or round-half-up(rate x n_base) for rate-based defects (D-031)."""
    if not spec:
        return 0
    if "n_episodes" in spec:
        return int(spec["n_episodes"])
    return int(np.floor(float(spec["rate"]) * n_base + 0.5))


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

    n_pause = defect_count(recipe.pause, n)
    n_wrong = defect_count(recipe.wrong_task, n)
    n_dup = defect_count(recipe.duplicate, n)
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
    # camera key -> (T, H, W, C) uint8. Keys are LeRobot feature names (observation.images.* / observation.image).
    cameras: dict[str, np.ndarray] = field(default_factory=dict)

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


def policy_rollout_seeds(protocol: Any, n: int, offset: int = 0) -> list[int]:
    """n curation seeds from the TOP of the range downward, skipping the first `offset` (earlier deferred sources)."""
    lo, hi = protocol.curation_seed_range
    if offset < 0 or n + offset > hi - lo + 1:
        raise ConfigError("not enough curation seeds")
    seeds = list(range(hi - offset, hi - offset - n, -1))
    e0, e1 = protocol.eval_seed_range()
    if any(e0 <= s < e1 for s in seeds):
        raise ConfigError("policy-rollout seeds overlap eval seeds")
    return seeds


def _split_observation(obs: Any) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """State vector plus HWC uint8 camera frames, keyed as LeRobot dataset features.

    ndarray -> state only (the toybed). A gym obs with ``agent_pos`` / ``pixels`` maps the way
    ``preprocess_observation`` does (envs/utils.py:68): ``pixels`` dict -> ``observation.images.<cam>``,
    a single ``pixels`` array -> ``observation.image``, ``agent_pos`` -> ``observation.state``.
    An obs that already uses those LeRobot keys is stored as-is.
    """
    if isinstance(obs, dict):
        if "observation.state" in obs or any(
            k == "observation.image" or k.startswith("observation.images.") for k in obs
        ):
            if "observation.state" not in obs:
                raise TypeError("observation has camera features but no observation.state")
            state = np.array(np.asarray(obs["observation.state"], dtype=np.float32).reshape(-1), copy=True)
            cams = {
                k: np.array(_to_hwc_uint8(v), dtype=np.uint8, copy=True)
                for k, v in obs.items()
                if k == "observation.image" or k.startswith("observation.images.")
            }
            return state, cams
        if "agent_pos" in obs or "pixels" in obs:
            if "agent_pos" not in obs:
                raise TypeError("gym observation has pixels but no agent_pos")
            state = np.array(np.asarray(obs["agent_pos"], dtype=np.float32).reshape(-1), copy=True)
            cams: dict[str, np.ndarray] = {}
            pixels = obs.get("pixels")
            if isinstance(pixels, dict):
                for name, img in pixels.items():
                    cams[f"observation.images.{name}"] = np.array(_to_hwc_uint8(img), dtype=np.uint8, copy=True)
            elif pixels is not None:
                cams["observation.image"] = np.array(_to_hwc_uint8(pixels), dtype=np.uint8, copy=True)
            return state, cams
        raise TypeError(f"unrecognized observation keys {sorted(obs)}")
    return np.array(np.asarray(obs, dtype=np.float32).reshape(-1), copy=True), {}


def _policy_jobs(sources: list[PolicySource], protocol: Any, seed_offset: int) -> list[tuple[PolicySource, int]]:
    """(source, curation seed) in collection order. One entry per episode."""
    seeds = policy_rollout_seeds(protocol, sum(s.n_episodes for s in sources), seed_offset)
    jobs: list[tuple[PolicySource, int]] = []
    k = 0
    for src in sources:
        for _ in range(src.n_episodes):
            jobs.append((src, seeds[k]))
            k += 1
    return jobs


def _rollout_one(src: PolicySource, seed: int, base, env_factory, max_steps: int) -> PolicyEpisode:
    """One episode. A loaded policy with reset() (chunked ACT / Diffusion) is reset here."""
    if hasattr(base, "reset"):
        base.reset()
    pol = D.PerturbedPolicy(base, make_perturbations(src.perturbations), seed=seed)
    env = env_factory()
    obs, _ = env.reset(seed=seed)
    states, actions, cams, success = [], [], {}, False
    for _ in range(max_steps):
        a = np.asarray(pol(obs), dtype=np.float32)
        state, frame_cams = _split_observation(obs)
        states.append(state)
        actions.append(a)
        if cams and set(frame_cams) != set(cams):
            raise ValueError(f"camera keys changed mid-episode: {sorted(cams)} -> {sorted(frame_cams)}")
        for key, img in frame_cams.items():
            cams.setdefault(key, []).append(img)
        obs, _, term, trunc, info = env.step(a)
        success = success or bool(info.get("is_success", False))
        if term or trunc:
            break
    stacked = {key: np.stack(frames) for key, frames in cams.items()}
    return PolicyEpisode(src, seed, np.array(states), np.array(actions), success, stacked)


def _release_cameras(ep: PolicyEpisode) -> None:
    """Drop this episode's camera arrays. The caller must not keep another reference to them."""
    ep.cameras.clear()


def iter_policy_episodes(sources: list[PolicySource], load_policy, env_factory, protocol: Any,
                         max_steps: int, seed_offset: int = 0) -> Iterator[PolicyEpisode]:
    """Yield one rollout at a time. Drop ``ep.cameras`` before pulling the next episode."""
    loaded: dict[int, Any] = {}
    for src, seed in _policy_jobs(sources, protocol, seed_offset):
        if id(src) not in loaded:
            loaded[id(src)] = load_policy(src)
        yield _rollout_one(src, seed, loaded[id(src)], env_factory, max_steps)


def collect_policy_episodes(sources: list[PolicySource], load_policy, env_factory, protocol: Any,
                            max_steps: int, seed_offset: int = 0) -> list[PolicyEpisode]:
    """load_policy(source) -> callable obs -> action; env_factory() -> gym-style env (info['is_success']).

    Keeps every episode, including camera frames. Fine for a few toy episodes. A full camera source
    (100 ALOHA episodes) must use write_policy_episodes, which writes and releases each episode first.
    """
    return list(iter_policy_episodes(sources, load_policy, env_factory, protocol, max_steps, seed_offset))


def _video_spec(frame: np.ndarray) -> dict[str, Any]:
    """One camera feature. Shape is HWC, matching hw_to_dataset_features (utils/feature_utils.py:91-99)
    and the ALOHA task dataset. names[2] == "channel" is what dataset_to_policy_features treats as HWC
    (utils/feature_utils.py:166)."""
    frame = np.asarray(frame)
    if frame.ndim != 3:
        raise ValueError(f"camera frame must be HWC, got shape {frame.shape}")
    h, w, c = (int(x) for x in frame.shape)
    spec: dict[str, Any] = {
        "dtype": "video",
        "shape": (h, w, c),
        "names": ["height", "width", "channel"],
    }
    if c == 1:
        spec["info"] = {"is_depth_map": True}
    return spec


def _camera_feature_spec(episodes: list[PolicyEpisode]) -> dict[str, dict]:
    """Video features for the cameras on these episodes. Empty when no episode recorded a camera."""
    keys = list(episodes[0].cameras)
    for ep in episodes:
        if set(ep.cameras) != set(keys):
            raise ValueError(
                f"camera keys differ across policy episodes: {sorted(keys)} vs {sorted(ep.cameras)}"
            )
        for key in keys:
            if ep.cameras[key].shape[1:] != episodes[0].cameras[key].shape[1:]:
                raise ValueError(f"{key} shape {ep.cameras[key].shape} != {episodes[0].cameras[key].shape}")
            if len(ep.cameras[key]) != len(ep.actions):
                raise ValueError(f"{key} has {len(ep.cameras[key])} frames for {len(ep.actions)} actions")
    return {key: _video_spec(episodes[0].cameras[key][0]) for key in keys}


def _episode_label(ep: PolicyEpisode, out_i: int, source_label: str) -> EpisodeLabel:
    names = ep.perturbation_names
    return EpisodeLabel(
        out_i, source_label, ep.source.checkpoint, ep.seed, defects=names,
        quality_tier=max([DEFECT_TIER[d] for d in names] or ["clean"], key=TIERS.index),
        checkpoint_step=ep.source.checkpoint_step, success=ep.success, perturbations=names,
    )


def _append_episode(ds, ep: PolicyEpisode, task: str, cam_keys: list[str]) -> None:
    for key in cam_keys:
        if len(ep.cameras[key]) != len(ep.actions):
            raise ValueError(f"{key} has {len(ep.cameras[key])} frames for {len(ep.actions)} actions")
    for t, (s, a) in enumerate(zip(ep.states, ep.actions, strict=True)):
        frame: dict[str, Any] = {"task": task, "observation.state": s, "action": a}
        for key in cam_keys:
            frame[key] = ep.cameras[key][t]
        ds.add_frame(frame)
    # Images go to a temp file or the streaming encoder inside add_frame (dataset_writer.py:254-265),
    # then save_episode encodes that episode and resets the buffer (lerobot_dataset.py:434).
    ds.save_episode(parallel_encoding=False)


def _open_policy_dataset(ep: PolicyEpisode, repo_id: str, root: Path, fps: int):
    """Create the dataset from the first episode's state, action, and camera shapes."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    cam_keys = list(ep.cameras)
    features: dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": (ep.states.shape[1],), "names": None},
        "action": {"dtype": "float32", "shape": (ep.actions.shape[1],), "names": None},
    }
    for key in cam_keys:
        features[key] = _video_spec(ep.cameras[key][0])
    # use_videos=False is only valid when there is no camera (dataset_metadata.py:850-853).
    ds = LeRobotDataset.create(
        repo_id=repo_id, fps=fps, features=features, root=root, use_videos=bool(cam_keys),
    )
    return ds, cam_keys


def _check_camera_layout(ep: PolicyEpisode, cam_keys: list[str], features: dict) -> None:
    if set(ep.cameras) != set(cam_keys):
        raise ValueError(f"camera keys differ across policy episodes: {sorted(cam_keys)} vs {sorted(ep.cameras)}")
    for key in cam_keys:
        got = tuple(int(x) for x in ep.cameras[key].shape[1:])
        want = tuple(int(x) for x in features[key]["shape"])
        if got != want:
            raise ValueError(f"{key} shape {ep.cameras[key].shape} != {features[key]['shape']}")


def write_policy_dataset(episodes: list[PolicyEpisode], repo_id: str, root: str | Path, fps: int, task: str,
                         shuffle_seed: int, source_label: str = "policy") -> Path:
    """LeRobot dataset + labels sidecar. Episode order shuffled by shuffle_seed (D-021).

    State and action are always written. Camera frames on the episodes are written as video features
    (dtype ``video``, HWC) and ``use_videos=True`` — dataset_metadata.py:850 refuses video keys otherwise.
    ``LeRobotDataset.create`` (datasets/lerobot_dataset.py:684), ``add_frame`` (:418; images may be HWC,
    datasets/feature_utils.py:330), ``save_episode`` (:434), ``finalize`` (:474).

    This holds every episode's frames. write_policy_episodes is the path that releases each episode first.
    """
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"{root} exists")
    if not episodes:
        raise ValueError("no episodes to write")
    cameras = _camera_feature_spec(episodes)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features: dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": (episodes[0].states.shape[1],), "names": None},
        "action": {"dtype": "float32", "shape": (episodes[0].actions.shape[1],), "names": None},
    }
    features.update(cameras)
    ds = LeRobotDataset.create(
        repo_id=repo_id, fps=fps, features=features, root=root, use_videos=bool(cameras),
    )
    order = np.random.default_rng(shuffle_seed).permutation(len(episodes))
    cam_keys = list(cameras)
    labels = []
    try:
        for out_i, j in enumerate(order):
            ep = episodes[int(j)]
            _append_episode(ds, ep, task, cam_keys)
            labels.append(_episode_label(ep, out_i, source_label))
    finally:
        ds.finalize()
    write_labels(LabelsSidecar(dataset=repo_id, recipe_sha256="policy-source", seed=shuffle_seed, episodes=labels),
                 sidecar_path(root))
    return root


def write_policy_episodes(sources: list[PolicySource], load_policy, env_factory, protocol: Any,
                          max_steps: int, repo_id: str, root: str | Path, fps: int, task: str,
                          shuffle_seed: int, seed_offset: int = 0, source_label: str = "policy",
                          ) -> tuple[Path, list[PolicyEpisode]]:
    """Roll out one episode, write it, and release its camera frames before the next rollout.

    Peak host memory stays on the order of one episode: frames are not kept on a list of
    PolicyEpisode objects. Dataset order is still the shuffle_seed permutation of collection order
    (D-021) — episodes are rolled out in that order and appended as they finish.
    Returned episodes have empty ``cameras``; the frames are in the dataset.
    """
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"{root} exists")
    jobs = _policy_jobs(sources, protocol, seed_offset)
    if not jobs:
        raise ValueError("no episodes to write")
    order = np.random.default_rng(shuffle_seed).permutation(len(jobs))
    loaded: dict[int, Any] = {}
    kept: list[PolicyEpisode] = []
    ds = None
    cam_keys: list[str] = []
    labels: list[EpisodeLabel] = []
    try:
        for out_i, j in enumerate(order):
            src, seed = jobs[int(j)]
            if id(src) not in loaded:
                loaded[id(src)] = load_policy(src)
            ep = _rollout_one(src, seed, loaded[id(src)], env_factory, max_steps)
            if ds is None:
                ds, cam_keys = _open_policy_dataset(ep, repo_id, root, fps)
            else:
                _check_camera_layout(ep, cam_keys, ds.meta.features)
            _append_episode(ds, ep, task, cam_keys)
            labels.append(_episode_label(ep, out_i, source_label))
            kept.append(PolicyEpisode(src, seed, ep.states, ep.actions, ep.success, {}))
            _release_cameras(ep)
    finally:
        if ds is not None:
            ds.finalize()
    write_labels(LabelsSidecar(dataset=repo_id, recipe_sha256="policy-source", seed=shuffle_seed, episodes=labels),
                 sidecar_path(root))
    return root, kept


# ---------------------------------------------------------------------------- deferred sources (D-031)


def deferred_seed_offset(recipe: Recipe, name: str) -> int:
    """Curation-seed offset of a deferred source: the episodes of the deferred sources listed before it, so two
    GPU-phase sources never share a seed (both take seeds from the top of curation_seed_range)."""
    off = 0
    for x in recipe.deferred_sources:
        if x["name"] == name:
            return off
        off += int(x["n_episodes"])
    raise ConfigError(f"no deferred source named {name!r}")


def replay_policy_sources(spec: dict, protocol: Any, seed_offset: int, checkpoint: str = "scripted",
                          checkpoint_step: int = 0) -> list[PolicySource]:
    """Expand a scripted_replay deferred source into one PolicySource per episode (D-011 replay mode: the scripted
    policy executed in sim with a Perturbation). A `sigma_range: [lo, hi]` jitter gets a per-episode sigma drawn
    uniformly from np.random.default_rng(<that episode's curation seed>), so it is fixed by the seed alone."""
    if spec.get("source") != "scripted_replay":
        raise ConfigError("replay_policy_sources needs a scripted_replay deferred source")
    seeds = policy_rollout_seeds(protocol, int(spec["n_episodes"]), seed_offset)
    out, k = [], 0
    for part in spec["perturbations"]:
        part = dict(part)
        n = int(part.pop("n_episodes"))
        for _ in range(n):
            p = dict(part)
            if "sigma_range" in p:
                lo, hi = (float(x) for x in p.pop("sigma_range"))
                p["sigma"] = float(np.random.default_rng(seeds[k]).uniform(lo, hi))
            out.append(PolicySource(f"{spec['name']}_{p['type']}_{k}", checkpoint, checkpoint_step, 1, (p,)))
            k += 1
    return out

