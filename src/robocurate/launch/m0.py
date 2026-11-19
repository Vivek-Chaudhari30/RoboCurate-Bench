"""M0 calibration pipeline (stages a-f), driven by scripts/m0_pipeline.sh. Built and dry-run on CPU with the fake
backend; the real run happens on hokkaido inside tmux `rcb-m0` (docs/runbook_gpu.md).

Stages
  a  wait for a free GPU: poll every 10 min; never start while another user is on the GPU (runs/gpu.py busy rule,
     CLAUDE.md); give up after 72 h.
  b  smoke test: 500 training steps + 10 eval episodes on calibration seeds (D-017).
  b2 I/O check (D-027, network scratch): if GPU utilization < 50% and data wait > 30% of step time, rerun the smoke
     test with DataLoader num_workers doubled, at most twice. Nothing else may change.
  c  GPU determinism and eval batch-invariance tests (tests/test_gpu.py, marker gpu). Failure: diagnose and exit.
  d  auto-configure: eval batch size = fastest of {5, 10, 25, 50} with peak memory < 12 GB; S = LeRobot's default
     step count (TrainPipelineConfig.steps, configs/train.py:111) capped at 2.5 h per run (train + evals); exit if the
     cap gives < 25% of the default. Writes steps, intermediate_save_freq (S/10), batch sizes and num_workers into
     the calibration config.
  e  calibration grid (resumable through the executor): D-018 cleanup, videos deleted after parsing, non-final
     checkpoints deleted after eval (except the rollout-source checkpoints, see ROLLOUT_KEEP), 200 GB disk guard,
     caps 36 GPU-h and 72 h of run time, packing (2 runs at once) only after a solo full-data run used < 50% GPU and
     < 15 GB, a run that fails twice is skipped, 3 skipped runs stop the stage; after each run one PROGRESS.md line
     and a commit of its results JSON.
  f  analysis: s_seed and s_sub with CIs, step curve, cost, disk, the D-026 power table over n in {3, 5},
     K in {8, 12}, R in {1, 2, 3}, projected launch GPU-h, and the morning summary. Proposes a design; never writes
     n, K or R anywhere (protocol decision, docs/preregistration.md section 9).
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

D027_SCRATCH = "/share/datasets/robocurate"
DISK_CAP_GB = 200.0
GPU_H_CAP = 36.0
RUN_H_CAP = 72.0
WAIT_H = 72.0
POLL_S = 600
SMOKE_STEPS, SMOKE_EPISODES = 500, 10
IO_UTIL_PCT, IO_DATA_FRAC, IO_RETRIES = 50.0, 0.30, 2
EVAL_BATCHES, EVAL_MEM_GB = (5, 10, 25, 50), 12.0
PER_RUN_H_CAP, MIN_S_FRAC = 2.5, 0.25
PACK_UTIL_PCT, PACK_MEM_GB = 50.0, 15.0
MAX_ATTEMPTS, MAX_SKIPPED = 2, 3
ROLLOUT_FRACTIONS = (0.10, 0.30, 0.60)  # D-031 rollout-source checkpoints (100% = the final checkpoint)
ROLLOUT_SEED = 1000  # first train seed's full-data run (D-022 reference-policy convention)
DESIGN_N, DESIGN_K, DESIGN_R = (3, 5), (8, 12), (1, 2, 3)
MDE_TARGET = 0.12  # docs/preregistration.md section 9


class KitError(RuntimeError):
    pass


# ---------------------------------------------------------------------------- startup


def check_startup(expected_scratch: str = D027_SCRATCH, allow_other_scratch: bool = False,
                  env: dict[str, str] | None = None) -> dict[str, str]:
    """env.sh must be sourced BEFORE Python starts: lerobot reads HF_LEROBOT_HOME once at import
    (lerobot/utils/constants.py:69), so a later export would silently send datasets to $HOME."""
    env = dict(os.environ) if env is None else env
    need = ("ROBOCURATE_SCRATCH", "HF_HOME", "HF_LEROBOT_HOME", "MUJOCO_GL")
    missing = [k for k in need if not env.get(k)]
    if missing:
        raise KitError(f"source scripts/env.sh before starting (missing {missing})")
    scratch = env["ROBOCURATE_SCRATCH"]
    if not allow_other_scratch and os.path.realpath(scratch) != os.path.realpath(expected_scratch):
        raise KitError(f"ROBOCURATE_SCRATCH={scratch}; D-027 requires {expected_scratch}")
    from lerobot.utils import constants

    if Path(constants.HF_LEROBOT_HOME).resolve() != Path(env["HF_LEROBOT_HOME"]).resolve():
        raise KitError(f"lerobot imported HF_LEROBOT_HOME={constants.HF_LEROBOT_HOME} != env "
                       f"{env['HF_LEROBOT_HOME']}: env.sh was sourced after Python started")
    return {k: env[k] for k in need}


def dir_size_gb(path: Path) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            with contextlib.suppress(OSError):
                total += os.lstat(os.path.join(root, f)).st_size
    return total / 1e9


def check_disk(scratch: Path, cap_gb: float = DISK_CAP_GB) -> dict[str, Any]:
    """D-027: df and quota before first use; usage under the scratch dir must stay below the cap."""
    du = shutil.disk_usage(scratch)
    q = shutil.which("quota")
    quota = subprocess.run([q, "-s"], capture_output=True, text=True).stdout.strip() if q else "quota not installed"
    used = dir_size_gb(scratch)
    info = {"scratch": str(scratch), "free_gb": du.free / 1e9, "used_by_us_gb": used, "cap_gb": cap_gb,
            "quota": quota}
    if used >= cap_gb:
        raise KitError(f"scratch usage {used:.1f} GB >= cap {cap_gb} GB (D-027)")
    if du.free / 1e9 < 20:
        raise KitError(f"only {du.free / 1e9:.1f} GB free on {scratch}")
    return info


# ---------------------------------------------------------------------------- probes (GPU side effects)


@dataclass
class SmokeResult:
    step_s: float  # mean (update + data) seconds per step
    data_frac: float  # data wait / step time
    gpu_util: float  # mean utilization during training, %
    peak_mem_gb: float
    eval_s: float  # wall-clock of the 10-episode smoke eval
    num_workers: int
    checkpoint: str = ""


class Probe:
    """Side-effecting measurements. LocalProbe runs LeRobot on the GPU; FakeProbe returns scripted numbers."""

    def gpu_busy(self) -> str | None: ...
    def smoke(self, num_workers: int) -> SmokeResult: ...
    def gpu_tests(self) -> tuple[bool, str]: ...
    def eval_timing(self, batch: int, n_episodes: int) -> tuple[float, float]: ...  # (seconds, peak GB)
    def util_sampler(self) -> Any: ...


class FakeProbe(Probe):
    """Deterministic stand-in for the dry run: the GPU frees after `busy_polls` polls; the first smoke run is
    I/O-bound (util 35%, data 45%) so stage b2 is exercised; num_workers 8 fixes it."""

    def __init__(self, busy_polls: int = 2, default_workers: int = 4):
        self.polls, self.busy_polls, self.default_workers = 0, busy_polls, default_workers

    def gpu_busy(self) -> str | None:
        self.polls += 1
        return "fake: pid 4242 (otheruser) uses 27556 MB > 2048 MB" if self.polls <= self.busy_polls else None

    def smoke(self, num_workers: int) -> SmokeResult:
        bound = num_workers <= self.default_workers
        return SmokeResult(step_s=0.20 if bound else 0.12, data_frac=0.45 if bound else 0.08,
                           gpu_util=35.0 if bound else 82.0, peak_mem_gb=9.5, eval_s=40.0, num_workers=num_workers,
                           checkpoint="fake://smoke")

    def gpu_tests(self) -> tuple[bool, str]:
        return True, "fake: GPU tests not run (no GPU in the dry run)"

    def eval_timing(self, batch: int, n_episodes: int) -> tuple[float, float]:
        sec = {5: 300.0, 10: 170.0, 25: 95.0, 50: 80.0}[batch] * n_episodes / 50
        mem = {5: 3.0, 10: 5.0, 25: 9.0, 50: 14.0}[batch]  # 50 envs exceed the 12 GB limit
        return sec, mem

    def util_sampler(self):
        return _ConstSampler(38.0, 6.0)


class _ConstSampler:
    def __init__(self, util: float, mem_gb: float):
        self.mean_util, self.peak_mem_gb = util, mem_gb

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


class UtilSampler:
    """Polls nvidia-smi utilization and total used memory every `interval_s` while a run is active."""

    def __init__(self, interval_s: float = 5.0):
        self.interval_s, self.samples, self.mem = interval_s, [], []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
                u, m = (float(x) for x in out.strip().splitlines()[0].split(","))
                self.samples.append(u)
                self.mem.append(m / 1024)
            except Exception:  # noqa: BLE001 - sampling is best effort
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=10)

    @property
    def mean_util(self) -> float:
        return float(np.mean(self.samples)) if self.samples else float("nan")

    @property
    def peak_mem_gb(self) -> float:
        return float(max(self.mem)) if self.mem else float("nan")


TRAIN_LOG_RE = re.compile(r"updt_s:(?P<u>[0-9.]+).*?data_s:(?P<d>[0-9.]+)")


def parse_train_log(text: str) -> tuple[float, float]:
    """Mean updt_s and data_s over lerobot-train's periodic log lines (AverageMeter names 'updt_s' and 'data_s',
    scripts/lerobot_train.py:568-569; rendered '{name}:{avg:.3f}' by utils/logging_utils.py:63-65)."""
    us, ds = [], []
    for m in TRAIN_LOG_RE.finditer(text):
        us.append(float(m["u"]))
        ds.append(float(m["d"]))
    if not us:
        raise KitError("no 'updt_s:... data_s:...' lines in the training log")
    return float(np.mean(us)), float(np.mean(ds))


class LocalProbe(Probe):  # pragma: no cover - needs a GPU; exercised only on hokkaido
    def __init__(self, cal, workdir: Path):
        self.cal, self.workdir = cal, Path(workdir)
        self._ckpt: Path | None = None
        self._subset: Path | None = None

    def gpu_busy(self) -> str | None:
        import getpass

        from robocurate.runs import gpu

        procs, util = gpu.query()
        return gpu.decide_busy(procs, util, getpass.getuser())

    def _dataset(self) -> Path:
        if self._subset is None:
            from robocurate.data.hub import load_metadata
            from robocurate.data.subset import materialize_subset

            meta = load_metadata(self.cal.dataset)
            eps = list(range(meta.total_episodes))
            self._subset = materialize_subset(self.cal.dataset, None, eps, self.workdir / "smoke_subset")
        return self._subset

    def smoke(self, num_workers: int) -> SmokeResult:
        from robocurate.config import load_protocol, load_task, resolve_config_path
        from robocurate.eval.lerobot_eval import build_eval_cmd
        from robocurate.runs.spec import Condition, RunSpec
        from robocurate.train.lerobot_train import SubsetDataset, build_train_cmd, final_checkpoint

        task = load_task(resolve_config_path(self.cal.task), mode="calibration")
        proto = load_protocol(resolve_config_path(self.cal.protocol), mode="calibration")
        out = self.workdir / f"smoke_w{num_workers}"
        spec = RunSpec(task.name, Condition.full(), 1.0, None, int(self.cal.train_seeds[0]))
        cmd = build_train_cmd(task, spec, SubsetDataset(f"{self.cal.dataset}_train", self._dataset()), out / "train",
                              steps=SMOKE_STEPS, num_workers=num_workers, batch_size=default_batch_size())
        cmd[0] = console_script(cmd[0])
        with UtilSampler() as us:
            t0 = time.time()
            log = _run(cmd, out / "train.log")
        updt, data = parse_train_log(log)
        ckpt = final_checkpoint(out / "train", SMOKE_STEPS)
        ecmd = build_eval_cmd(ckpt, SMOKE_EPISODES, int(proto.calibration_seed_start), SMOKE_EPISODES,
                              env_type=task.env_type, env_task=task.env_task, output_dir=out / "eval")
        ecmd[0] = console_script(ecmd[0])
        t1 = time.time()
        _run(ecmd, out / "eval.log")
        self._ckpt = ckpt
        del t0
        return SmokeResult(step_s=updt + data, data_frac=data / max(updt + data, 1e-9), gpu_util=us.mean_util,
                           peak_mem_gb=us.peak_mem_gb, eval_s=time.time() - t1, num_workers=num_workers,
                           checkpoint=str(ckpt))

    def gpu_tests(self) -> tuple[bool, str]:
        env = dict(os.environ, RCB_GPU_TEST_CKPT=str(self._ckpt), RCB_GPU_TEST_DATASET=str(self._dataset()),
                   RCB_GPU_TEST_CAL=str(self.cal.path))
        p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-m", "gpu", "tests/test_gpu.py"],
                           capture_output=True, text=True, env=env)
        return p.returncode == 0, (p.stdout + p.stderr)[-4000:]

    def eval_timing(self, batch: int, n_episodes: int) -> tuple[float, float]:
        from robocurate.config import load_protocol, load_task, resolve_config_path
        from robocurate.eval.lerobot_eval import build_eval_cmd

        task = load_task(resolve_config_path(self.cal.task), mode="calibration")
        proto = load_protocol(resolve_config_path(self.cal.protocol), mode="calibration")
        out = self.workdir / f"evaltiming_b{batch}"
        cmd = build_eval_cmd(self._ckpt, n_episodes, int(proto.calibration_seed_start), batch,
                             env_type=task.env_type, env_task=task.env_task, output_dir=out)
        cmd[0] = console_script(cmd[0])
        with UtilSampler() as us:
            t0 = time.time()
            _run(cmd, out.with_suffix(".log"))
        return time.time() - t0, us.peak_mem_gb

    def util_sampler(self):
        return UtilSampler()


def console_script(name: str) -> str:
    """The env's console script next to sys.executable (PATH may not include the env's bin)."""
    p = Path(sys.executable).parent / name
    return str(p) if p.exists() else (shutil.which(name) or name)


def _run(cmd: list[str], log: Path) -> str:  # pragma: no cover - GPU path
    log.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    log.write_text(p.stdout + p.stderr)
    if p.returncode != 0:
        raise KitError(f"{cmd[0]} exited {p.returncode}; see {log}")
    return p.stdout + p.stderr


def default_batch_size() -> int:
    """D-017: training batch = LeRobot's default (TrainPipelineConfig.batch_size, configs/train.py:102)."""
    from lerobot.configs.train import TrainPipelineConfig

    return int(next(f.default for f in dataclasses.fields(TrainPipelineConfig) if f.name == "batch_size"))


def default_steps() -> int:
    """LeRobot's default step count (TrainPipelineConfig.steps, configs/train.py:111)."""
    from lerobot.configs.train import TrainPipelineConfig

    return int(next(f.default for f in dataclasses.fields(TrainPipelineConfig) if f.name == "steps"))


def default_num_workers() -> int:
    from lerobot.configs.train import TrainPipelineConfig

    return int(next(f.default for f in dataclasses.fields(TrainPipelineConfig) if f.name == "num_workers"))


# ---------------------------------------------------------------------------- stages a - d (pure decisions)


def stage_a_wait(probe: Probe, log: Callable[[str], None], sleep: Callable[[float], None] = time.sleep,
                 poll_s: float = POLL_S, max_h: float = WAIT_H) -> float:
    """Returns seconds waited. Raises KitError after max_h."""
    waited = 0.0
    while True:
        why = probe.gpu_busy()
        if why is None:
            log(f"[a] GPU free after {waited / 3600:.2f} h")
            return waited
        if waited + poll_s > max_h * 3600:
            raise KitError(f"GPU still busy after {max_h} h ({why}); giving up")
        log(f"[a] GPU busy ({why}); next poll in {poll_s / 60:.0f} min")
        sleep(poll_s)
        waited += poll_s


def io_bound(r: SmokeResult) -> bool:
    return r.gpu_util < IO_UTIL_PCT and r.data_frac > IO_DATA_FRAC


def stage_b(probe: Probe, log: Callable[[str], None], workers0: int) -> tuple[SmokeResult, list[SmokeResult]]:
    """Smoke test + I/O check (D-027): only num_workers may change, doubled at most IO_RETRIES times."""
    history = [probe.smoke(workers0)]
    log(f"[b] smoke: {asdict_round(history[-1])}")
    while io_bound(history[-1]) and len(history) <= IO_RETRIES:
        w = history[-1].num_workers * 2
        log(f"[b2] I/O-bound (util {history[-1].gpu_util:.0f}% < {IO_UTIL_PCT:.0f}%, data "
            f"{history[-1].data_frac:.0%} > {IO_DATA_FRAC:.0%}); retry with num_workers={w}")
        history.append(probe.smoke(w))
        log(f"[b2] smoke: {asdict_round(history[-1])}")
    if io_bound(history[-1]):
        log("[b2] still I/O-bound after the allowed retries; continuing (recorded) - see D-027")
    return history[-1], history


def choose_eval_batch(timings: dict[int, tuple[float, float]], mem_cap_gb: float = EVAL_MEM_GB) -> int:
    ok = {b: t for b, (t, m) in timings.items() if m < mem_cap_gb}
    if not ok:
        raise KitError(f"no eval batch size fits under {mem_cap_gb} GB: {timings}")
    return min(ok, key=lambda b: (ok[b], b))


def choose_steps(default: int, step_s: float, eval_s_per_run: float, cap_h: float = PER_RUN_H_CAP,
                 min_frac: float = MIN_S_FRAC) -> dict[str, Any]:
    """S = min(LeRobot default, largest S with S * step_s + eval time <= cap), rounded down to a multiple of 10
    (intermediate checkpoints every S/10). Exit if S < min_frac * default."""
    budget_s = cap_h * 3600 - eval_s_per_run
    s_cap = int(budget_s // step_s) if budget_s > 0 else 0
    steps = min(default, s_cap) // 10 * 10
    out = {"default": default, "cap": s_cap, "S": steps, "capped": steps < default, "frac": steps / default,
           "per_run_h": (steps * step_s + eval_s_per_run) / 3600}
    if steps < min_frac * default:
        raise KitError(f"the {cap_h} h cap allows S={steps} < {min_frac:.0%} of LeRobot's default {default}: {out}")
    return out


def write_calibration_values(path: Path, values: dict[str, Any]) -> str:
    """Rewrite `key: <value>` lines of a calibration YAML in place, keeping comments; refuses unknown keys."""
    text = Path(path).read_text()
    for k, v in values.items():
        pat = re.compile(rf"^({re.escape(k)}:\s*)([^#\n]*?)(\s*(#.*)?)$", re.M)
        if pat.search(text):
            text = pat.sub(lambda m, v=v: f"{m.group(1)}{v}{m.group(3)}", text, count=1)
        else:
            text = text.rstrip("\n") + f"\n{k}: {v}\n"
    Path(path).write_text(text)
    from robocurate.config import load_calibration

    load_calibration(path, mode="calibration")  # must still validate
    return text


def asdict_round(r) -> dict:
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in dataclasses.asdict(r).items()}


# ---------------------------------------------------------------------------- stage e


@dataclass
class GridState:
    gpu_h: float = 0.0
    run_h: float = 0.0
    failures: dict[str, int] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    done: list[str] = field(default_factory=list)
    packing: int = 1
    stop_reason: str | None = None


def cleanup_run_dir(run_dir: Path, final_step: int, keep_steps: tuple[int, ...] = ()) -> list[str]:
    """D-018 + D-027: delete the run's materialized subset, eval videos (after parsing) and every non-final
    checkpoint not in keep_steps. Returns the removed paths."""
    removed = []
    if not run_dir.exists():
        return removed
    for p in [run_dir / "subset", *run_dir.glob("eval*/videos")]:
        if p.exists():
            shutil.rmtree(p)
            removed.append(str(p))
    ck = run_dir / "train" / "checkpoints"
    if ck.exists():
        for d in ck.iterdir():
            if d.is_dir() and d.name.isdigit() and int(d.name) not in (final_step, *keep_steps):
                shutil.rmtree(d)
                removed.append(str(d))
    return removed


def rollout_keep_steps(spec, steps: int) -> tuple[int, ...]:
    """Checkpoints kept for the D-031 policy-rollout source: full-data, first train seed, 10/30/60% of S
    (D-027 otherwise keeps only final checkpoints; this exception is listed in the runbook)."""
    if spec.condition.kind == "full" and spec.train_seed == ROLLOUT_SEED:
        return tuple(int(round(f * steps)) for f in ROLLOUT_FRACTIONS)
    return ()


class Committer:
    def __init__(self, repo: Path, enabled: bool, log: Callable[[str], None]):
        self.repo, self.enabled, self.log = repo, enabled, log

    def commit(self, paths: list[Path], message: str) -> None:
        cmds = [["git", "-C", str(self.repo), "add", *map(str, paths)],
                ["git", "-C", str(self.repo), "commit", "-q", "-m", message]]
        if not self.enabled:
            self.log("[e] (dry run, not executed) " + " ; ".join(" ".join(c) for c in cmds))
            return
        for c in cmds:
            subprocess.run(c, check=True)


def stage_e(plan, backend, results_dir: Path, scratch: Path, probe: Probe, progress: Path, committer: Committer,
            log: Callable[[str], None], *, sleep: Callable[[float], None] = time.sleep,
            run_dir_of: Callable[[str], Path] | None = None, disk_cap_gb: float = DISK_CAP_GB,
            gpu_h_cap: float = GPU_H_CAP, run_h_cap: float = RUN_H_CAP, state: GridState | None = None,
            clock: Callable[[], float] = time.time) -> GridState:
    from concurrent.futures import ThreadPoolExecutor

    from robocurate import results
    from robocurate.runs.executor import execute, result_path
    from robocurate.runs.spec import run_id

    st = state or GridState()
    hashes = backend.config_hashes()
    steps = plan.steps
    todo = [s for s in plan.specs() if not results.is_valid_file(result_path(results_dir, run_id(s, hashes)))]
    full_first = sorted(todo, key=lambda s: (s.condition.kind != "full", s.train_seed))
    log(f"[e] {len(full_first)} runs to do ({len(plan.specs()) - len(full_first)} already done)")
    run_dir_of = run_dir_of or (lambda rid: scratch / "runs" / rid)

    def one(spec) -> tuple[Any, str, dict | None, str | None, float, float]:
        rid = run_id(spec, hashes)
        t0 = clock()
        with probe.util_sampler() as us:
            try:
                execute([spec], backend, results_dir, log=log)
                err = None
            except Exception as e:  # noqa: BLE001 - a failed run is counted, never fatal by itself
                err = f"{type(e).__name__}: {e}"
        rec_path = result_path(results_dir, rid)
        rec = json.loads(rec_path.read_text()) if results.is_valid_file(rec_path) else None
        del t0
        return spec, rid, rec, err, us.mean_util, us.peak_mem_gb

    queue = list(full_first)
    while queue:
        if st.run_h >= run_h_cap:
            st.stop_reason = f"run-time cap {run_h_cap} h reached"
            break
        if st.gpu_h >= gpu_h_cap:
            st.stop_reason = f"GPU-h cap {gpu_h_cap} reached"
            break
        used = dir_size_gb(scratch)
        if used >= disk_cap_gb:
            st.stop_reason = f"disk guard: {used:.1f} GB >= {disk_cap_gb} GB"
            break
        stage_a_wait(probe, log, sleep=sleep)
        batch, queue = queue[: st.packing], queue[st.packing:]
        t0 = clock()
        if len(batch) == 1:
            outs = [one(batch[0])]
        else:
            with ThreadPoolExecutor(max_workers=len(batch)) as ex:
                outs = list(ex.map(one, batch))
        wall_h = (clock() - t0) / 3600
        st.gpu_h += wall_h  # one GPU occupied for the batch's wall-clock, packed or not
        st.run_h += wall_h
        for spec, rid, rec, err, util, mem in outs:
            label = f"{spec.condition.label} b={spec.budget} si={spec.subset_index} seed={spec.train_seed}"
            if err or rec is None:
                st.failures[rid] = st.failures.get(rid, 0) + 1
                log(f"[e] FAILED {rid} {label} (attempt {st.failures[rid]}): {err}")
                if st.failures[rid] < MAX_ATTEMPTS:
                    queue.insert(0, spec)
                else:
                    st.skipped.append(rid)
                    log(f"[e] skipping {rid} after {MAX_ATTEMPTS} failures")
                    if len(st.skipped) >= MAX_SKIPPED:
                        st.stop_reason = f"{MAX_SKIPPED} runs failed twice"
                continue
            removed = cleanup_run_dir(run_dir_of(rid), int(steps) if not _tbd(steps) else -1,
                                      rollout_keep_steps(spec, int(steps)) if not _tbd(steps) else ())
            st.done.append(rid)
            line = (f"| {time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime())} | M0 {plan.grid.name} | {rid} {label} | "
                    f"success {rec['success_rate']:.3f} | wall {rec['wall_clock_s'] / 3600:.2f} h | GPU-h total "
                    f"{st.gpu_h:.2f} | cleaned {len(removed)} paths |\n")
            with open(progress, "a") as f:
                f.write(line)
            committer.commit([result_path(results_dir, rid), progress],
                             f"m0: {plan.grid.name} {label} success {rec['success_rate']:.3f}")
            if (st.packing == 1 and spec.condition.kind == "full" and util < PACK_UTIL_PCT
                    and mem < PACK_MEM_GB):
                st.packing = 2
                log(f"[e] packing enabled: solo full-data run used {util:.0f}% GPU, {mem:.1f} GB")
        if st.stop_reason:
            break
    log(f"[e] done={len(st.done)} skipped={len(st.skipped)} GPU-h={st.gpu_h:.2f} stop={st.stop_reason}")
    return st


def _tbd(x) -> bool:
    from robocurate.config import is_tbd

    return is_tbd(x)


# ---------------------------------------------------------------------------- stage f


def chi2_sd_ci(s2: float, df: int, level: float = 0.95) -> tuple[float, float]:
    from scipy import stats as sps

    if df <= 0:
        return float("nan"), float("nan")
    a = (1 - level) / 2
    return math.sqrt(df * s2 / sps.chi2.ppf(1 - a, df)), math.sqrt(df * s2 / sps.chi2.ppf(a, df))


def grid_units(K: int, m_selection: int, weighting: bool) -> int:
    """Training units per (design) in a launch grid: full + K subsets + oracle + entrants (+ RA-BC: 1 + K)."""
    return 1 + K + 1 + m_selection + ((1 + K) if weighting else 0)


def analyze(records: list[dict], m_selection: int, weighting: bool, scratch: Path | None = None) -> dict[str, Any]:
    from robocurate.report.stats import balanced_mde, balanced_power, variance_components

    full = [r for r in records if r["spec"]["condition"]["kind"] == "full"]
    rnd: dict[int, list[float]] = {}
    for r in records:
        if r["spec"]["condition"]["kind"] == "random":
            rnd.setdefault(int(r["spec"]["subset_index"]), []).append(float(r["success_rate"]))
    out: dict[str, Any] = {"n_records": len(records), "full_rates": [r["success_rate"] for r in full],
                           "subset_rates": {str(k): v for k, v in sorted(rnd.items())}}
    if len(rnd) >= 2:
        vc = variance_components([rnd[k] for k in sorted(rnd)], [[r["success_rate"] for r in full]])
        s_seed, s_sub = math.sqrt(vc.s_seed2), math.sqrt(vc.s_sub2)
        from scipy import stats as sps

        a = 0.025
        msb_lo = vc.df_b * vc.msb / sps.chi2.ppf(1 - a, vc.df_b)
        msb_hi = vc.df_b * vc.msb / sps.chi2.ppf(a, vc.df_b)
        sub_ci = tuple(math.sqrt(max(0.0, (x - vc.s_seed2) / vc.n0)) for x in (msb_lo, msb_hi))
        out.update({"s_seed": s_seed, "s_seed_ci95": chi2_sd_ci(vc.s_seed2, vc.df_w), "df_w": vc.df_w,
                    "s_sub": s_sub, "s_sub_truncated": vc.s_sub2_raw < 0, "s_sub_ci95_approx": sub_ci,
                    "df_b": vc.df_b, "K_m0": vc.K,
                    "note": "s_sub CI: chi-square bounds on MS_between minus the s_seed point estimate, / n0, "
                            "truncated at 0 (approximate)"})
        table = []
        for n in DESIGN_N:
            for K in DESIGN_K:
                for R in DESIGN_R:
                    mde = balanced_mde(s_sub, s_seed, n, K, R, n_comparisons=max(m_selection, 1))
                    pw = balanced_power(0.15, s_sub, s_seed, n, K, R, n_comparisons=max(m_selection, 1))
                    runs = R * n * grid_units(K, m_selection, weighting)
                    table.append({"n": n, "K": K, "R": R, "mde": mde, "power_15pp": pw, "runs": runs})
        out["power_table"] = table
    steps: dict[int, list[float]] = {}
    for r in full:
        for it in (r.get("extra") or {}).get("intermediate", []):
            steps.setdefault(int(it["step"]), []).append(float(it["success_rate"]))
    out["step_curve"] = {str(k): float(np.mean(v)) for k, v in sorted(steps.items())}
    walls = [r["wall_clock_s"] / 3600 for r in records]
    out["cost"] = {"per_run_h_mean": float(np.mean(walls)) if walls else None,
                   "per_run_h_max": float(np.max(walls)) if walls else None, "total_run_h": float(np.sum(walls)),
                   "peak_gpu_mem_mb_max": max([r.get("peak_gpu_mem_mb") or 0 for r in records], default=0)}
    if "power_table" in out and walls:
        per = out["cost"]["per_run_h_mean"]
        for row in out["power_table"]:
            row["gpu_h"] = row["runs"] * per
        ok = [r for r in out["power_table"] if r["mde"] <= MDE_TARGET]
        pick = min(ok, key=lambda r: (r["gpu_h"], -r["K"])) if ok else min(out["power_table"], key=lambda r: r["mde"])
        out["proposed_design"] = {**pick, "meets_target": bool(ok), "rule": "docs/preregistration.md section 9 "
                                  "(cheapest with MDE <= 12 pp; else smallest MDE). PROPOSAL ONLY - Vivek decides."}
    if scratch is not None:
        out["disk_gb"] = dir_size_gb(scratch)
    return out


def summary_markdown(task: str, a: dict, st: GridState | None, extra: dict) -> str:
    L = [f"### M0 morning summary: {task}", ""]
    for k, v in extra.items():
        L.append(f"- {k}: {v}")
    if st:
        L.append(f"- grid: done {len(st.done)}, skipped {len(st.skipped)}, GPU-h {st.gpu_h:.2f}, "
                 f"stop: {st.stop_reason or 'completed'}")
    if "s_seed" in a:
        L.append(f"- s_seed {a['s_seed']:.3f} (95% CI {a['s_seed_ci95'][0]:.3f}-{a['s_seed_ci95'][1]:.3f}), "
                 f"s_sub {a['s_sub']:.3f}{' (truncated)' if a['s_sub_truncated'] else ''} (approx. 95% CI "
                 f"{a['s_sub_ci95_approx'][0]:.3f}-{a['s_sub_ci95_approx'][1]:.3f})")
        L += ["", "| n | K | R | runs | MDE (80%) | power at 15 pp | GPU-h |", "|---|---|---|---|---|---|---|"]
        for r in a["power_table"]:
            L.append(f"| {r['n']} | {r['K']} | {r['R']} | {r['runs']} | {r['mde']:.3f} | {r['power_15pp']:.2f} | "
                     f"{r.get('gpu_h', float('nan')):.1f} |")
        if "proposed_design" in a:
            p = a["proposed_design"]
            L.append(f"\nProposed (not adopted): n={p['n']} K={p['K']} R={p['R']}, MDE {p['mde']:.3f}, "
                     f"{p['gpu_h']:.1f} GPU-h, meets 12 pp: {p['meets_target']}.")
    L.append(f"- step curve (full data, mean over seeds): {a.get('step_curve')}")
    L.append(f"- cost: {a.get('cost')}; disk {a.get('disk_gb', float('nan')):.2f} GB")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------- driver


def main(argv: list[str] | None = None) -> int:
    import argparse

    from robocurate.config import load_calibration, load_grid
    from robocurate.paths import repo_root

    ap = argparse.ArgumentParser(prog="m0_pipeline")
    ap.add_argument("--task", choices=("a", "b"), default="a")
    ap.add_argument("--backend", choices=("local", "fake"), default="local")
    ap.add_argument("--dry-run", action="store_true", help="fake probe, temp config copy, no commits")
    ap.add_argument("--fake-params", help="fake backend parameters (dry run)")
    ap.add_argument("--state-dir", help="default: $ROBOCURATE_SCRATCH/m0/<task>")
    ap.add_argument("--results-dir", help="default: results/m0/<calibration name> in the repo")
    ap.add_argument("--stages", default="a,b,c,d,e,f")
    a = ap.parse_args(argv)
    if a.backend == "fake" and not a.dry_run:
        raise KitError("--backend fake is only for --dry-run")
    if a.backend == "local" and a.dry_run:
        raise KitError("--dry-run uses --backend fake")
    env = check_startup(allow_other_scratch=a.dry_run)
    repo = repo_root()
    cal_src = repo / ("configs/calibration/m0_act_transfer_cube_human.yaml" if a.task == "a"
                      else "configs/calibration/pusht_diffusion_m0.yaml")
    grid_path = repo / ("configs/grids/task_a_v1.yaml" if a.task == "a" else "configs/grids/pusht_diffusion_v0.yaml")
    scratch = Path(env["ROBOCURATE_SCRATCH"])
    state = Path(a.state_dir) if a.state_dir else scratch / "m0" / a.task
    state.mkdir(parents=True, exist_ok=True)
    logf = open(state / "m0_pipeline.log", "a")  # noqa: SIM115 - kept open for the whole pipeline

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    if a.dry_run:  # never touch the repo's calibration config, results or PROGRESS.md
        cal_path = state / cal_src.name
        cal_text = cal_src.read_text()
        for k in ("task", "protocol"):
            cal_text = re.sub(rf"^{k}: (configs/\S+)", lambda m, k=k: f"{k}: {repo / m.group(1)}", cal_text,
                              flags=re.M)
        cal_path.write_text(cal_text)
        results_dir = Path(a.results_dir) if a.results_dir else state / "results"
        progress = state / "PROGRESS_dryrun.md"
        probe: Probe = FakeProbe()
        sleep = lambda s: None  # noqa: E731
    else:  # pragma: no cover - GPU host
        cal_path = cal_src
        results_dir = Path(a.results_dir) if a.results_dir else repo / "results" / "m0" / cal_src.stem
        progress = repo / "PROGRESS.md"
        probe = LocalProbe(load_calibration(cal_path, mode="calibration"), state)
        sleep = time.sleep
    committer = Committer(repo, enabled=not a.dry_run, log=log)
    stages = set(a.stages.split(","))
    summary: dict[str, Any] = {"scratch": str(scratch), "dry_run": a.dry_run}
    log(f"start task {a.task} backend {a.backend} cal {cal_path} state {state}")
    summary["disk"] = check_disk(scratch)
    log(f"disk: {summary['disk']}")

    if "a" in stages:
        summary["waited_h"] = stage_a_wait(probe, log, sleep=sleep) / 3600
    smoke = None
    if "b" in stages:
        smoke, hist = stage_b(probe, log, default_num_workers())
        summary["smoke"] = [asdict_round(h) for h in hist]
        summary["num_workers"] = smoke.num_workers
    if "c" in stages:
        ok, detail = probe.gpu_tests()
        log(f"[c] GPU determinism / batch-invariance tests: {'PASS' if ok else 'FAIL'}: {detail[-300:]}")
        summary["gpu_tests"] = {"ok": ok, "detail": detail[-2000:]}
        if not ok:
            (state / "m0_summary.json").write_text(json.dumps(summary, indent=1, default=str))
            raise KitError("stage c failed: GPU determinism / batch invariance (diagnosis in the log); exiting")
    if "d" in stages:
        if smoke is None:
            raise KitError("stage d needs stage b in the same invocation")
        t50 = {b: probe.eval_timing(b, 50) for b in EVAL_BATCHES}
        eb = choose_eval_batch(t50)
        cal = load_calibration(cal_path, mode="calibration")
        n_mid = int(cal.n_intermediate_rollouts)
        eval_s = t50[eb][0] * cal.n_final_rollouts / 50 + 9 * t50[min(eb, n_mid)][0] * n_mid / 50
        sres = choose_steps(default_steps(), smoke.step_s, eval_s)
        vals = {"steps": sres["S"], "intermediate_save_freq": sres["S"] // 10, "eval_batch_size": eb,
                "batch_size": default_batch_size(), "num_workers": smoke.num_workers}
        write_calibration_values(cal_path, vals)
        log(f"[d] eval timings (50 eps): {t50}; eval batch {eb}; steps {sres}; wrote {vals} -> {cal_path}")
        summary["autoconf"] = {"eval_timings_50": t50, **sres, "written": vals}
        if not a.dry_run:  # pragma: no cover
            committer.commit([cal_path], f"m0: stage d autoconf {cal_src.stem} S={sres['S']} eval_batch={eb}")
    st = None
    if "e" in stages:
        from robocurate.runs.planner import make_calibration_plan

        if a.dry_run:
            from robocurate.runs.backends.fake import FakeBackend, load_fake_params

            if not a.fake_params:
                raise KitError("--dry-run needs --fake-params")
            plan = make_calibration_plan(str(cal_path), mode="fake")
            backend = FakeBackend(plan, load_fake_params(a.fake_params))
        else:  # pragma: no cover
            from robocurate.runs.backends.local import LocalBackend

            plan = make_calibration_plan(str(cal_path), mode="calibration")
            backend = LocalBackend(plan, "calibration", plan.grid.dataset, None)
        st = stage_e(plan, backend, results_dir, scratch, probe, progress, committer, log, sleep=sleep)
        summary["grid"] = dataclasses.asdict(st)
    if "f" in stages:
        recs = [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json")) if p.name != "leaderboard.json"]
        from robocurate.cleaners import registry

        g = load_grid(grid_path)
        m_sel = sum(1 for c in g.condition_specs() if c.kind == "cleaner" and registry.disabled_reason(c.name) is None)
        weighting = any(c.kind == "weighting" for c in g.condition_specs())
        an = analyze(recs, m_sel, weighting, scratch)
        summary["analysis"] = an
        md = summary_markdown(f"task {a.task} ({cal_src.stem})", an, st,
                              {"dry run (fake)": a.dry_run, "num_workers": summary.get("num_workers"),
                               "autoconf": {k: summary.get("autoconf", {}).get(k) for k in ("S", "per_run_h")}})
        (state / "m0_summary.md").write_text(md)
        with open(progress, "a") as f:
            f.write("\n" + md)
        log(f"[f] analysis written to {state / 'm0_summary.md'}")
    (state / "m0_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    log("finished")
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        sys.exit(main())
    except KitError as e:
        print(f"m0_pipeline: STOP: {e}", file=sys.stderr)
        sys.exit(2)
