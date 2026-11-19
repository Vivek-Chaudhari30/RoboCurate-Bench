# ruff: noqa
# Diagnostic for the CUPID sanity investigation (PROGRESS.md). Usage: python scripts/diagnostics/cupid/rank.py <repo root>. CPU only.
"""Diagnostic: CUPID (paper defaults d=4000, lambda=0, m=100) on toybeds with N_train frames < d vs N_train >> d."""
import json
import sys
import tempfile

import numpy as np

sys.path.insert(0, sys.argv[1] + "/tests")
from toybed.demos import flatten, make_dataset
from toybed.env import PointReachEnv

import robocurate.cleaners.entrants.cupid as C
from robocurate.cleaners import registry
from robocurate.config import load_protocol
from robocurate.curation.context import CurationContext, ReferencePolicy
from robocurate.curation.policy import MLPPolicy, train_bc
from robocurate.curation.rollouts import GymRolloutBackend, RolloutService

PROTO = load_protocol(sys.argv[1] + "/configs/protocol.yaml")
def run(scale, proj=None, seed=0):
    view, labels = make_dataset(30 * scale, 6 * scale, 6 * scale, 8 * scale, seed=seed)
    ref = train_bc(MLPPolicy(4, 2), *flatten(view), steps=400, seed=0).save(tempfile.mkdtemp() + "/ref")
    old = C.PROJ_DIM
    if proj: C.PROJ_DIM = proj
    try:
        cl = registry.get("cupid")
        cl.context = CurationContext(rollouts=RolloutService.from_protocol(PROTO, GymRolloutBackend(PointReachEnv), cap=100),
                                     reference_policy=ReferencePolicy(ref, MLPPolicy.load))
        ids = set(cl.select(view, len(labels) // 2, np.random.default_rng(0)))
        psi = cl._psi
    finally:
        C.PROJ_DIM = old
    w = [psi[i] for i in labels if labels[i] == "wrong"]; c = [psi[i] for i in labels if labels[i] == "clean"]
    N = sum(view.lengths().values())
    return {"scale": scale, "N_frames": N, "d": proj or old,
            "kept": {k: round(float(np.mean([i in ids for i in labels if labels[i] == k])), 3) for k in ("clean", "pause", "jitter", "wrong")},
            "auc_wrong_below_clean": round(float(np.mean([[x < y for y in c] for x in w])), 3), **cl.diagnostics}
out = [run(1), run(1, proj=400), run(10), run(10, seed=1)]
for o in out: print(json.dumps(o))
