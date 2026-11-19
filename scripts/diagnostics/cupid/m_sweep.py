# ruff: noqa
# Diagnostic for the CUPID sanity investigation (PROGRESS.md). Usage: python scripts/diagnostics/cupid/m_sweep.py <repo root>. CPU only.
"""Diagnostic only: CUPID influence (our implementation's math) vs number of rollouts m. Not a change to the entrant."""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, sys.argv[1] + "/tests")
from toybed.demos import flatten, make_dataset
from toybed.env import PointReachEnv

from robocurate.cleaners.entrants.cupid import PROJ_DIM, StochasticPolicy, per_sample_projected_grads
from robocurate.curation.policy import MLPPolicy, train_bc
from robocurate.curation.rollouts import GymRolloutBackend

view, labels = make_dataset()
pol = train_bc(MLPPolicy(4, 2), *flatten(view), steps=400, seed=0); pol.eval()
eps = list(view)
S = np.concatenate([e.state for e in eps]).astype(np.float32); A = np.concatenate([e.actions for e in eps]).astype(np.float32)
with torch.no_grad():
    sigma = float((pol(torch.as_tensor(S)) - torch.as_tensor(A)).pow(2).mean().sqrt())
n_params = sum(p.numel() for p in pol.parameters())
P = torch.randn(n_params, PROJ_DIM, generator=torch.Generator().manual_seed(1)) / PROJ_DIM**0.5
phi = per_sample_projected_grads(pol, S, A, P)
Hinv = torch.linalg.pinv((phi.T @ phi).double()).float()
bounds = np.cumsum([0] + [e.length for e in eps])
be = GymRolloutBackend(PointReachEnv)
def kept_and_auc(m, rep):
    res = be.run(StochasticPolicy(pol, sigma, 1000 + rep), list(range(700000 + rep * 100000, 700000 + rep * 100000 + m)))
    u = torch.zeros(PROJ_DIM)
    for r in res:
        if r.length:
            u += (1.0 if r.success else -1.0) * per_sample_projected_grads(pol, r.states, r.actions, P).sum(0)
    pf = (phi @ (Hinv @ u)).numpy()
    psi = {e.episode_index: float(pf[bounds[j]:bounds[j+1]].sum()) for j, e in enumerate(eps)}
    k = len(labels) // 2
    top = set(sorted(psi, key=lambda i: -psi[i])[:k])
    w = [psi[i] for i in labels if labels[i] == "wrong"]; c = [psi[i] for i in labels if labels[i] == "clean"]
    auc = float(np.mean([[x < y for y in c] for x in w]))  # P(wrong ranked below clean)
    return {"wrong_kept": float(np.mean([i in top for i in labels if labels[i] == "wrong"])),
            "clean_kept": float(np.mean([i in top for i in labels if labels[i] == "clean"])),
            "auc_wrong_below_clean": round(auc, 3), "succ": int(sum(r.success for r in res))}
out = {m: [kept_and_auc(m, rep) for rep in range(3)] for m in (100, 1000, 4000)}
print(json.dumps(out, indent=0))
