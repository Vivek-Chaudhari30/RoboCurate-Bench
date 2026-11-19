# ruff: noqa
# Diagnostic for the CUPID sanity investigation (PROGRESS.md). Usage: python scripts/diagnostics/cupid/diag.py <repo root>. CPU only.
"""CUPID toybed diagnostic: influence under four H / model-output choices, identical rollouts. CPU only."""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, sys.argv[1] + "/tests")
from torch.func import functional_call, grad, hessian, jacrev, vmap
from toybed.demos import flatten, make_dataset
from toybed.env import PointReachEnv

from robocurate.cleaners.entrants.cupid import PROJ_DIM, StochasticPolicy
from robocurate.config import load_protocol
from robocurate.curation.policy import MLPPolicy, train_bc
from robocurate.curation.rollouts import GymRolloutBackend, RolloutService

torch.manual_seed(0)
PROTO = load_protocol(sys.argv[1] + "/configs/protocol.yaml")
view, labels = make_dataset()
pol = train_bc(MLPPolicy(4, 2), *flatten(view), steps=400, seed=0)
pol.eval()
eps = list(view)
S = np.concatenate([e.state for e in eps]).astype(np.float32); A = np.concatenate([e.actions for e in eps]).astype(np.float32)
St, At = torch.as_tensor(S), torch.as_tensor(A)
with torch.no_grad():
    sigma = float((pol(St) - At).pow(2).mean().sqrt())
svc = RolloutService.from_protocol(PROTO, GymRolloutBackend(PointReachEnv), cap=100)
seed = 12345
res = svc.rollout(StochasticPolicy(pol, sigma, seed), svc.next_seeds(100))
nsucc = sum(r.success for r in res)
names = [k for k, _ in pol.named_parameters()]
params = {k: v.detach() for k, v in pol.named_parameters()}
n_params = sum(v.numel() for v in params.values())
def flat(d, n): return torch.cat([d[k].reshape(n, -1) for k in names], 1)
def loss1(p, s, a): return ((functional_call(pol, p, (s[None],)) - a[None]) ** 2).mean()
def out_sq(p, s): return (functional_call(pol, p, (s[None],)) ** 2).mean()
def mu(p, s): return functional_call(pol, p, (s[None],))[0]
G = flat(vmap(grad(loss1), in_dims=(None, 0, 0))(params, St, At), len(St))          # (N, P) loss grads
Rs = np.concatenate([r.states for r in res if r.length]).astype(np.float32)
Ra = np.concatenate([r.actions for r in res if r.length]).astype(np.float32)
Rw = np.concatenate([np.full(r.length, 1.0 if r.success else -1.0) for r in res if r.length])
Rst, Rat = torch.as_tensor(Rs), torch.as_tensor(Ra)
Gr = flat(vmap(grad(loss1), in_dims=(None, 0, 0))(params, Rst, Rat), len(Rst))
u = (torch.as_tensor(Rw, dtype=torch.float32)[:, None] * Gr).sum(0)                 # sum_tau R sum grad l
bounds = np.cumsum([0] + [e.length for e in eps])
def episode_scores(per_frame): return {e.episode_index: float(per_frame[bounds[j]:bounds[j+1]].sum()) for j, e in enumerate(eps)}
def kept(psi):
    k = len(labels) // 2
    order = sorted(psi, key=lambda i: -psi[i])[:k]
    return {kind: round(float(np.mean([i in order for i in labels if labels[i] == kind])), 3) for kind in ("clean","pause","jitter","wrong")}
out = {"n_params": n_params, "sigma": sigma, "rollouts": len(res), "successes": int(nsucc), "N_train_frames": len(St)}
# A. ours: phi = projected loss grads, H = phi^T phi (empirical Fisher), pinv
P = torch.randn(n_params, PROJ_DIM, generator=torch.Generator().manual_seed(seed)) / PROJ_DIM**0.5
phi = G @ P; Hp = phi.T @ phi
wA = torch.linalg.pinv(Hp.double()).float() @ (u @ P)
out["A_ours_empirical_fisher_proj"] = kept(episode_scores((phi @ wA).numpy()))
# A'. same without projection
HF = G.T @ G
wA2 = torch.linalg.pinv(HF.double()).float() @ u
out["A2_empirical_fisher_noproj"] = kept(episode_scores((G @ wA2).numpy()))
# B. exact Hessian of the mean training loss (textbook influence)
def total_loss(p): return ((functional_call(pol, p, (St,)) - At) ** 2).mean(-1).mean()
Hd = hessian(total_loss)(params)
H = torch.cat([torch.cat([Hd[a][b].reshape(params[a].numel(), params[b].numel()) for b in names], 1) for a in names], 0)
evals = torch.linalg.eigvalsh(H.double())
out["exact_H_eig_min_max"] = [float(evals.min()), float(evals.max())]
wB = torch.linalg.pinv(H.double() * len(St)).float() @ u   # H of the SUM of per-sample losses
out["B_exact_hessian"] = kept(episode_scores((G @ wB).numpy()))
# C. Gauss-Newton with model output mu(s) (2 outputs): H_GN = sum_i J_i^T J_i * (2/A)
J = vmap(jacrev(mu), in_dims=(None, 0))(params, St)            # dict of (N, 2, ...)
Jf = torch.cat([J[k].reshape(len(St), 2, -1) for k in names], 2)   # (N, 2, P)
HGN = torch.einsum("noP,noQ->PQ", Jf, Jf) * (2.0 / 2)
wC = torch.linalg.pinv(HGN.double()).float() @ u
out["C_gauss_newton_output_mu"] = kept(episode_scores((G @ wC).numpy()))
# D. official 'square' analog: f = mean(mu(s)^2), out_to_loss = 1 (train and target features = grad f)
F = flat(vmap(grad(out_sq), in_dims=(None, 0))(params, St), len(St))
Fr = flat(vmap(grad(out_sq), in_dims=(None, 0))(params, Rst), len(Rst))
uF = (torch.as_tensor(Rw, dtype=torch.float32)[:, None] * Fr).sum(0)
phiF = F @ P
wD = torch.linalg.pinv((phiF.T @ phiF).double()).float() @ (uF @ P)
out["D_official_square_output_proj"] = kept(episode_scores((phiF @ wD).numpy()))
# E. sanity: oracle-ish -- per-episode mean training loss (high loss = wrong) for reference
pl = ((pol(St) - At) ** 2).mean(-1).detach().numpy()
out["ref_low_loss_kept"] = kept({k: -v for k, v in episode_scores(pl).items()})
print(json.dumps(out, indent=1))
