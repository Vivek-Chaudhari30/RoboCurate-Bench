"""Demo-SCORE (reimplementation). Chen, Lessing et al., "Curating Demonstrations using Online Experience",
RSS 2025, arXiv 2503.03707v2. Official code github.com/alessing/demo-score@b82a5ece97b8975643b14703734afa8f24ab7b0a
has NO license at the repo root, so it is not wrapped; this is a reimplementation from the paper.

Paper (Sec. IV-A, App. A-B): a success classifier q(y=1|s_t) on proprioceptive state only; MLP [8, 8], sigmoid,
binary cross-entropy, dropout 0.3, AdamW weight decay 0.1 (code: lr 1e-4, batch 1024,
demo_score/train_stepwise_classifier.py:73-74); keep the epoch with the lowest validation BCE. Demo score
q_bar(tau) = mean_t q(y=1|s_t) (Sec. IV-B, Alg. 1).

Benchmark adaptations (declared, not tuned):
  * rollouts: the paper trains on rollouts of base-policy checkpoints 1..C-1 and validates on checkpoint C. Under
    D-022 a cleaner gets ONE reference checkpoint and <= 100 curation rollouts, so we roll out the reference
    policy 100 times and hold out the last 20% of rollouts (by seed order) for validation / epoch selection.
  * selection: the paper keeps demos with q_bar > gamma; the selection track fixes the budget, so we keep the
    top-k by q_bar (ties D-021). The paper's extra step (adding successful rollouts to the training data) is not
    a selection of the given dataset and is omitted.
  * code-only BCE class weighting (train_stepwise_classifier.py:172) is NOT used (paper: plain BCE).
  * epochs: not stated in the paper; the reference code's default num_eps = 3000
    (train_stepwise_classifier.py:73); we keep the best validation epoch (paper; code :187).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from robocurate.cleaners.base import ScoreCleaner
from robocurate.data.episodes import DatasetView, EpisodeData

HIDDEN, DROPOUT, LR, WEIGHT_DECAY, BATCH = (8, 8), 0.3, 1e-4, 0.1, 1024
N_ROLLOUTS, VAL_FRACTION, MAX_EPOCHS = 100, 0.2, 3000


def _mlp(d: int) -> nn.Module:
    layers, prev = [], d
    for h in HIDDEN:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(DROPOUT)]
        prev = h
    return nn.Sequential(*layers, nn.Linear(prev, 1))


def train_classifier(S_tr, y_tr, S_va, y_va, seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    mu, sd = S_tr.mean(0), S_tr.std(0) + 1e-6
    net = _mlp(S_tr.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lossf = nn.BCEWithLogitsLoss()
    Xtr, Ytr = torch.as_tensor((S_tr - mu) / sd, dtype=torch.float32), torch.as_tensor(y_tr, dtype=torch.float32)
    Xva, Yva = torch.as_tensor((S_va - mu) / sd, dtype=torch.float32), torch.as_tensor(y_va, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    best, best_state = float("inf"), None
    for _ in range(MAX_EPOCHS):
        net.train()
        for idx in torch.randperm(len(Xtr), generator=g).split(BATCH):
            loss = lossf(net(Xtr[idx]).squeeze(-1), Ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(lossf(net(Xva).squeeze(-1), Yva))
        if v < best:
            best, best_state = v, {k: t.clone() for k, t in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    net.norm = (mu, sd)
    return net


class DemoScore(ScoreCleaner):
    name, version = "demo_score", "1"
    requires_rollouts = N_ROLLOUTS
    requires_reference_policy = True
    implementation = "reimplementation"
    source = ("Chen, Lessing et al. 2025, arXiv 2503.03707v2 Sec. IV, App. A-B; ref code (no license) "
              "github.com/alessing/demo-score@b82a5ec")

    def select(self, view: DatasetView, budget: int, rng: np.random.Generator) -> list[int]:
        ctx = self.context
        pol = ctx.reference_policy.load()
        n = min(N_ROLLOUTS, ctx.rollouts.remaining)
        res = ctx.rollouts.rollout(pol, ctx.rollouts.next_seeds(n))
        n_va = max(1, int(round(VAL_FRACTION * len(res))))
        tr, va = res[:-n_va], res[-n_va:]

        def stack(rs):
            S = np.concatenate([r.states for r in rs]).astype(np.float32)
            y = np.concatenate([np.full(len(r.states), float(r.success)) for r in rs]).astype(np.float32)
            return S, y

        (S_tr, y_tr), (S_va, y_va) = stack(tr), stack(va)
        net = train_classifier(S_tr, y_tr, S_va, y_va, seed=int(rng.integers(2**31)))
        mu, sd = net.norm
        self._q = {}
        with torch.no_grad():
            for e in view:
                x = torch.as_tensor((np.asarray(e.state, dtype=np.float32) - mu) / sd)
                self._q[e.episode_index] = float(torch.sigmoid(net(x)).mean())
        return super().select(view, budget, rng)

    def score(self, episode: EpisodeData) -> float:
        return self._q[episode.episode_index]
