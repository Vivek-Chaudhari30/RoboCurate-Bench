"""A minimal behavior-cloning policy with a `loss(batch)` interface (torch, CPU).

Used as (a) the policy class of the CPU toy testbed and (b) the fake reference policy in conformance checks.
Generic influence methods (CUPID) only need `loss(batch) -> scalar` and `parameters()`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


class MLPPolicy(nn.Module):
    """obs (state_dim) -> action (action_dim). Batch: {"observation.state": (B, S), "action": (B, A)}."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, action_dim))
        with torch.no_grad():
            for p in self.net.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * (0.5 / max(1, p.shape[-1]) ** 0.5))
        self.state_dim, self.action_dim, self.hidden = state_dim, action_dim, hidden

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

    def per_sample_loss(self, batch: dict) -> torch.Tensor:
        pred = self(batch["observation.state"])
        return ((pred - batch["action"]) ** 2).mean(-1)

    def loss(self, batch: dict) -> torch.Tensor:
        return self.per_sample_loss(batch).mean()

    @torch.no_grad()
    def act(self, state: np.ndarray) -> np.ndarray:
        return self(torch.as_tensor(np.asarray(state, dtype=np.float32))).numpy()

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "dims": (self.state_dim, self.action_dim, self.hidden)},
                   path / "mlp_policy.pt")
        return path

    @classmethod
    def load(cls, path: str | Path) -> MLPPolicy:
        d = torch.load(Path(path) / "mlp_policy.pt", weights_only=True)
        m = cls(*d["dims"])
        m.load_state_dict(d["state_dict"])
        return m


def make_batch(states: np.ndarray, actions: np.ndarray) -> dict:
    return {"observation.state": torch.as_tensor(np.array(states, dtype=np.float32)),  # copy: views are read-only
            "action": torch.as_tensor(np.array(actions, dtype=np.float32))}


def train_bc(policy: MLPPolicy, states: np.ndarray, actions: np.ndarray, *, steps: int = 300, lr: float = 1e-2,
             batch_size: int = 256, seed: int = 0, weights: np.ndarray | None = None) -> MLPPolicy:
    """Plain (optionally sample-weighted) BC with Adam. Weighted loss mirrors LeRobot's
    sum(w * l) / sum(w) (lerobot/scripts/lerobot_train.py:162)."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    n = len(states)
    w = None if weights is None else np.asarray(weights, dtype=np.float32)
    for _ in range(steps):
        idx = rng.integers(0, n, size=min(batch_size, n))
        b = make_batch(states[idx], actions[idx])
        per = policy.per_sample_loss(b)
        if w is None:
            loss = per.mean()
        else:
            ww = torch.as_tensor(w[idx])
            loss = (per * ww).sum() / (ww.sum() + 1e-6)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return policy
