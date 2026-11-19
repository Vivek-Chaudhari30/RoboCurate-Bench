"""Small DDPM diffusion policy for the 2D reach toybed (CPU only), used to test CUPID in its native policy family
(D-029). Structure follows Diffusion Policy as configured in LeRobot 0.6.1 (policies/diffusion/
configuration_diffusion.py:139-147): DDPM, num_train_timesteps 100, beta_schedule "squaredcos_cap_v2", epsilon
prediction, clip_sample to [-1, 1], inference with all 100 steps (num_inference_steps None -> num_train_timesteps).
Toy simplifications (declared, fixed before any CUPID run): one observation step, an action chunk of HORIZON = 8
executed open-loop (n_action_steps = 8), an MLP denoiser instead of the 1D U-Net, no EMA, observations
standardized with dataset statistics (actions are already in [-1, 1], the MIN_MAX range).

`make_samples` is the sampler: one training sample per frame t = (obs_t, actions[t : t + HORIZON]), padded by
repeating the last action (Diffusion Policy pads past the episode end the same way). Rollout targets for CUPID are
built with the same function from the executed rollout.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

HORIZON, T_TRAIN, HIDDEN, TEMB = 8, 100, 128, 16


def squaredcos_cap_v2_betas(T: int = T_TRAIN, max_beta: float = 0.999) -> torch.Tensor:
    """diffusers DDPMScheduler beta_schedule="squaredcos_cap_v2" (betas_for_alpha_bar, cosine alpha_bar of
    Nichol & Dhariwal 2021): beta_i = min(1 - abar((i+1)/T) / abar(i/T), max_beta)."""
    def abar(t):
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    return torch.tensor([min(1 - abar((i + 1) / T) / abar(i / T), max_beta) for i in range(T)], dtype=torch.float32)


def _time_embedding(t: torch.Tensor, dim: int = TEMB) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
    x = t.float().reshape(-1, 1) * freqs.reshape(1, -1)
    return torch.cat([x.sin(), x.cos()], dim=-1)


class Denoiser(nn.Module):
    """eps_theta(noisy_chunk, t, obs) -> (B, HORIZON, act_dim)."""

    def __init__(self, obs_dim: int = 4, act_dim: int = 2, horizon: int = HORIZON, hidden: int = HIDDEN):
        super().__init__()
        self.horizon, self.act_dim = horizon, act_dim
        self.net = nn.Sequential(nn.Linear(horizon * act_dim + obs_dim + TEMB, hidden), nn.Mish(),
                                 nn.Linear(hidden, hidden), nn.Mish(), nn.Linear(hidden, horizon * act_dim))

    def forward(self, noisy: torch.Tensor, t: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        b = noisy.shape[0]
        x = torch.cat([noisy.reshape(b, -1), obs.reshape(b, -1), _time_embedding(t).reshape(b, -1)], dim=-1)
        return self.net(x).reshape(b, self.horizon, self.act_dim)


class ToyDiffusionPolicy(nn.Module):
    policy_family = "diffusion"

    def __init__(self, obs_dim: int = 4, act_dim: int = 2, horizon: int = HORIZON, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.horizon = horizon
        self.denoiser = Denoiser(obs_dim, act_dim, horizon)
        betas = squaredcos_cap_v2_betas()
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(1.0 - betas, 0))
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))
        self.num_train_timesteps = T_TRAIN
        self._queue: list[np.ndarray] = []
        self._gen = torch.Generator().manual_seed(seed)

    # ---- data
    def set_stats(self, states: np.ndarray) -> None:
        s = torch.as_tensor(np.asarray(states, dtype=np.float32))
        self.obs_mean.copy_(s.mean(0))
        self.obs_std.copy_(s.std(0).clamp_min(1e-6))

    def normalize_obs(self, obs) -> torch.Tensor:
        return (torch.as_tensor(np.array(obs, dtype=np.float32)) - self.obs_mean) / self.obs_std

    def make_samples(self, states: np.ndarray, actions: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """(obs (N, obs_dim) normalized, chunks (N, HORIZON, act_dim)) for one episode."""
        a = np.asarray(actions, dtype=np.float32)
        n = len(a)
        idx = np.minimum(np.arange(n)[:, None] + np.arange(self.horizon)[None, :], n - 1)
        return self.normalize_obs(states), torch.as_tensor(a[idx])

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        ab = self.alphas_cumprod[t.long()].reshape(-1, *([1] * (x0.dim() - 1)))
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    # ---- training (DDPM epsilon loss)
    def ddpm_loss(self, obs: torch.Tensor, chunks: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        t = torch.randint(0, self.num_train_timesteps, (len(obs),), generator=gen)
        noise = torch.randn(chunks.shape, generator=gen)
        pred = self.denoiser(self.add_noise(chunks, noise, t), t, obs)
        return ((pred - noise) ** 2).mean()

    # ---- inference (DDPM ancestral sampling, all T steps, clip x0 to [-1, 1])
    @torch.no_grad()
    def sample(self, obs: torch.Tensor) -> torch.Tensor:
        b = obs.shape[0]
        x = torch.randn((b, self.horizon, self.denoiser.act_dim), generator=self._gen)
        ab, betas = self.alphas_cumprod, self.betas
        for t in range(self.num_train_timesteps - 1, -1, -1):
            tt = torch.full((b,), t, dtype=torch.long)
            eps = self.denoiser(x, tt, obs)
            ab_t = ab[t]
            ab_prev = ab[t - 1] if t > 0 else torch.tensor(1.0)
            x0 = ((x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()).clamp(-1, 1)
            c0 = ab_prev.sqrt() * betas[t] / (1 - ab_t)
            ct = (1 - betas[t]).sqrt() * (1 - ab_prev) / (1 - ab_t)
            x = c0 * x0 + ct * x
            if t > 0:
                var = betas[t] * (1 - ab_prev) / (1 - ab_t)
                x = x + var.clamp_min(1e-20).sqrt() * torch.randn(x.shape, generator=self._gen)
        return x

    def reset(self) -> None:
        self._queue = []

    def act(self, obs) -> np.ndarray:
        if not self._queue:
            chunk = self.sample(self.normalize_obs(obs).reshape(1, -1))[0].numpy()
            self._queue = list(chunk)
        return self._queue.pop(0).astype(np.float32)

    def reseed(self, seed: int) -> ToyDiffusionPolicy:
        self._gen = torch.Generator().manual_seed(int(seed))
        return self

    # ---- persistence (ReferencePolicy loader)
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "model.pt")
        return path

    @classmethod
    def load(cls, path: str | Path) -> ToyDiffusionPolicy:
        p = cls()
        p.load_state_dict(torch.load(Path(path) / "model.pt", weights_only=True))
        p.eval()
        return p


def train_diffusion(view, ids=None, steps: int = 4000, batch: int = 256, lr: float = 1e-3, seed: int = 0
                    ) -> ToyDiffusionPolicy:
    ids = view.episode_indices if ids is None else list(ids)
    pol = ToyDiffusionPolicy(seed=seed)
    eps = [view.episode(i) for i in ids]
    pol.set_stats(np.concatenate([e.state for e in eps]))
    obs_l, chunk_l = zip(*[pol.make_samples(e.state, e.actions) for e in eps], strict=True)
    obs_all, chunks = torch.cat(obs_l), torch.cat(chunk_l)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(pol.denoiser.parameters(), lr=lr, betas=(0.95, 0.999), weight_decay=1e-6)
    pol.train()
    for _ in range(steps):
        j = torch.randint(0, len(obs_all), (batch,), generator=gen)
        loss = pol.ddpm_loss(obs_all[j], chunks[j], gen)
        opt.zero_grad()
        loss.backward()
        opt.step()
    pol.eval()
    return pol


def diffusion_success_rate(policy: ToyDiffusionPolicy, seeds, sample_seed: int = 0) -> float:
    from toybed.env import PointReachEnv

    policy.reseed(sample_seed)
    ok = 0
    for s in seeds:
        env = PointReachEnv()
        obs, _ = env.reset(seed=int(s))
        policy.reset()
        info = {"is_success": False}
        for _ in range(env.max_steps):
            obs, _, term, trunc, info = env.step(policy.act(obs))
            if term or trunc:
                break
        ok += bool(info["is_success"])
    return ok / len(seeds)
