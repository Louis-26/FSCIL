"""Forward/reverse diffusion in image space (Sec. 3.2, Eqs. 3-6) + DDIM sampling.

    Eq. 3   q(v_t | v_0) = N( sqrt(a_bar_t) v_0, (1 - a_bar_t) I )
    Eq. 5   L = E_{t,v0,eps} || eps - eps_theta(v_t, t, phi(p_c)) ||^2
    Eq. 4   deterministic DDIM update, T_sample = 50 by default

The noise schedule is the cosine schedule of Nichol & Dhariwal (ref [24] in the
paper) and T = 1000 as stated in Sec. 4.

Classifier-free guidance is *not* mentioned in the paper.  We train with a 10%
condition-dropout so that guidance is available, and default the sampler to
`guidance=1.0` -- which is mathematically identical to no guidance at all, so
the faithful setting is the default and guidance is an opt-in ablation.
"""
from __future__ import annotations

import math

import torch


def cosine_beta_schedule(T: int, s: float = 0.008, max_beta: float = 0.999):
    """Nichol & Dhariwal cosine schedule."""
    def f(t):
        return math.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
    betas = []
    for i in range(T):
        betas.append(min(1 - f(i + 1) / f(i), max_beta))
    return torch.tensor(betas, dtype=torch.float64)


def linear_beta_schedule(T: int):
    scale = 1000 / T
    return torch.linspace(scale * 1e-4, scale * 0.02, T, dtype=torch.float64)


class GaussianDiffusion:
    """Holds the schedule and the training / sampling maths (no parameters)."""

    def __init__(self, num_timesteps: int = 1000, schedule: str = "cosine",
                 device: torch.device | str = "cuda"):
        self.T = num_timesteps
        betas = (cosine_beta_schedule(num_timesteps) if schedule == "cosine"
                 else linear_beta_schedule(num_timesteps))
        alphas = 1.0 - betas
        ab = torch.cumprod(alphas, dim=0)
        self.betas = betas.float().to(device)
        self.alphas_cumprod = ab.float().to(device)
        self.sqrt_ab = ab.sqrt().float().to(device)
        self.sqrt_1mab = (1 - ab).sqrt().float().to(device)
        self.device = device

    # ------------------------------------------------------------------ #
    @staticmethod
    def _bcast(v: torch.Tensor, ndim: int) -> torch.Tensor:
        """(B,) -> (B, 1, 1, ...) so the schedule broadcasts over images (4-D)
        or feature vectors (2-D) alike."""
        return v.reshape(-1, *([1] * (ndim - 1)))

    def q_sample(self, v0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor | None = None):
        """Eq. 3: draw v_t given a clean sample v_0."""
        if noise is None:
            noise = torch.randn_like(v0)
        a = self._bcast(self.sqrt_ab[t], v0.ndim)
        b = self._bcast(self.sqrt_1mab[t], v0.ndim)
        return a * v0 + b * noise, noise

    def training_loss(self, model, v0: torch.Tensor, cond: torch.Tensor,
                      p_uncond: float = 0.1):
        """Eq. 5, with classifier-free-guidance condition dropout."""
        b = v0.shape[0]
        t = torch.randint(0, self.T, (b,), device=v0.device)
        vt, noise = self.q_sample(v0, t)
        drop = torch.rand(b, device=v0.device) < p_uncond
        eps = model(vt, t, cond, drop_mask=drop)
        return torch.nn.functional.mse_loss(eps, noise)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def ddim_sample(self, model, shape, cond: torch.Tensor | None,
                    steps: int = 50, eta: float = 0.0, guidance: float = 1.0,
                    device=None, generator: torch.Generator | None = None,
                    progress=None, clip_denoised: bool = True):
        """Eq. 4.  `guidance=1.0` reproduces the paper's plain conditional sampler.

        Works for image tensors (B,C,H,W) and for feature vectors (B,D)."""
        device = device or self.device
        v = torch.randn(shape, device=device, generator=generator)

        ts = torch.linspace(0, self.T - 1, steps, device=device).long().flip(0)
        it = enumerate(ts)
        if progress is not None:
            it = progress(it, total=len(ts))

        for i, t in it:
            tb = t.expand(shape[0])
            if guidance != 1.0 and cond is not None:
                eps_c = model(v, tb, cond)
                eps_u = model(v, tb, None)
                eps = eps_u + guidance * (eps_c - eps_u)
            else:
                eps = model(v, tb, cond)

            ab_t = self.alphas_cumprod[t]
            ab_prev = self.alphas_cumprod[ts[i + 1]] if i + 1 < len(ts) \
                else torch.tensor(1.0, device=device)

            # predicted clean image  (the bracketed term of Eq. 4)
            v0 = (v - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
            if clip_denoised:
                v0 = v0.clamp(-1, 1)

            sigma = eta * ((1 - ab_prev) / (1 - ab_t)).sqrt() * (1 - ab_t / ab_prev).sqrt()
            dir_xt = (1 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * eps
            v = ab_prev.sqrt() * v0 + dir_xt
            if eta > 0 and i + 1 < len(ts):
                v = v + sigma * torch.randn(shape, device=device, generator=generator)
        return v


def to_pil_batch(v: torch.Tensor):
    """[-1,1] float tensor -> uint8 numpy (B,H,W,3) for CLIP re-encoding."""
    v = (v.clamp(-1, 1) + 1) * 127.5
    return v.permute(0, 2, 3, 1).round().to(torch.uint8).cpu().numpy()
