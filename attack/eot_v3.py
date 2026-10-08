"""Phase 2 · EoT-PGD v3 — protection aimed at what social media actually does.

What went wrong before
----------------------
v1 (attack/eot.py) averaged the loss over random sharing transforms, but the
easy transforms (no change, mild JPEG) give a loss and a gradient that are
100-1000x larger than the hard ones (resize -> JPEG, forwarding). Summing
gradients therefore optimised almost only for the easy cases, so the result
survived JPEG q=90/75 but died after any resize (WhatsApp-like 0 %).

v2's low-resolution perturbation survived resizing, but StarGAN reacts mainly
to fine detail, so a smooth perturbation of the same size barely disrupted it,
even without sharing.

What v3 does
------------
* Full-resolution perturbation (keeps the detail StarGAN is sensitive to).
* Paired target: each sample t is scored as || G(t(x+δ)) - G(t(x)) ||², the
  exact quantity the evaluation measures (an attacker who downloads the
  shared clean photo still gets G(t(x)); only the difference counts).
* Log loss: maximise Σ_t log ||·||². Its gradient is ∇m / m, so a transform
  where the protection is currently weak (m small) gets a large weight
  instead of being drowned out. This is the main fix.
* A transform mix weighted towards resize -> JPEG chains and double sharing,
  with 10 % identity so the unshared photo stays protected.
* Optional momentum (MI-FGSM, Dong et al. CVPR 2018) to smooth the noisy
  EoT gradient.
* All EoT samples go through the model in ONE batch (faster on a GPU). The
  model must accept a batch of B images and return B*K outputs, sample by
  sample (the notebook's StarGANEdits does this).
"""

from __future__ import annotations

import random
from typing import Callable

import torch

from attack.diff_jpeg import diff_jpeg
from eval.robustness import gaussian_blur, resize

Transform = Callable[[torch.Tensor], torch.Tensor]


def sharing_sampler(rng: random.Random, rounding: str = "ste") -> Transform:
    """identity 10 % · JPEG 20 % · resize 15 % · blur 5 %
    · resize→JPEG 35 % · (resize→JPEG)×2 15 %."""
    def jp(x, q):
        return diff_jpeg(x, q, rounding=rounding)

    r = rng.random()
    if r < 0.10:
        return lambda x: x
    if r < 0.30:
        q = rng.uniform(50, 90)
        return lambda x: jp(x, q)
    if r < 0.45:
        s = rng.uniform(0.5, 0.9)
        return lambda x: resize(x, s)
    if r < 0.50:
        sig = rng.uniform(0.5, 1.5)
        return lambda x: gaussian_blur(x, sig)
    if r < 0.85:
        s, q = rng.uniform(0.5, 0.95), rng.uniform(50, 85)
        return lambda x: jp(resize(x, s), q)
    s1, q1 = rng.uniform(0.6, 0.95), rng.uniform(55, 85)
    s2, q2 = rng.uniform(0.6, 0.95), rng.uniform(55, 85)
    return lambda x: jp(resize(jp(resize(x, s1), q1), s2), q2)


def eot_pgd_v3(
    model,
    x: torch.Tensor,
    eps: float = 8 / 255,
    steps: int = 60,
    n_samples: int = 4,
    step_size: float | None = None,
    loss: str = "log",
    paired: bool = True,
    momentum: float = 0.0,
    rounding: str = "ste",
    seed: int = 0,
) -> torch.Tensor:
    """Return a protected copy of x (1 x 3 x H x W, values in [0, 1]).

    loss     : "log" (v3) or "mse" (v1 behaviour)
    paired   : compare against G(t(x)) (v3) or G(x) (v1)
    momentum : 0 = plain sign-PGD, 0.9 = MI-FGSM
    """
    if loss not in ("log", "mse"):
        raise ValueError(f"unknown loss {loss!r}")
    if x.shape[0] != 1:
        raise ValueError("protect one image at a time (x must be 1 x 3 x H x W)")
    if step_size is None:
        step_size = eps / 10
    rng = random.Random(seed)
    torch.manual_seed(seed)

    model.eval()
    x = x.detach()
    with torch.no_grad():
        target = model(x).detach()

    delta = torch.empty_like(x).uniform_(-eps, eps)
    delta = ((x + delta).clamp(0, 1) - x).requires_grad_(True)
    velocity = torch.zeros_like(x)

    for _ in range(steps):
        ts = [sharing_sampler(rng, rounding) for _ in range(n_samples)]
        xp = x + delta
        out = model(torch.cat([t(xp) for t in ts]))
        with torch.no_grad():
            if paired:
                ref = model(torch.cat([t(x) for t in ts]))
            else:
                ref = target.repeat(n_samples, 1, 1, 1)
        per_sample = ((out - ref) ** 2).reshape(n_samples, -1).mean(1)
        obj = torch.log(per_sample + 1e-6).sum() if loss == "log" else per_sample.sum()
        grad, = torch.autograd.grad(obj, delta)
        with torch.no_grad():
            if momentum > 0:
                velocity = momentum * velocity + grad / (grad.abs().mean() + 1e-12)
                direction = velocity.sign()
            else:
                direction = grad.sign()
            delta += step_size * direction
            delta.clamp_(-eps, eps)
            delta.copy_((x + delta).clamp(0, 1) - x)

    return (x + delta).detach()
