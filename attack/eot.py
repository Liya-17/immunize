"""Phase 2 · PGD + Expectation over Transformation (task 2.2).

Plain PGD (attack/pgd.py) finds a perturbation that breaks the model on the
EXACT protected pixels. Social media changes those pixels (JPEG, resize), and
the protection can wash out. EoT (Athalye et al., ICML 2018) fixes this by
optimising the perturbation *through* random versions of those changes:

    maximise  E_t [ || G(t(x + δ)) - G(x) ||² ]   subject to  ||δ||∞ ≤ ε

where t is sampled each step from a distribution of sharing transforms
(differentiable JPEG, resize, blur, and chains of them). The perturbation that
comes out is one that still breaks G *after* the photo has been compressed.
"""

from __future__ import annotations

import random
from typing import Callable

import torch
import torch.nn.functional as F

from attack.diff_jpeg import diff_jpeg
from eval.robustness import gaussian_blur, resize

Transform = Callable[[torch.Tensor], torch.Tensor]


def default_sampler(rng: random.Random) -> Transform:
    """Draw one random, differentiable 'sharing' transform.

    Mix (probabilities): identity 10 % · JPEG 35 % · resize 15 % · blur 10 %
    · resize→JPEG chain 30 % (the WhatsApp/Instagram pattern).
    JPEG quality 50–95 covers what messaging and social apps use.
    """
    r = rng.random()
    if r < 0.10:
        return lambda x: x
    if r < 0.45:
        q = rng.uniform(50, 95)
        return lambda x: diff_jpeg(x, q)
    if r < 0.60:
        s = rng.uniform(0.5, 1.0)
        return lambda x: resize(x, s)
    if r < 0.70:
        sig = rng.uniform(0.4, 1.5)
        return lambda x: gaussian_blur(x, sig)
    s, q = rng.uniform(0.6, 0.95), rng.uniform(55, 85)
    return lambda x: diff_jpeg(resize(x, s), q)


def eot_pgd_disrupt(
    model,
    x: torch.Tensor,
    eps: float = 8 / 255,
    steps: int = 60,
    step_size: float | None = None,
    n_samples: int = 4,
    sampler: Callable[[random.Random], Transform] = default_sampler,
    random_start: bool = True,
    seed: int = 0,
) -> torch.Tensor:
    """Return a protected copy of `x` that stays disruptive after compression.

    Same interface as `attack.pgd.pgd_disrupt`, plus:
      n_samples : transforms averaged per step (the "expectation"); cost is
                  roughly steps × n_samples forward/backward passes.
      sampler   : function(rng) -> differentiable transform.
    """
    if step_size is None:
        step_size = eps / 10
    rng = random.Random(seed)
    torch.manual_seed(seed)

    model.eval()
    x = x.detach()
    with torch.no_grad():
        target = model(x).detach()  # the normal deepfake we want to move away from

    delta = torch.empty_like(x).uniform_(-eps, eps) if random_start else torch.zeros_like(x)
    delta = ((x + delta).clamp(0, 1) - x).requires_grad_(True)

    for _ in range(steps):
        grad = torch.zeros_like(x)
        for _ in range(n_samples):
            t = sampler(rng)
            out = model(t(x + delta))
            loss = F.mse_loss(out, target)
            g, = torch.autograd.grad(loss, delta)
            grad += g
        with torch.no_grad():
            delta += step_size * grad.sign()
            delta.clamp_(-eps, eps)
            delta.copy_((x + delta).clamp(0, 1) - x)

    return (x + delta).detach()
