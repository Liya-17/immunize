"""Phase 2 · PGD + Expectation over Transformation (task 2.2).

Plain PGD (attack/pgd.py) finds a perturbation that breaks the model on the
EXACT protected pixels. Social media changes those pixels (JPEG, resize), and
the protection can wash out. EoT (Athalye et al., ICML 2018) fixes this by
optimising the perturbation *through* random versions of those changes:

    maximise  E_t [ || G(t(x + δ)) - G(t(x)) ||² ]   subject to  ||δ||∞ ≤ ε

where t is sampled each step from a distribution of sharing transforms
(differentiable JPEG, resize, blur, and chains of them). The perturbation that
comes out is one that still breaks G *after* the photo has been compressed.

v2 options (all off by default, so old notebooks give the same results):

  paired_target=True  Compare against G(t(x_clean)) with the SAME t, which is
                      exactly how eval/robustness.py scores success. v1
                      compared against G(x_clean), so part of the "disruption"
                      it optimised was just compression's own effect, which
                      the evaluation then subtracts away.
  lowres=128 (or 64)  Optimise δ on a low-resolution grid and upsample it.
                      JPEG throws away high-frequency detail and resizing
                      averages it out; a smooth δ lives in the frequencies
                      both keep. Bilinear upsampling never exceeds the grid's
                      values, so ||δ||∞ ≤ ε still holds.
  loss="log"          Maximise log(MSE) per transform instead of MSE. The
                      gradient of log(m) is ∇m / m, so the transforms where
                      protection is weakest (small m — heavy JPEG, resize
                      chains) get the most weight instead of being drowned
                      out by easy ones.
  sampler=make_sampler("sharing", rounding="cubic")
                      A transform mix weighted towards what actually kills
                      protection (resize→JPEG chains, low quality, double
                      sharing), with Shin & Song's cubic rounding instead of
                      the straight-through estimator.
"""

from __future__ import annotations

import math
import random
from typing import Callable

import torch
import torch.nn.functional as F

from attack.diff_jpeg import diff_jpeg
from eval.robustness import gaussian_blur, resize

Transform = Callable[[torch.Tensor], torch.Tensor]
Sampler = Callable[[random.Random], Transform]


def make_sampler(mix: str = "default", rounding: str = "ste") -> Sampler:
    """Build a function rng -> random differentiable sharing transform.

    mix="default" : identity 10 % · JPEG q50–95 35 % · resize 15 % · blur 10 %
                    · resize→JPEG chain 30 %   (the v1 mix)
    mix="sharing" : JPEG q45–85 25 % · resize 10 % · blur 5 %
                    · resize→JPEG chain 45 % · chain applied twice 15 %
                    (no identity: plain PGD already handles the unshared case)
    rounding      : "ste" (v1) or "cubic" — passed to diff_jpeg.
    """
    def jp(x, q):
        return diff_jpeg(x, q, rounding=rounding)

    def default(rng: random.Random) -> Transform:
        r = rng.random()
        if r < 0.10:
            return lambda x: x
        if r < 0.45:
            q = rng.uniform(50, 95)
            return lambda x: jp(x, q)
        if r < 0.60:
            s = rng.uniform(0.5, 1.0)
            return lambda x: resize(x, s)
        if r < 0.70:
            sig = rng.uniform(0.4, 1.5)
            return lambda x: gaussian_blur(x, sig)
        s, q = rng.uniform(0.6, 0.95), rng.uniform(55, 85)
        return lambda x: jp(resize(x, s), q)

    def sharing(rng: random.Random) -> Transform:
        r = rng.random()
        if r < 0.25:
            q = rng.uniform(45, 85)
            return lambda x: jp(x, q)
        if r < 0.35:
            s = rng.uniform(0.45, 0.9)
            return lambda x: resize(x, s)
        if r < 0.40:
            sig = rng.uniform(0.5, 1.5)
            return lambda x: gaussian_blur(x, sig)
        if r < 0.85:
            s, q = rng.uniform(0.5, 0.95), rng.uniform(50, 85)
            return lambda x: jp(resize(x, s), q)
        s1, q1 = rng.uniform(0.6, 0.95), rng.uniform(55, 85)
        s2, q2 = rng.uniform(0.6, 0.95), rng.uniform(55, 85)
        return lambda x: jp(resize(jp(resize(x, s1), q1), s2), q2)

    if mix == "default":
        return default
    if mix == "sharing":
        return sharing
    raise ValueError(f"unknown mix {mix!r}")


default_sampler = make_sampler("default", "ste")


def _upsample(d: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if d.shape[-2:] == size:
        return d
    return F.interpolate(d, size=size, mode="bilinear", align_corners=False)


def eot_pgd_disrupt(
    model,
    x: torch.Tensor,
    eps: float = 8 / 255,
    steps: int = 60,
    step_size: float | None = None,
    n_samples: int = 4,
    sampler: Sampler = default_sampler,
    random_start: bool = True,
    seed: int = 0,
    *,
    paired_target: bool = False,
    lowres: int | None = None,
    loss: str = "mse",
) -> torch.Tensor:
    """Return a protected copy of `x` that stays disruptive after compression.

    Same interface as `attack.pgd.pgd_disrupt`, plus:
      n_samples     : transforms averaged per step (the "expectation"); cost is
                      roughly steps × n_samples forward/backward passes
                      (+1 no-grad forward each when paired_target=True).
      sampler       : function(rng) -> differentiable transform (make_sampler).
      paired_target : score against G(t(x)) instead of G(x)   — see module doc.
      lowres        : side length of the low-resolution δ grid, or None.
      loss          : "mse" or "log".
    The defaults reproduce v1 exactly.
    """
    if loss not in ("mse", "log"):
        raise ValueError(f"unknown loss {loss!r}")
    if step_size is None:
        step_size = eps / 10
    rng = random.Random(seed)
    torch.manual_seed(seed)

    model.eval()
    x = x.detach()
    h, w = x.shape[-2:]
    with torch.no_grad():
        target = model(x).detach()  # the normal deepfake we want to move away from

    if lowres is None:
        grid = (h, w)
    else:
        grid = (lowres, max(1, round(lowres * w / h)))
    shape = (*x.shape[:2], *grid)

    delta = torch.empty(shape, device=x.device, dtype=x.dtype).uniform_(-eps, eps) \
        if random_start else torch.zeros(shape, device=x.device, dtype=x.dtype)
    if lowres is None:
        delta = (x + delta).clamp(0, 1) - x  # v1 behaviour: keep x+δ inside [0, 1]
    delta.requires_grad_(True)

    def protected(d):
        if lowres is None:
            return x + d  # v1: δ is already kept feasible after every step
        return (x + _upsample(d, (h, w))).clamp(0, 1)

    for _ in range(steps):
        grad = torch.zeros_like(delta)
        for _ in range(n_samples):
            t = sampler(rng)
            out = model(t(protected(delta)))
            if paired_target:
                with torch.no_grad():
                    ref = model(t(x))
            else:
                ref = target
            m = F.mse_loss(out, ref)
            obj = torch.log(m + 1e-8) if loss == "log" else m
            g, = torch.autograd.grad(obj, delta)
            grad += g
        with torch.no_grad():
            delta += step_size * grad.sign()
            delta.clamp_(-eps, eps)
            if lowres is None:
                delta.copy_((x + delta).clamp(0, 1) - x)

    return protected(delta).detach()
