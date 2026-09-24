"""PGD disruption baseline.

Finds a perturbation delta (||delta||_inf <= eps) that maximises the distance
between the model's output on the clean image and on the protected image.
Works with any differentiable image-to-image model `model(x) -> image`.
"""

import torch
import torch.nn.functional as F


def pgd_disrupt(
    model,
    x: torch.Tensor,
    eps: float = 8 / 255,
    step_size: float | None = None,
    steps: int = 40,
    random_start: bool = True,
) -> torch.Tensor:
    """Return a protected copy of `x` (values in [0, 1], shape N x C x H x W)."""
    if step_size is None:
        step_size = 2.5 * eps / steps

    model.eval()
    x = x.detach()
    with torch.no_grad():
        target = model(x).detach()  # what the model normally produces

    delta = torch.empty_like(x).uniform_(-eps, eps) if random_start else torch.zeros_like(x)
    delta = ((x + delta).clamp(0, 1) - x).requires_grad_(True)

    for _ in range(steps):
        out = model(x + delta)
        loss = F.mse_loss(out, target)  # push output away from the normal output
        grad, = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            delta += step_size * grad.sign()
            delta.clamp_(-eps, eps)
            delta.copy_((x + delta).clamp(0, 1) - x)

    return (x + delta).detach()
