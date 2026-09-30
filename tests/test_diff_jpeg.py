"""Sanity checks for Phase 2 code. Run:  python tests/test_diff_jpeg.py

1. Our differentiable JPEG must match the real (PIL/libjpeg) JPEG closely —
   otherwise EoT would be optimising against the wrong compression.
2. Gradients must flow through it (that is the whole point).
3. EoT-PGD must respect the ε budget.
"""

import os
import sys

import numpy as np
import torch
from skimage import data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attack.diff_jpeg import diff_jpeg  # noqa: E402
from attack.eot import eot_pgd_disrupt  # noqa: E402
from eval.robustness import TRANSFORMS, jpeg  # noqa: E402


def _faces():
    a = data.astronaut()[20:276, 140:396]
    c = data.chelsea()[:256, :256]
    return torch.stack([torch.from_numpy(i.copy()).permute(2, 0, 1).float() / 255 for i in (a, c)])


def test_matches_real_jpeg():
    x = _faces()
    for q in (90, 75, 50):
        real, ours = jpeg(x, q), diff_jpeg(x, q)
        gap = torch.mean((real - ours) ** 2).item()
        jpeg_err = torch.mean((real - x) ** 2).item()
        p = 10 * np.log10(1 / gap)
        print(f"  q={q}: PSNR(ours vs real JPEG) = {p:.1f} dB; our error is "
              f"{gap / jpeg_err:.0%} of JPEG's own error")
        assert p > 40, "differentiable JPEG drifted from real JPEG"


def test_gradient_flows():
    x = _faces().requires_grad_(True)
    diff_jpeg(x, 75).sum().backward()
    assert x.grad is not None and x.grad.abs().mean() > 0.5
    print("  gradient OK")


def test_eot_budget_and_transforms():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU(),
                                torch.nn.Conv2d(8, 3, 3, padding=1))
    x = _faces()[:1, :, :64, :64]
    eps = 8 / 255
    xp = eot_pgd_disrupt(model, x, eps=eps, steps=3, n_samples=2)
    assert (xp - x).abs().max() <= eps + 1e-6 and xp.min() >= 0 and xp.max() <= 1
    for name, t in TRANSFORMS.items():
        assert t(xp).shape == xp.shape, name
    print("  EoT budget + all transforms OK")


if __name__ == "__main__":
    test_matches_real_jpeg()
    test_gradient_flows()
    test_eot_budget_and_transforms()
    print("All Phase 2 checks passed.")
