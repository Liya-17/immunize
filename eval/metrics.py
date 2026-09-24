"""Evaluation metrics.

Invisibility (clean vs protected input): higher is better -> psnr, ssim
Disruption (model output on clean vs protected): higher is better -> output_l2
All tensors are N x C x H x W with values in [0, 1].
"""

import numpy as np
import torch
from skimage.metrics import structural_similarity


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2).item()
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    scores = []
    for x, y in zip(a, b):
        x = x.detach().cpu().permute(1, 2, 0).numpy()
        y = y.detach().cpu().permute(1, 2, 0).numpy()
        scores.append(structural_similarity(x, y, channel_axis=2, data_range=1.0))
    return float(np.mean(scores))


def output_l2(out_clean: torch.Tensor, out_protected: torch.Tensor) -> float:
    """Mean per-image L2 distance between the model's two outputs."""
    diff = (out_clean - out_protected).flatten(1)
    return diff.norm(dim=1).mean().item()
