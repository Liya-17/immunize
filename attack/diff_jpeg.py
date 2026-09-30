"""Phase 2 · Differentiable JPEG (task 2.3).

Real JPEG contains a rounding step (quantisation of DCT coefficients) whose
gradient is zero almost everywhere, so PGD cannot "see through" it. This module
re-implements the JPEG pipeline in PyTorch with a differentiable rounding, so
EoT-PGD can optimise a perturbation that survives compression:

    RGB -> YCbCr -> 4:2:0 chroma subsampling -> 8x8 blocks -> DCT
        -> divide by quality-scaled quantisation table -> ROUND -> multiply
        -> inverse DCT -> upsample chroma -> RGB

Forward pass is (very close to) real libjpeg output; backward pass uses either
  * "ste"   : straight-through estimator — forward exact round, gradient 1
              (BPDA, Athalye et al. 2018). Default.
  * "cubic" : round(x) + (x - round(x))^3, Shin & Song 2017.

`tests/test_diff_jpeg.py` checks the forward pass against PIL's real encoder.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# Standard JPEG (ITU-T T.81 Annex K) quantisation tables for quality 50.
_Q_LUMA = torch.tensor([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=torch.float32)

_Q_CHROMA = torch.tensor([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=torch.float32)


def _dct_matrix(n: int = 8) -> torch.Tensor:
    """Orthonormal DCT-II matrix; C @ B @ C.T equals JPEG's forward DCT."""
    c = torch.zeros(n, n)
    for k in range(n):
        for i in range(n):
            a = math.sqrt(1 / n) if k == 0 else math.sqrt(2 / n)
            c[k, i] = a * math.cos(math.pi * (2 * i + 1) * k / (2 * n))
    return c


_DCT = _dct_matrix()


def quality_to_tables(quality: float, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """libjpeg's quality scaling (same formula PIL uses)."""
    q = float(min(max(quality, 1), 100))
    s = 5000 / q if q < 50 else 200 - 2 * q
    tables = []
    for base in (_Q_LUMA, _Q_CHROMA):
        t = torch.floor((base * s + 50) / 100).clamp(1, 255)
        tables.append(t.to(device))
    return tables[0], tables[1]


def _round(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "ste":
        return x + (torch.round(x) - x).detach()
    if mode == "cubic":
        r = torch.round(x)
        return r + (x - r) ** 3
    raise ValueError(f"unknown rounding mode {mode!r}")


def _rgb_to_ycbcr(x: torch.Tensor) -> torch.Tensor:  # x in 0..255
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128
    return torch.stack([y, cb, cr], 1)


def _ycbcr_to_rgb(x: torch.Tensor) -> torch.Tensor:
    y, cb, cr = x[:, 0], x[:, 1] - 128, x[:, 2] - 128
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.stack([r, g, b], 1)


def _blocks_quantize(ch: torch.Tensor, qtable: torch.Tensor, mode: str) -> torch.Tensor:
    """ch: N x H x W (H, W multiples of 8), values 0..255. Returns decoded channel."""
    n, h, w = ch.shape
    d = _DCT.to(ch.device, ch.dtype)
    blk = (ch - 128).view(n, h // 8, 8, w // 8, 8).permute(0, 1, 3, 2, 4)  # N,hb,wb,8,8
    coef = d @ blk @ d.T
    coef = _round(coef / qtable, mode) * qtable
    blk = d.T @ coef @ d
    return blk.permute(0, 1, 3, 2, 4).reshape(n, h, w) + 128


def diff_jpeg(x: torch.Tensor, quality: float = 75, subsampling: bool = True,
              rounding: str = "ste") -> torch.Tensor:
    """Differentiable JPEG compress + decompress. x: N x 3 x H x W in [0, 1]."""
    n, _, h, w = x.shape
    m = 16 if subsampling else 8
    ph, pw = (-h) % m, (-w) % m
    img = F.pad(x * 255, (0, pw, 0, ph), mode="replicate")
    ycc = _rgb_to_ycbcr(img)
    ql, qc = quality_to_tables(quality, x.device)
    ql, qc = ql.to(x.dtype), qc.to(x.dtype)

    y = _blocks_quantize(ycc[:, 0], ql, rounding)
    chroma = ycc[:, 1:]
    if subsampling:  # 4:2:0 — average 2x2, code at half resolution, upsample back
        small = F.avg_pool2d(chroma, 2)
        sh, sw = small.shape[-2:]
        small = F.pad(small, (0, (-sw) % 8, 0, (-sh) % 8), mode="replicate")
        dec = torch.stack([_blocks_quantize(small[:, i], qc, rounding) for i in range(2)], 1)
        dec = dec[..., :sh, :sw]
        chroma = F.interpolate(dec, scale_factor=2, mode="bilinear", align_corners=False)
    else:
        chroma = torch.stack([_blocks_quantize(chroma[:, i], qc, rounding) for i in range(2)], 1)

    rgb = _ycbcr_to_rgb(torch.cat([y.unsqueeze(1), chroma], 1))
    rgb = rgb[..., :h, :w] / 255
    # decoder clamps to 0..255 — keep gradient flowing through the clamp (STE)
    return rgb + (rgb.clamp(0, 1) - rgb).detach()
