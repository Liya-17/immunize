"""Phase 2 · Real-world transforms: does the protection survive sharing?

When a protected photo is posted on WhatsApp / Instagram it is saved as an
8-bit file, JPEG-compressed and often resized before a deepfaker downloads it.
These functions reproduce that processing with the *real* (non-differentiable)
operations — PIL's libjpeg encoder, antialiased resizing, Gaussian blur — so
they are what we EVALUATE with. The differentiable copies used during the
attack live in `attack/diff_jpeg.py` and `attack/eot.py`.

All tensors are N x C x H x W, float in [0, 1], on any device.

How protection is scored after a transform t (see `survival_table`):
    reference = model(t(x_clean))        # the fake an attacker gets from an
                                         # UNprotected photo shared the same way
    attacked  = model(t(x_protected))    # the fake from the protected photo
    output_mse = mean((reference - attacked)^2)
    success    = output_mse >= MSE_SUCCESS
Comparing against model(t(x_clean)) — not model(x_clean) — means compression's
own small effect on the fake is not counted as "protection".
"""

from __future__ import annotations

import io
from typing import Callable, Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MSE_SUCCESS = 0.05  # same threshold as the Review-1 table


# ----------------------------------------------------------------------------
# Single transforms
# ----------------------------------------------------------------------------
def _to_pil(img: torch.Tensor) -> Image.Image:
    arr = (img.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


def _from_pil(pil: Image.Image, like: torch.Tensor) -> torch.Tensor:
    arr = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).to(like.device, like.dtype)


def quantize_8bit(x: torch.Tensor) -> torch.Tensor:
    """Saving as PNG: values rounded to 0..255. Every protected photo gets this first."""
    return (x.clamp(0, 1) * 255).round() / 255


def jpeg(x: torch.Tensor, quality: int = 75, subsampling: int = 2) -> torch.Tensor:
    """Real JPEG encode + decode with PIL/libjpeg.

    subsampling: 0 = 4:4:4, 1 = 4:2:2, 2 = 4:2:0 (what social media apps use).
    """
    out = []
    for img in x:
        buf = io.BytesIO()
        _to_pil(img).save(buf, format="JPEG", quality=int(quality), subsampling=subsampling)
        buf.seek(0)
        out.append(_from_pil(Image.open(buf), img))
    return torch.stack(out)


def resize(x: torch.Tensor, scale: float = 0.5) -> torch.Tensor:
    """Platform downscales the photo; the attacker scales it back to the model's size."""
    h, w = x.shape[-2:]
    small = F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False, antialias=True)
    return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False).clamp(0, 1)


def gaussian_blur(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    k = int(2 * round(3 * sigma) + 1)
    t = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    c = x.shape[1]
    x = F.pad(x, (k // 2,) * 4, mode="reflect")
    x = F.conv2d(x, g.view(1, 1, 1, k).repeat(c, 1, 1, 1), groups=c)
    x = F.conv2d(x, g.view(1, 1, k, 1).repeat(c, 1, 1, 1), groups=c)
    return x.clamp(0, 1)


# ----------------------------------------------------------------------------
# Social-media-like chains (approximations — see note in docs/ROADMAP.md 2.7:
# the real WhatsApp/Instagram test is done by actually sending the photos)
# ----------------------------------------------------------------------------
def whatsapp_like(x: torch.Tensor) -> torch.Tensor:
    """Downscale to 75 % + JPEG q=70, 4:2:0 — roughly WhatsApp's 'standard quality'."""
    return jpeg(resize(x, 0.75), quality=70)


def instagram_like(x: torch.Tensor) -> torch.Tensor:
    """JPEG q=80, 4:2:0, light resampling — roughly an Instagram feed upload."""
    return jpeg(resize(x, 0.9), quality=80)


def double_share(x: torch.Tensor) -> torch.Tensor:
    """Photo forwarded twice (compressed, re-compressed). Worst realistic case."""
    return whatsapp_like(whatsapp_like(x))


# Name -> function. Order = column order in the results table.
TRANSFORMS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "none (PNG 8-bit)": lambda x: x,
    "JPEG q=90": lambda x: jpeg(x, 90),
    "JPEG q=75": lambda x: jpeg(x, 75),
    "JPEG q=50": lambda x: jpeg(x, 50),
    "Resize 0.5x": lambda x: resize(x, 0.5),
    "Blur σ=1": lambda x: gaussian_blur(x, 1.0),
    "WhatsApp-like": whatsapp_like,
    "Instagram-like": instagram_like,
    "Forwarded twice": double_share,
}


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
@torch.no_grad()
def survival_rows(model, x_clean: torch.Tensor, x_protected: torch.Tensor,
                  transforms: Dict[str, Callable] | None = None) -> list[dict]:
    """One row per transform for ONE image: output MSE and success after sharing.

    `x_protected` is quantized to 8 bits first (it is saved as a file before sharing).
    The clean photo gets the same quantization so both go through identical steps.
    """
    transforms = transforms or TRANSFORMS
    xc, xp = quantize_8bit(x_clean), quantize_8bit(x_protected)
    rows = []
    for name, t in transforms.items():
        ref = model(t(xc))
        att = model(t(xp))
        mse = torch.mean((ref - att) ** 2).item()
        rows.append({"transform": name, "output_mse": mse, "success": mse >= MSE_SUCCESS})
    return rows
