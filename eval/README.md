# eval/ — Member 3

| Metric | Measures | Good value |
|---|---|---|
| PSNR (clean vs protected input) | invisibility | > 35 dB |
| SSIM (clean vs protected input) | invisibility | > 0.95 |
| Output L2 (model output clean vs protected) | disruption | as high as possible |
| ArcFace identity distance *(Phase 2)* | disruption of identity | high |

Robustness tests (Phase 2): JPEG q=90/75/50, resize 0.5×, Gaussian blur σ=1.
