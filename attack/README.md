# attack/ — Member 1

Perturbation algorithms.

| File | What |
|---|---|
| `pgd.py` | PGD disruption baseline (Phase 1) |
| `eot.py` | *(Phase 2)* PGD + Expectation-over-Transformation for JPEG/resize/blur robustness |
| `diff_jpeg.py` | *(Phase 2)* differentiable JPEG (4:2:0, IJG quality tables, STE rounding) used inside EoT |
| `ensemble.py` | *(Phase 3)* attack multiple models at once |

Usage:
```python
from attack.pgd import pgd_disrupt
x_protected = pgd_disrupt(model, x, eps=8/255, steps=40)
```

Phase 2 (survives compression):
```python
from attack.eot import eot_pgd_disrupt
x_protected = eot_pgd_disrupt(model, x, eps=8/255, steps=60, n_samples=4)
```
Cost ≈ steps × n_samples model passes (≈ 6× plain PGD at the defaults).
