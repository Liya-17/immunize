# attack/ — Member 1

Perturbation algorithms.

| File | What |
|---|---|
| `pgd.py` | PGD disruption baseline (Phase 1) |
| `eot.py` | *(Phase 2)* PGD + Expectation-over-Transformation for JPEG/resize/blur robustness |
| `ensemble.py` | *(Phase 3)* attack multiple models at once |

Usage:
```python
from attack.pgd import pgd_disrupt
x_protected = pgd_disrupt(model, x, eps=8/255, steps=40)
```
