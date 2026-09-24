# models/ — Member 2

Target face-manipulation models and weight loaders. **Weights are never committed.**

## Target model(s)

| Model | Paper / repo | Weights link | Status |
|---|---|---|---|
| *(choose in Phase 1)* | | | |

Each model gets a loader `models/<name>.py` exposing:
```python
def load(device="cuda") -> torch.nn.Module:  # returns model with forward(x) -> image in [0,1]
```
