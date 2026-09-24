# Contributing

## The workflow (every task, every time)

```bash
# 1. Start from the latest main
git checkout main
git pull

# 2. Create a branch for your issue  (folder/short-description)
git checkout -b attack/pgd-baseline

# 3. Work, then commit small, clear steps
git add attack/pgd.py
git commit -m "attack: add PGD disruption baseline (#5)"

# 4. Push your branch
git push -u origin attack/pgd-baseline
```

5. On GitHub, open a **Pull Request** into `main`. Write `Closes #5` in the
   description so the issue closes automatically.
6. One teammate reviews and approves → merge → delete the branch.

## Branch names

`attack/...`, `models/...`, `eval/...`, `app/...`, `docs/...`, `fix/...`

## Commit messages

`<area>: <what changed> (#issue)` — e.g. `eval: add SSIM metric (#7)`

## Rules

- Never push directly to `main`.
- Never commit datasets, model weights (`*.pth`, `*.pt`, `*.ckpt`,
  `*.safetensors`) or result images in bulk. Put download links in the README
  of the relevant folder instead.
- Notebooks: one owner per notebook; clear large outputs before committing.
- Record every experiment's settings (model, ε, steps, seed) with its results.
- If you add a package, add it to `requirements.txt` in the same PR.

## Resolving a merge conflict

```bash
git checkout main && git pull
git checkout your-branch
git merge main          # fix the conflicted files, then:
git add . && git commit
git push
```
