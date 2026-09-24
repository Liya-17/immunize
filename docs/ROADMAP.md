# Roadmap

Every step below should become a **GitHub Issue** with an owner and a milestone.
Tick the boxes here as issues close.

Milestones to create in GitHub (Issues → Milestones): `Review-1`, `Review-2`,
`Review-3`, `Prototype Demo`, `End-Sem`. Set each due date from the course
calendar.

---

## Phase 0 — Repository & team setup (Team Lead) · this week

- [ ] Invite both teammates as collaborators with **Write** access
- [ ] Protect `main` (PR required, 1 approval) — see [`GITHUB_SETUP.md`](GITHUB_SETUP.md)
- [ ] Create labels: `attack`, `models`, `eval`, `app`, `docs`, `bug`, `experiment`
- [ ] Create the five milestones above
- [ ] Create a Project board: **To do / In progress / In review / Done**
- [ ] Fill in names and GitHub usernames in the README team table
- [ ] Everyone clones the repo and runs `python setup_check.py` successfully
- [ ] Agree on one compute platform (local GPU / Colab / Kaggle)

## Phase 1 — Baseline (→ Review-1)

**Goal: one image, one model, proof that the idea works.**

| # | Task | Owner | Folder |
|---|---|---|---|
| 1.1 | Literature review: 5–6 key papers, one-paragraph summary each in `docs/literature.md` | All | `docs/` |
| 1.2 | Choose the target model (public code + weights) and document it | Member 2 | `models/` |
| 1.3 | Load the model and generate a normal fake from a clean photo | Member 2 | `models/` |
| 1.4 | Prepare a small test set (20–50 face images, e.g. CelebA-HQ subset) | Member 2 | `data/` |
| 1.5 | Implement PGD disruption (`attack/pgd.py`) | Member 1 | `attack/` |
| 1.6 | Implement metrics: PSNR, SSIM, output L2 (`eval/metrics.py`) | Member 3 | `eval/` |
| 1.7 | Run baseline at ε = 4, 8, 16 / 255 and record the results table | Member 3 | `eval/` |
| 1.8 | Before/after figure: clean → fake vs protected → broken fake | Member 3 | `docs/` |
| 1.9 | Review-1 slides: problem, why immunization > detection, baseline results, plan | Team Lead | `docs/` |

**Done when:** the perturbed photo is visually identical (PSNR > 35 dB) and the
model's output on it is visibly corrupted.

## Phase 2 — Robustness (→ Review-2)

**Goal: protection that survives real-world photo processing.** This is where
the project's novelty comes from.

| # | Task | Owner |
|---|---|---|
| 2.1 | Measure how much baseline protection survives JPEG (q = 90/75/50), resize, Gaussian blur | Member 3 |
| 2.2 | Add Expectation-over-Transformation (EoT): optimise δ through random JPEG / resize / blur | Member 1 |
| 2.3 | Differentiable JPEG approximation for EoT | Member 1 |
| 2.4 | Add a face-identity metric (ArcFace cosine distance) for disruption | Member 3 |
| 2.5 | Compare baseline vs robust version in one results table | Member 3 |
| 2.6 | Review-2 slides | Team Lead |

## Phase 3 — Transferability & second model (→ Review-3)

| # | Task | Owner |
|---|---|---|
| 3.1 | Add a second target model (e.g. a face-swap model such as SimSwap) | Member 2 |
| 3.2 | Test transfer: perturbation built on model A, evaluated on model B | Member 3 |
| 3.3 | Ensemble attack: optimise against both models together | Member 1 |
| 3.4 | Perceptual constraint (LPIPS) so the perturbation is even less visible | Member 1 |
| 3.5 | Ablation table: ε, number of PGD steps, EoT on/off, ensemble on/off | Member 3 |
| 3.6 | Review-3 slides | Team Lead |

## Phase 4 — Prototype demonstration

| # | Task | Owner |
|---|---|---|
| 4.1 | Gradio app: upload photo → download protected photo | Member 3 |
| 4.2 | Side-by-side panel: face-swap on original vs on protected | Member 3 |
| 4.3 | Speed: protection in under ~30 s per image on a Colab GPU | Member 1 |
| 4.4 | Deploy on Hugging Face Spaces (or record a demo video as fallback) | Member 2 |

## Phase 5 — End-semester

- [ ] Final report in `docs/` (abstract, related work, method, experiments, results, limitations)
- [ ] Clean README with final results table and figures
- [ ] Reproducibility: one command / notebook that regenerates the main table
- [ ] Tag the final version: `git tag v1.0 && git push --tags`
- [ ] Final presentation and demo rehearsal

---

## Weekly routine

- **Monday:** 20-minute team call; move cards on the board; assign the week's issues.
- **During the week:** one branch per issue, open a PR when done.
- **Friday:** merge reviewed PRs into `main`; update the results table.
