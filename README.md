# Immunize

## Proactive Face-Swap Protection

Immunize is a research project that explores **proactive protection** against
unauthorized face manipulation and face-swap attacks.

Most deepfake work focuses on *detecting* a fake after it has been made. Immunize
works one step earlier: before a photo is shared, we add a small, imperceptible
**adversarial perturbation** to it. The protected photo looks the same to a human,
but when a face-swap / face-editing model uses it as input, the model's output
comes out visibly broken.

> Engineering Capstone Project-1 (23IE4053A) · Dept. of AI & Data Science ·
> K L Deemed to be University · 2026-27 Odd Semester

---

## How it works

```text
 clean photo x ──► [ Immunize: x' = x + δ,  ||δ||∞ ≤ ε ] ──► protected photo x'
                                                               │
                         looks identical to x (high PSNR/SSIM) │
                                                               ▼
                                            face-swap model G(x') ──► distorted output
```

The perturbation δ is found with Projected Gradient Descent (PGD): it maximises
the distance between `G(x)` and `G(x + δ)` while keeping δ within an ε-ball so it
stays invisible.

## Goals

1. **Baseline:** reproduce PGD-based disruption against one public face
   manipulation model.
2. **Robustness:** keep protection working after JPEG compression, resizing and
   blur (what happens when a photo is uploaded to social media).
3. **Transferability:** protect against models the perturbation was not
   optimised on.
4. **Demo:** a simple web app where a user uploads a photo and downloads a
   protected version, with a before/after face-swap comparison.

## Team

| Member | Role | Owns |
|---|---|---|
| Member 1 (Team Lead) | Attack algorithm | `attack/` |
| Member 2 | Models & infrastructure | `models/`, environment, GPU setup |
| Member 3 | Evaluation & demo | `eval/`, `app/` |

*(Replace with names and GitHub usernames.)*

## Project structure

```text
immunize/
├── attack/        # perturbation algorithms (PGD baseline, robust variants)
├── models/        # target face-swap / face-editing models, weight loaders
├── eval/          # metrics: invisibility (PSNR, SSIM) and disruption (L2, identity)
├── app/           # demo web app (Gradio)
├── data/          # local only – see data/README.md (not committed)
├── notebooks/     # experiments, one owner per notebook
├── docs/          # roadmap, GitHub setup guide, review slides, reports
├── .github/       # issue & pull-request templates
├── setup_check.py # run this first to verify your environment
├── requirements.txt
└── CONTRIBUTING.md
```

## Setup

```bash
git clone https://github.com/Liya-17/immunize.git
cd immunize
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
python setup_check.py
```

No local GPU? Use Google Colab or Kaggle:

```python
!git clone https://github.com/Liya-17/immunize.git
%cd immunize
!pip install -r requirements.txt
!python setup_check.py
```

Model weights and datasets are **never committed**. Download links live in
`models/README.md` and `data/README.md`.

## Roadmap

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the full step-by-step plan mapped to
Review-1, Review-2, Review-3, the prototype demonstration and the end-semester
evaluation.

## Results

| Experiment | Target model | ε | PSNR ↑ | SSIM ↑ | Output L2 ↑ | Notes |
|---|---|---|---|---|---|---|
| PGD baseline | – | 8/255 | – | – | – | *to be filled* |

## References

- Ruiz, Bargal, Sclaroff. *Disrupting Deepfakes: Adversarial Attacks Against
  Conditional Image Translation Networks and Facial Manipulation Systems.*
  ECCV Workshops 2020.
- Yeh et al. *Disrupting Image-Translation-Based DeepFake Algorithms with
  Adversarial Attacks.* WACV Workshops 2020.
- Madry et al. *Towards Deep Learning Models Resistant to Adversarial Attacks.*
  ICLR 2018 (PGD).
- Salman et al. *Raising the Cost of Malicious AI-Powered Image Editing
  (PhotoGuard).* ICML 2023.
