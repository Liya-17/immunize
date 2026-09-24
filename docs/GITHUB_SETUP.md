# GitHub setup checklist (Team Lead)

These are done once, in the GitHub website, by the repository owner.

## 1. Add teammates
**Settings → Collaborators → Add people** → enter each teammate's GitHub
username → role **Write**. They must accept the email invite.

## 2. Protect the `main` branch
**Settings → Branches → Add branch ruleset** (or *Add classic branch protection rule*):
- Branch name pattern: `main`
- ✅ Require a pull request before merging
- ✅ Require approvals: **1**
- ✅ Block force pushes
- Save.

Now nobody (including the lead) can push straight to `main`; everything goes
through a reviewed pull request.

## 3. Labels
**Issues → Labels → New label**:
`attack`, `models`, `eval`, `app`, `docs`, `experiment` (keep the default `bug`).

## 4. Milestones
**Issues → Milestones → New milestone**:
`Review-1`, `Review-2`, `Review-3`, `Prototype Demo`, `End-Sem` — each with its
due date.

## 5. Project board
**Projects tab → New project → Board**. Name it `Immunize`. Columns:
**To do · In progress · In review · Done**. Link it to this repository
(project settings → *Manage access* / *Link a repository*).

## 6. Create issues from the roadmap
For each task in [`ROADMAP.md`](ROADMAP.md): **Issues → New issue → Task**
template → set assignee, label and milestone → add to the project board.

## 7. Repository details
On the repo home page, click ⚙ next to **About**:
- Description: *Proactive protection of face images against face-swap deepfakes using adversarial perturbations*
- Topics: `deepfake`, `adversarial-attacks`, `face-swap`, `pytorch`, `computer-vision`

## 8. Share with evaluators
If your guide / coordinator wants to follow progress, add them as a collaborator
with **Read** access. Commit history, issues and PRs are your evidence of
individual contribution at each review.
