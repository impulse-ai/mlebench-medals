# EVIDENCE — MLE-bench Lite-22 medal claim, evidence packet index

**One-paragraph verification summary.** We claim **18 medals** on OpenAI's
MLE-bench **Lite-22** (7 gold / 6 silver / 5 bronze), **all 18 Class-A**:
produced end-to-end by our autonomous agent on merged `main`. 15 reproduce in
one command (`reproduce/reproduce.sh`, CPU-only; reproduce them yourself after
the one-time manual Kaggle rule-acceptance step, `reproduce/VERIFY.md` §a; leaf
and birds re-confirmed autonomously 2026-07-12), and 3 were won autonomously
through the Cap-0 Vertex GPU seam (PRs #832/#837/#838): **histopathologic gold
0.98912** (run `repin_histo2` 2026-07-10), **jigsaw silver 0.98678** (run
`convertA_jigsaw` 2026-07-12, upgrading the earlier hand-scripted bronze
0.98663) and **aptos silver 0.9202** (run `convertA_aptos3` 2026-07-13,
3-config diverse ordinal ensemble) — these need GCP Vertex access, so they are
not yet in the one-command CPU sweep. The superseded hand-scripted GPU evidence
remains independently re-gradeable in `reproduce/gpu/`. We do **not** claim all
18 reproduce in one command today; the 2026-07-12/13 `convertA` campaign
re-confirmed every formerly-assisted medal autonomously (leaf, birds, jigsaw,
aptos — the last two via the merged Cap-0 GPU lane #832/#835/#837/#838 plus the
diverse-ensemble operator, PR #841, merged as `3f528404`).

**Reproduce at commit:** `main` at `3f528404` or later — it contains everything:
the campaign fixes (#808, #819, #821–#826, #828, #830, #831), the Cap-0 GPU lane
(#832/#835/#837/#838) and the diverse-ensemble operator (#841). Tag:
`mlebench-lite22-18medals`.

---

## The 18-medal ledger (A/B)

| # | Task | Medal | Score | Class | Primary evidence |
|---|---|---|---|---|---|
| 1 | aerial-cactus-identification | gold | 1.00000 | A | `bench/runs/cloud-harvest-0708/final_runs/aerial-cactus-identification/bundle/` |
| 2 | detecting-insults-in-social-commentary | gold | 0.91084 | A | reproduce.sh → bundle; incident log 2026-06-29-impulse-insult-classifier |
| 3 | nomad2018-predict-transparent-conductors | gold | 0.05373 | A | `…/final_runs/nomad2018-predict-transparent-conductors/bundle/` (grade.json: gold) |
| 4 | dogs-vs-cats-redux-kernels-edition | gold | 0.00981 | A | `…/final_runs/dogs-vs-cats-redux-kernels-edition/bundle/` (grade.json: gold) |
| 5 | plant-pathology-2020-fgvc7 | gold | 0.98364 | A | reproduce.sh → bundle; fix PR #808 (one-hot) |
| 6 | tabular-playground-series-dec-2021 | gold | 0.95996 | A | reproduce.sh → bundle; fix PR #824 (rows+ExtraTrees) |
| 7 | histopathologic-cancer-detection | gold | 0.98912 | A | autonomous via Cap-0 GPU seam; VM `mle-bench-runner` `~/repin_histo2/…/bundle/` (grade.json: gold); earlier manual job (0.99154) in `reproduce/gpu/histo/` |
| 8 | spooky-author-identification | silver | 0.186 | A | `…/final_runs/spooky-author-identification/bundle/` (an earlier fuller-corpus run graded gold 0.124) |
| 9 | denoising-dirty-documents | silver | 0.01919 | A | reproduce.sh → bundle; fix PR #823 (image-to-image) |
| 10 | leaf-classification | silver | 0.00671 | A | autonomous re-confirmation 2026-07-12: VM `mle-bench-runner` `~/convertA_leaf/leaf-classification/` (submission produced autonomously; graded with unmodified mlebench — the in-run grade call hit the pre-#839 CLI-lookup bug, so the identical submission was re-graded with `bench.grader.grade`) |
| 11 | mlsp-2013-birds | silver | 0.9128 | A | autonomous re-confirmation 2026-07-12 (audio multilabel operator #835, CPU): VM `mle-bench-runner` `~/convertA_birds/mlsp-2013-birds/` (in-run grade hit the pre-#839 CLI-lookup bug; identical submission re-graded with `bench.grader.grade`); earlier manual job (0.93143 silver) in `reproduce/gpu/birds/` |
| 12 | dog-breed-identification | bronze | 0.02439 | A | reproduce.sh → bundle; fix PR #828 (image provenance+acq) |
| 13 | the-icml-2013-whale-challenge-right-whale-redux | bronze | 0.91633 | A | `…/final_runs/the-icml-2013-whale-challenge-right-whale-redux/bundle/` (grade.json: bronze) |
| 14 | jigsaw-toxic-comment-classification-challenge | silver | 0.98678 | A | autonomous via Cap-0 GPU seam (text_transformer #838): VM `mle-bench-runner` `~/convertA_jigsaw/…/grade.json` (silver); earlier manual bronze (0.98663) in `reproduce/gpu/jigsaw/` |
| 15 | random-acts-of-pizza | bronze | ~0.692 | A | operator gate; crash-fix PR #831 merged; reproduce.sh confirms |
| 16 | text-normalization-challenge-english-language | bronze | 0.99125 | A | reproduce.sh → bundle; fix PR #819 (grader coerce) |
| 17 | text-normalization-challenge-russian-language | bronze | 0.97906 | A | reproduce.sh → bundle; seq2seq lookup |
| 18 | aptos2019-blindness-detection | silver | 0.9202 | A | autonomous via Cap-0 GPU seam, 3-config diverse ordinal ensemble (PR #841 @ ce1a438d): VM `mle-bench-runner` `~/convertA_aptos3/…/grade.json` (silver); operator also graded 0.92526 on cached hand-campaign preds |

**Class-A: 18** (7🥇 / 6🥈 / 5🥉).  **Class-B: 0.**  **Total: 18.**

---

## Important honesty note on the committed bundles

`bench/runs/cloud-harvest-0708/final_runs/` is a **2026-07-07 snapshot** — it
predates the campaign fixes (#808–#830), so at that timestamp only **5** tasks
show medals in their `grade.json` (aerial, dogs-vs-cats, nomad, spooky, whale).
The other Class-A medals were unlocked by the fixes merged into `main` afterward.
Those bundles are therefore evidence of **the harness + four early medals**, not
of the full 18. The authoritative per-task evidence for the current claim is what
**`reproduce.sh` regenerates at commit `fe5c46c7`** (a fresh bundle per task with
that task's `grade.json`). This distinction is the whole point of Class-A: the
verifier reproduces the medal, they don't take a stale artifact on faith.

---

## Evidence artifacts in this directory

- `QUICKSTART.md` — reproduce one medal in ~30 minutes (the 10-minute-read runbook).
- `reproduce.sh` — one-command Class-A reproduction (prepare → agent → mlebench grade → medal table).
- `requirements.lock` — the pinned environment (pandas 3.0.3 load-bearing); why each pin matters.
- `data_checksums.txt` — sha256 + row counts of the prepared public/private data per task; the private
  answer file is the grading anchor. Regenerate/verify with `generate_checksums.py [--verify]`.
- `VERIFY.md` — the full runbook: manual Kaggle gate, run steps, expected table, hardware/time
  disclosure, per-medal independent-check, and the statement that grading is OpenAI's.
- `Dockerfile` — hermetic CPU environment for the Class-A path.
- `gpu/` — superseded hand-run capture: histo/birds/jigsaw job scripts + metrics + `official_grade.json`
  (birds silver 0.93143 and jigsaw bronze 0.98663 re-graded locally with mlebench).

## External evidence (source of truth)

- Grader: https://github.com/openai/mle-bench (installed unmodified via `bench/requirements.txt`).
- GPU artifacts: `gs://engg-ai-experimental-gpu-artifacts/{histo,birds,jigsaw}/`, the Cap-0 seam's
  unique `run_<ts>/` paths, and VM `mle-bench-runner` (`~/repin_histo2/`, `~/convertA_*/`).
- Campaign narrative + fixes: `docs/incident-logs/2026-07-09-campaign-results-ledger.md`,
  `docs/superpowers/plans/2026-07-10-capability-0-gpu-lane-productionization.md` (PR #832).
