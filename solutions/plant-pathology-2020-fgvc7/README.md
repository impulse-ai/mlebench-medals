# plant-pathology-2020-fgvc7 — 🥇 gold — 0.98364 (mean column-wise ROC AUC)

- **Kaggle competition:** https://www.kaggle.com/c/plant-pathology-2020-fgvc7
- **Medal:** 🥇 gold — official score **0.98364 (mean column-wise ROC AUC)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** [`engine/operators/image_finetune.py`](../../engine/operators/image_finetune.py), [`engine/operators/image_finetune_ensemble.py`](../../engine/operators/image_finetune_ensemble.py)

## Approach

Full end-to-end fine-tuning of a torchvision backbone — the escalation tier above frozen embeddings — trained on the leaf images with held-out validation. This medal required the one-hot label fix (PR #808) that landed after the 2026-07-07 snapshot.

## Evidence in this directory

- Snapshot artifacts from the 2026-07-07 `cloud-harvest-0708` run: `grade.json`, `result.json`, `submission.csv`.
- NOTE: the snapshot `grade.json` here predates the campaign fixes and shows no medal; the authoritative evidence is regeneration via `reproduce/agent-run.sh` (see the honesty note in `reproduce/EVIDENCE.md`).

## Verify it yourself

The authoritative per-task evidence is a fresh regeneration: run
[`reproduce/agent-run.sh`](../../reproduce/agent-run.sh) (or follow
[`reproduce/QUICKSTART.md`](../../reproduce/QUICKSTART.md)) and read the
`grade.json` OpenAI's grader emits. See [`reproduce/EVIDENCE.md`](../../reproduce/EVIDENCE.md)
for the full 18-medal ledger and [`reproduce/VERIFY.md`](../../reproduce/VERIFY.md)
for the runbook.

---
© 2026 Impulse AI. All rights reserved.
