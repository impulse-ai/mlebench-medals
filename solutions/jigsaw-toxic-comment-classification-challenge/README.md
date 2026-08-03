# jigsaw-toxic-comment-classification-challenge — 🥈 silver — 0.98678 (mean column-wise ROC AUC)

- **Kaggle competition:** https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
- **Medal:** 🥈 silver — official score **0.98678 (mean column-wise ROC AUC)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** [`engine/operators/text_transformer.py`](../../engine/operators/text_transformer.py)

## Approach

A GPU transformer fine-tune operator: where word/char TF-IDF plateaus (~0.9805 mean AUC here), it fine-tunes pretrained transformer backbones (distilbert / roberta, sigmoid + BCE for multi-label) on a Vertex AI T4 through the Cap-0 seam and blends the transformers. Won autonomously on 2026-07-12, upgrading an earlier hand-scripted bronze (0.98663).

## Evidence in this directory

- Snapshot artifacts from the 2026-07-07 `cloud-harvest-0708` run: `result.json`.
- The 2026-07-07 snapshot for this task contains no `grade.json`; the authoritative evidence is the autonomous GPU run recorded in `reproduce/EVIDENCE.md`.

## Verify it yourself

The authoritative per-task evidence is a fresh regeneration: run
[`reproduce/agent-run.sh`](../../reproduce/agent-run.sh) (or follow
[`reproduce/QUICKSTART.md`](../../reproduce/QUICKSTART.md)) and read the
`grade.json` OpenAI's grader emits. See [`reproduce/EVIDENCE.md`](../../reproduce/EVIDENCE.md)
for the full 18-medal ledger and [`reproduce/VERIFY.md`](../../reproduce/VERIFY.md)
for the runbook.

GPU lane: the job scripts and the independently re-gradeable `official_grade.json` for this task's GPU runs live in [`reproduce/gpu/jigsaw/`](../../reproduce/gpu/jigsaw/).

---
© 2026 Impulse AI. All rights reserved.
