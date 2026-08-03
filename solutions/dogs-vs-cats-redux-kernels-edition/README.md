# dogs-vs-cats-redux-kernels-edition — 🥇 gold — 0.00981 (log loss)

- **Kaggle competition:** https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition
- **Medal:** 🥇 gold — official score **0.00981 (log loss)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** [`engine/operators/image_embedding.py`](../../engine/operators/image_embedding.py)

## Approach

A pretrained image-embedding operator: images are embedded with a frozen pretrained backbone, a logistic head is fit on top, and backbone/calibration choices are made using out-of-fold predictions only. The cats-vs-dogs prior in the pretrained features is near-perfect, giving a 0.0098 log loss.

## Evidence in this directory

- Snapshot artifacts from the 2026-07-07 `cloud-harvest-0708` run: `grade.json`, `result.json`, `submission.csv`.
- The snapshot `grade.json` in this directory shows the gold medal (score 0.00981).

## Verify it yourself

The authoritative per-task evidence is a fresh regeneration: run
[`reproduce/agent-run.sh`](../../reproduce/agent-run.sh) (or follow
[`reproduce/QUICKSTART.md`](../../reproduce/QUICKSTART.md)) and read the
`grade.json` OpenAI's grader emits. See [`reproduce/EVIDENCE.md`](../../reproduce/EVIDENCE.md)
for the full 18-medal ledger and [`reproduce/VERIFY.md`](../../reproduce/VERIFY.md)
for the runbook.

---
© 2026 Impulse AI. All rights reserved.
