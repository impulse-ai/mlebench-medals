# aptos2019-blindness-detection — 🥈 silver — 0.9202 (quadratic weighted kappa)

- **Kaggle competition:** https://www.kaggle.com/c/aptos2019-blindness-detection
- **Medal:** 🥈 silver — official score **0.9202 (quadratic weighted kappa)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** [`engine/operators/image_finetune_ensemble.py`](../../engine/operators/image_finetune_ensemble.py)

## Approach

A 3-config diverse ordinal ensemble: diverse backbone families at different resolutions and seeds, each trained as a regression head with k-fold OOF and blended by OOF-weighted averaging so decorrelated configs dominate. Built for ordinal tasks whose test distribution differs from train; won autonomously via the Vertex GPU lane on 2026-07-13 (run `convertA_aptos3`).

## Evidence in this directory

- Snapshot artifacts from the 2026-07-07 `cloud-harvest-0708` run: `grade.json`, `result.json`, `submission.csv`.
- NOTE: the snapshot `grade.json` here predates the campaign fixes and shows no medal; the authoritative evidence is the autonomous GPU run recorded in `reproduce/EVIDENCE.md`.

## Verify it yourself

The authoritative per-task evidence is a fresh regeneration: run
[`reproduce/agent-run.sh`](../../reproduce/agent-run.sh) (or follow
[`reproduce/QUICKSTART.md`](../../reproduce/QUICKSTART.md)) and read the
`grade.json` OpenAI's grader emits. See [`reproduce/EVIDENCE.md`](../../reproduce/EVIDENCE.md)
for the full 18-medal ledger and [`reproduce/VERIFY.md`](../../reproduce/VERIFY.md)
for the runbook.

---
© 2026 Impulse AI. All rights reserved.
