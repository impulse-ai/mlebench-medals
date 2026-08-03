# aerial-cactus-identification — 🥇 gold — 1.00000 (ROC AUC)

- **Kaggle competition:** https://www.kaggle.com/c/aerial-cactus-identification
- **Medal:** 🥇 gold — official score **1.00000 (ROC AUC)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** [`bench/ees_core/operators/image_aerial.py`](../../bench/ees_core/operators/image_aerial.py)

## Approach

A zip-backed image operator: it streams the aerial photos directly out of the competition zip archives, downsamples them to 32×32 RGB, and fits a small classifier ensemble whose probabilities are validated on a held-out slice of the training rows (ROC AUC). The cactus-vs-terrain signal is strong enough that this pure-CPU pipeline reaches a perfect 1.0 AUC.

## Evidence in this directory

- Snapshot artifacts from the 2026-07-07 `cloud-harvest-0708` run: `grade.json`, `result.json`, `submission.csv`.
- The snapshot `grade.json` in this directory shows the gold medal (score 1.0).

## Verify it yourself

The authoritative per-task evidence is a fresh regeneration: run
[`reproduce/reproduce.sh`](../../reproduce/reproduce.sh) (or follow
[`reproduce/QUICKSTART.md`](../../reproduce/QUICKSTART.md)) and read the
`grade.json` OpenAI's grader emits. See [`reproduce/EVIDENCE.md`](../../reproduce/EVIDENCE.md)
for the full 18-medal ledger and [`reproduce/VERIFY.md`](../../reproduce/VERIFY.md)
for the runbook.

---
© 2026 Impulse AI. All rights reserved.
