# aerial-cactus-identification — 🥇 gold — 1.00000 (ROC AUC)

- **Kaggle competition:** https://www.kaggle.com/c/aerial-cactus-identification
- **Best verified result:** 🥇 gold — **1.00000 (ROC AUC)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** `image_aerial` (Impulse proprietary operator family; source not published)

## Approach

A zip-backed image operator: it streams the aerial photos directly out of the competition zip archives, downsamples them to 32×32 RGB, and fits a small classifier ensemble whose probabilities are validated on a held-out slice of the training rows (ROC AUC). The cactus-vs-terrain signal is strong enough that this pure-CPU pipeline reaches a perfect 1.0 AUC.

## Evidence in this directory

- Files from the engine's development run of 7 July 2026: `grade.json`, `result.json`, `submission.csv`.
- The three confirmation runs below are recorded, with their submission hashes, in the [results ledger](../../results/lite22-three-run.json).

## Three-run confirmation

| Run | Medal | Score |
|---|---|---:|
| 1 | gold | 1.00000 |
| 2 | gold | 1.00000 |
| 3 | gold | 1.00000 |

**Best verified result:** gold, `1.00000`.
**Method:** trained on the public data.
**Best-grade evidence SHA-256:** `be68a3a22c5ee339302a9661cfa668e374d6b2e97f45f393121ff53b8860875c`.

## Verify it yourself

The checked-in [results ledger](../../results/lite22-three-run.json) carries the confirmation records and hashes. Every published submission can be independently re-graded with OpenAI's unmodified `mlebench grade-sample` tooling — no Impulse code required. The [evidence index](../../reproduce/EVIDENCE.md) and [full runbook](../../reproduce/VERIFY.md) cover the remaining checks.

---
© 2026 Impulse AI. All rights reserved.
