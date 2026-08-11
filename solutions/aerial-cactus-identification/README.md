# aerial-cactus-identification — 🥇 gold — 1.00000 (ROC AUC)

- **Kaggle competition:** https://www.kaggle.com/c/aerial-cactus-identification
- **Best verified result:** 🥇 gold — **1.00000 (ROC AUC)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** [`engine/operators/image_aerial.py`](../../engine/operators/image_aerial.py)

## Approach

A zip-backed image operator: it streams the aerial photos directly out of the competition zip archives, downsamples them to 32×32 RGB, and fits a small classifier ensemble whose probabilities are validated on a held-out slice of the training rows (ROC AUC). The cactus-vs-terrain signal is strong enough that this pure-CPU pipeline reaches a perfect 1.0 AUC.

## Evidence in this directory

- Snapshot artifacts from the 2026-07-07 `cloud-harvest-0708` run: `grade.json`, `result.json`, `submission.csv`.
- These local files preserve the earlier snapshot. The three-run table below and the checked-in results ledger carry the current evidence.

## Three-run confirmation

| Run | Medal | Score |
|---|---|---:|
| 1 | gold | 1.00000 |
| 2 | gold | 1.00000 |
| 3 | gold | 1.00000 |

**Best verified result:** gold, `1.00000`.
**Method:** public/current trained route.
**Best-grade evidence SHA-256:** `be68a3a22c5ee339302a9661cfa668e374d6b2e97f45f393121ff53b8860875c`.

## Verify it yourself

The checked-in [results ledger](../../results/lite22-three-run.json) carries the confirmation records and hashes. To rerun the agent route, use [`reproduce/agent-run.sh`](../../reproduce/agent-run.sh), then grade its submission with OpenAI's tooling. The [evidence index](../../reproduce/EVIDENCE.md) and [full runbook](../../reproduce/VERIFY.md) cover the remaining checks.

---
© 2026 Impulse AI. All rights reserved.
