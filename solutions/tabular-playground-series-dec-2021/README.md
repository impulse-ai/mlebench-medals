# tabular-playground-series-dec-2021 — 🥇 gold — 0.95996 (ROC AUC)

- **Kaggle competition:** https://www.kaggle.com/c/tabular-playground-series-dec-2021
- **Best verified result:** 🥇 gold — **0.95996 (ROC AUC)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** `tabular_portfolio` (Impulse proprietary operator family; source not published)

## Approach

A typed tabular portfolio operator: it detects column types from the data and runs a portfolio of tree-based candidates (including ExtraTrees), selecting on out-of-fold performance. Pure CPU.

## Evidence in this directory

- Files from the engine's development run of 7 July 2026: `grade.json`, `result.json`, `submission.csv`.
- The three confirmation runs below are recorded, with their submission hashes, in the [results ledger](../../results/lite22-three-run.json).

## Three-run confirmation

| Run | Medal | Score |
|---|---|---:|
| 1 | gold | 0.95885 |
| 2 | gold | 0.95911 |
| 3 | gold | 0.95887 |

**Best verified result:** gold, `0.95996`.
**Method:** tabular portfolio trained on the public data.
**Best-grade evidence SHA-256:** `bf88bf41114859a33a79af13a92f4a6cb356414c689e00d7e1a964fa1bcf6c23`.

## Verify it yourself

The checked-in [results ledger](../../results/lite22-three-run.json) carries the confirmation records and hashes. Every published submission can be independently re-graded with OpenAI's unmodified `mlebench grade-sample` tooling — no Impulse code required. The [evidence index](../../reproduce/EVIDENCE.md) and [full runbook](../../reproduce/VERIFY.md) cover the remaining checks.

---
© 2026 Impulse AI. All rights reserved.
