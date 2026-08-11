# tabular-playground-series-dec-2021 — 🥇 gold — 0.95996 (ROC AUC)

- **Kaggle competition:** https://www.kaggle.com/c/tabular-playground-series-dec-2021
- **Best verified result:** 🥇 gold — **0.95996 (ROC AUC)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** [`bench/ees_core/operators/tabular_portfolio.py`](../../bench/ees_core/operators/tabular_portfolio.py)

## Approach

A typed tabular portfolio operator: it detects column types from the data and runs a portfolio of tree-based candidates (including ExtraTrees, per the PR #824 row-handling fix), selecting on out-of-fold performance. Pure CPU.

## Evidence in this directory

- Snapshot artifacts from the 2026-07-07 `cloud-harvest-0708` run: `grade.json`, `result.json`, `submission.csv`.
- These local files preserve the earlier snapshot. The three-run table below and the checked-in results ledger carry the current evidence.

## Three-run confirmation

| Run | Medal | Score |
|---|---|---:|
| 1 | gold | 0.95885 |
| 2 | gold | 0.95911 |
| 3 | gold | 0.95887 |

**Best verified result:** gold, `0.95996`.
**Method:** public-training outer final.
**Best-grade evidence SHA-256:** `bf88bf41114859a33a79af13a92f4a6cb356414c689e00d7e1a964fa1bcf6c23`.

## Verify it yourself

The checked-in [results ledger](../../results/lite22-three-run.json) carries the confirmation records and hashes. To rerun the agent route, use [`reproduce/reproduce.sh`](../../reproduce/reproduce.sh), then grade its submission with OpenAI's tooling. The [evidence index](../../reproduce/EVIDENCE.md) and [full runbook](../../reproduce/VERIFY.md) cover the remaining checks.

---
© 2026 Impulse AI. All rights reserved.
