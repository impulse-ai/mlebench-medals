# tabular-playground-series-dec-2021 — 🥇 gold — 0.95996 (ROC AUC)

- **Kaggle competition:** https://www.kaggle.com/c/tabular-playground-series-dec-2021
- **Medal:** 🥇 gold — official score **0.95996 (ROC AUC)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** [`bench/ees_core/operators/tabular_portfolio.py`](../../bench/ees_core/operators/tabular_portfolio.py)

## Approach

A typed tabular portfolio operator: it detects column types from the data and runs a portfolio of tree-based candidates (including ExtraTrees, per the PR #824 row-handling fix), selecting on out-of-fold performance. Pure CPU.

## Evidence in this directory

- Snapshot artifacts from the 2026-07-07 `cloud-harvest-0708` run: `grade.json`, `result.json`, `submission.csv`.
- NOTE: the snapshot `grade.json` here predates the campaign fixes and shows no medal (score 0.9554); the authoritative evidence is regeneration via `reproduce/reproduce.sh`.

## Verify it yourself

The authoritative per-task evidence is a fresh regeneration: run
[`reproduce/reproduce.sh`](../../reproduce/reproduce.sh) (or follow
[`reproduce/QUICKSTART.md`](../../reproduce/QUICKSTART.md)) and read the
`grade.json` OpenAI's grader emits. See [`reproduce/EVIDENCE.md`](../../reproduce/EVIDENCE.md)
for the full 18-medal ledger and [`reproduce/VERIFY.md`](../../reproduce/VERIFY.md)
for the runbook.

---
© 2026 Impulse AI. All rights reserved.
