# predict-volcanic-eruptions-ingv-oe — 🥇 gold — 2,740,004

- **Kaggle competition:** https://www.kaggle.com/c/predict-volcanic-eruptions-ingv-oe
- **Result:** 🥇 gold — **2,740,004**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** `tabular_portfolio` (Impulse proprietary operator family; source not published)

## Approach

Feature engineering plus LightGBM regression. The run matched every segment id to its sensor signal file (4,431 files) and engineered 200 features from the signals, then added 45 pairwise interaction features, for 245 features per segment. LightGBM regressors with an L1 objective were trained with five-fold out-of-fold validation on the 3,987 training segments.

The run trained fifteen LightGBM configurations (tree size, learning rate, sampling and seed) plus one further variant, and the final submission is a weighted blend of eight of them, chosen on out-of-fold predictions. The blend's out-of-fold mean absolute error was 2,912,033.

## Evidence in this directory

- `submission.csv`: the graded submission. Its SHA-256 is listed below and in the results ledger.
- `grade.json`: the `mlebench` grade report for that submission (the grader host's file path is omitted).

## Result

**Result:** gold, `2,740,004`.
**Medal lines (gold / silver / bronze):** 3,971,366 / 4,808,057 / 4,999,330; lower is better.
**Method:** Engine run.
**LLM:** GPT-6-Luna.
**Run wall time:** 6.05 hours.
**Graded:** 2026-10-09.
**Submission SHA-256:** `a961df2957712d376541518fdc8f772bdc9f4587846c7270dbf55e21bfcf3bb9`.
**Grade report SHA-256:** `91b93db10c0c45683e3942d8297d861881025aa9231ac45fedd5ba2e9ee4a464`.

## Verify it yourself

The checked-in [results ledger](../../results/hard15.json) carries this record and its hashes. The submission can be independently re-graded with OpenAI's unmodified `mlebench grade-sample` tooling — no Impulse code required. The [HARD evidence index](../../reproduce/EVIDENCE_HARD.md) lists every HARD result.

---
© 2026 Impulse AI. All rights reserved.
