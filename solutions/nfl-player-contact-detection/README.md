# nfl-player-contact-detection — 🥉 bronze — 0.71956

- **Kaggle competition:** https://www.kaggle.com/c/nfl-player-contact-detection
- **Result:** 🥉 bronze — **0.71956**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** `pairwise_interaction` with a `pair_temporal_stack` second stage (Impulse proprietary operator family; source not published)

## Approach

Stage one builds pairwise features from the player-tracking table for every candidate player pair at every time step: position, speed, distance and related tracking columns for both players, their differences, and each player's change from the previous and to the next step. Pairs farther apart than a learned distance cutoff are dropped, which keeps 99.7% of true contacts. Boosted-tree classifiers are trained with grouped k-fold validation, with separate decision thresholds for player–player and player–ground rows.

Stage two is a temporal stacking model along each player pair's timeline, cross-fitted on five folds. It raised out-of-fold Matthews correlation from 0.702 to 0.719.

The final submission is a weighted blend of twenty stage-one models and the second stage, which carries just over half of the weight.

## Evidence in this directory

- `submission.csv`: the graded submission. Its SHA-256 is listed below and in the results ledger.
- `grade.json`: the `mlebench` grade report for that submission (the grader host's file path is omitted).

## Result

**Result:** bronze, `0.71956`.
**Medal lines (gold / silver / bronze):** 0.78167 / 0.72203 / 0.69652; higher is better.
**Method:** Engine run.
**LLM:** GPT-6-Luna.
**Run wall time:** 12.33 hours.
**Graded:** 2026-10-08.
**Submission SHA-256:** `970d3ce25a44e2ecb2d7d111783e14b687b969cad9812b6cd5d56d30c534d6b5`.
**Grade report SHA-256:** `92985c78e5d4c360316931b9d69d3ed5652ba48a8eb61495dc4b9b35b7d9deb6`.

## Verify it yourself

The checked-in [results ledger](../../results/hard15.json) carries this record and its hashes. The submission can be independently re-graded with OpenAI's unmodified `mlebench grade-sample` tooling — no Impulse code required. The [HARD evidence index](../../reproduce/EVIDENCE_HARD.md) lists every HARD result.

---
© 2026 Impulse AI. All rights reserved.
