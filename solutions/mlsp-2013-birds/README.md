# mlsp-2013-birds — 🥈 silver — 0.93170 (ROC AUC)

- **Kaggle competition:** https://www.kaggle.com/c/mlsp-2013-birds
- **Best verified result:** 🥈 silver — **0.93170 (ROC AUC)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Route:** the engine's audio pipeline, run three times independently.

## Approach

The engine's audio pipeline detects the long-format submission contract and fits a multi-label classifier on CPU. Three independent runs of the pipeline each produced the same `0.93170` result.

## Evidence in this directory

- Files from the engine's development run of 7 July 2026: `grade.json`, `result.json`, `submission.csv`.
- The three confirmation runs below are recorded, with their submission hashes, in the [results ledger](../../results/lite22-three-run.json).

## Three-run confirmation

| Run | Medal | Score |
|---|---|---:|
| 1 | silver | 0.93170 |
| 2 | silver | 0.93170 |
| 3 | silver | 0.93170 |

**Best verified result:** silver, `0.93170`.
**Method:** the engine's audio pipeline; three independent runs.
**Best-grade evidence SHA-256:** `a45a528c1c73a16cc8d8d45d4b9ef4b9c37ac6924484d4cab210cee19f90ba7f`.

## Verify it yourself

The checked-in [results ledger](../../results/lite22-three-run.json) carries the confirmation records and hashes. Every published submission can be independently re-graded with OpenAI's unmodified `mlebench grade-sample` tooling — no Impulse code required. The [evidence index](../../reproduce/EVIDENCE.md) and [full runbook](../../reproduce/VERIFY.md) cover the remaining checks.


---
© 2026 Impulse AI. All rights reserved.
