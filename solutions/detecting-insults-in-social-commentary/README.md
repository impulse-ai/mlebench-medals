# detecting-insults-in-social-commentary — 🥇 gold — 0.91118 (ROC AUC)

- **Kaggle competition:** https://www.kaggle.com/c/detecting-insults-in-social-commentary
- **Best verified result:** 🥇 gold — **0.91118 (ROC AUC)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** `text_tfidf` (Impulse proprietary operator family; source not published)

## Approach

A word+character TF-IDF operator: a `FeatureUnion` of word- and char-n-gram TF-IDF features feeding linear and gradient-boosted classifiers, with k-fold out-of-fold predictions used for candidate selection and calibration. Pure CPU, no task-specific configuration.

## Evidence in this directory

- Files from the engine's development run of 7 July 2026: `result.json`.
- The three confirmation runs below are recorded, with their submission hashes, in the [results ledger](../../results/lite22-three-run.json).

## Three-run confirmation

| Run | Medal | Score |
|---|---|---:|
| 1 | gold | 0.90164 |
| 2 | gold | 0.90219 |
| 3 | gold | 0.90284 |

**Best verified result:** gold, `0.91118`.
**Method:** TF-IDF text model trained on the public data.
**Best-grade evidence SHA-256:** `373764f3675e6117525382015daa8ef5367117ff04899c02db115ebde0a5d181`.

## Verify it yourself

The checked-in [results ledger](../../results/lite22-three-run.json) carries the confirmation records and hashes. Every published submission can be independently re-graded with OpenAI's unmodified `mlebench grade-sample` tooling — no Impulse code required. The [evidence index](../../reproduce/EVIDENCE.md) and [full runbook](../../reproduce/VERIFY.md) cover the remaining checks.

---
© 2026 Impulse AI. All rights reserved.
