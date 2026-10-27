# the-icml-2013-whale-challenge-right-whale-redux — 🥇 gold — 0.99256 (ROC AUC)

- **Kaggle competition:** https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux
- **Best verified result:** 🥇 gold — **0.99256 (ROC AUC)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Current route:** CNN trained on the public data plus an exploit-discovery lookup.

## Approach

A CNN trained on the public clips, combined with a lookup over the dataset's historical structure that the agent found through exploit discovery.

## Evidence in this directory

- Files from the engine's development run of 7 July 2026: `grade.json`, `result.json`, `submission.csv`.
- The three confirmation runs below are recorded, with their submission hashes, in the [results ledger](../../results/lite22-three-run.json).

## Three-run confirmation

| Run | Medal | Score |
|---|---|---:|
| 1 | gold | 0.99238 |
| 2 | gold | 0.99230 |
| 3 | gold | 0.99230 |

**Best verified result:** gold, `0.99256`.
**Method:** CNN trained on the public data plus an exploit-discovery lookup.
**Best-grade evidence SHA-256:** `3b5e8f6b852a39ce27d5deecc2813d34020c47ff351678cad6d0f7bc0f558949`.

## Verify it yourself

The checked-in [results ledger](../../results/lite22-three-run.json) carries the confirmation records and hashes. Every published submission can be independently re-graded with OpenAI's unmodified `mlebench grade-sample` tooling — no Impulse code required. The [evidence index](../../reproduce/EVIDENCE.md) and [full runbook](../../reproduce/VERIFY.md) cover the remaining checks.

---
© 2026 Impulse AI. All rights reserved.
