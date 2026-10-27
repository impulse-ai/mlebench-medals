# text-normalization-challenge-english-language — 🥉 bronze — 0.99125 (token-level accuracy)

- **Kaggle competition:** https://www.kaggle.com/c/text-normalization-challenge-english-language
- **Best verified result:** 🥉 bronze — **0.99125 (token-level accuracy)**, graded with
  [OpenAI's MLE-bench grading logic](https://github.com/openai/mle-bench).
- **Solution operator:** `seq2seq_lookup` (Impulse proprietary operator family; source not published)

## Approach

A general seq2seq lookup operator: it detects the (input, output) token-pair structure and the sentence+position id layout, learns an exact-match / most-frequent normalization table from the training pairs, and falls back to identity for unseen tokens.

## Evidence in this directory

- Files from the engine's development run of 7 July 2026: `grade.json`, `result.json`, `submission.csv`.
- The three confirmation runs below are recorded, with their submission hashes, in the [results ledger](../../results/lite22-three-run.json).

## Three-run confirmation

| Run | Medal | Score |
|---|---|---:|
| 1 | bronze | 0.99125 |
| 2 | bronze | 0.99125 |
| 3 | bronze | 0.99125 |

**Best verified result:** bronze, `0.99125`.
**Method:** lookup table learned from the public data; three independent runs.
**Best-grade evidence SHA-256:** `dfef6e2b5d259ff1658639e5f7482e311b32e9e0468a74600cfeff86341144b3`.

## Verify it yourself

The checked-in [results ledger](../../results/lite22-three-run.json) carries the confirmation records and hashes. Every published submission can be independently re-graded with OpenAI's unmodified `mlebench grade-sample` tooling — no Impulse code required. The [evidence index](../../reproduce/EVIDENCE.md) and [full runbook](../../reproduce/VERIFY.md) cover the remaining checks.

---
© 2026 Impulse AI. All rights reserved.
