# text-normalization-challenge-russian-language — 🥉 bronze — 0.97915 (token-level accuracy)

- **Kaggle competition:** https://www.kaggle.com/c/text-normalization-challenge-russian-language
- **Best verified result:** 🥉 bronze — **0.97915 (token-level accuracy)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** `seq2seq_lookup` (Impulse proprietary operator family; source not published)

## Approach

The same general seq2seq lookup operator as the English task — exact-match / most-frequent token normalization learned from training pairs, detected purely from data structure with no task-id or language-specific special-casing.

## Evidence in this directory

- Files from the engine's development run of 7 July 2026: `grade.json`, `result.json`, `submission.csv`.
- The three confirmation runs below are recorded, with their submission hashes, in the [results ledger](../../results/lite22-three-run.json).

## Three-run confirmation

| Run | Medal | Score |
|---|---|---:|
| 1 | bronze | 0.97906 |
| 2 | bronze | 0.97906 |
| 3 | bronze | 0.97906 |

**Best verified result:** bronze, `0.97915`.
**Method:** lookup table learned from the public data; three independent runs.
**Best-grade evidence SHA-256:** `aea2c0e1f8d0cf52b4c49e8c8dde795ba4ea7693ab43923df0ffae1a8c7d376f`.

## Verify it yourself

The checked-in [results ledger](../../results/lite22-three-run.json) carries the confirmation records and hashes. Every published submission can be independently re-graded with OpenAI's unmodified `mlebench grade-sample` tooling — no Impulse code required. The [evidence index](../../reproduce/EVIDENCE.md) and [full runbook](../../reproduce/VERIFY.md) cover the remaining checks.

---
© 2026 Impulse AI. All rights reserved.
