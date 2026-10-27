# dogs-vs-cats-redux-kernels-edition — 🥇 gold — 0.00597 (log loss)

- **Kaggle competition:** https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition
- **Best verified result:** 🥇 gold — **0.00597 (log loss)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Current solution operator:** `image_finetune` (Impulse proprietary operator family; source not published)

## Approach

Image fine-tuning on the competition's public training set, run with three independent GPU seeds. Each seed is one confirmation run; the blend derived from them is not counted among the confirmations.

## Evidence in this directory

- Files from the engine's development run of 7 July 2026: `grade.json`, `result.json`, `submission.csv`.
- The three confirmation runs below are recorded, with their submission hashes, in the [results ledger](../../results/lite22-three-run.json).

## Three-run confirmation

| Run | Medal | Score |
|---|---|---:|
| 1 | gold | 0.00920 |
| 2 | gold | 0.00870 |
| 3 | gold | 0.00974 |

**Best verified result:** gold, `0.00597`.
**Method:** image fine-tuning on the public data; three GPU seeds.
**Best-grade evidence SHA-256:** `6005accddf7b756344e89b0eed9333f37ac59bcc43db047493e26b2437cd2ca9`.

## Verify it yourself

The checked-in [results ledger](../../results/lite22-three-run.json) carries the confirmation records and hashes. Every published submission can be independently re-graded with OpenAI's unmodified `mlebench grade-sample` tooling — no Impulse code required. The [evidence index](../../reproduce/EVIDENCE.md) and [full runbook](../../reproduce/VERIFY.md) cover the remaining checks.

---
© 2026 Impulse AI. All rights reserved.
