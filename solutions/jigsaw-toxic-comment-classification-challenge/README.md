# jigsaw-toxic-comment-classification-challenge — 🥇 gold — 0.98750 (mean column-wise ROC AUC)

- **Kaggle competition:** https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
- **Best verified result:** 🥇 gold — **0.98750 (mean column-wise ROC AUC)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** `text_transformer` (Impulse proprietary operator family; source not published)

## Approach

A GPU transformer fine-tune operator: where word/char TF-IDF plateaus (~0.9805 mean AUC here), it fine-tunes pretrained transformer backbones (distilbert / roberta, sigmoid + BCE for multi-label) on a single T4 GPU and blends the transformers.

## Evidence in this directory

- Files from the engine's development run of 7 July 2026: `result.json`.
- The three confirmation runs below are recorded, with their submission hashes, in the [results ledger](../../results/lite22-three-run.json).

## Three-run confirmation

| Run | Medal | Score |
|---|---|---:|
| 1 | silver | 0.98723 |
| 2 | gold | 0.98750 |
| 3 | silver | 0.98701 |

**Best verified result:** gold, `0.98750`.
**Method:** transformer fine-tuning on the public data.
**Best-grade evidence SHA-256:** `44afed4c88531ccb48a15295512e27b466d15866a8978d9573103168f4fe3a49`.

## Verify it yourself

The checked-in [results ledger](../../results/lite22-three-run.json) carries the confirmation records and hashes. Every published submission can be independently re-graded with OpenAI's unmodified `mlebench grade-sample` tooling — no Impulse code required. The [evidence index](../../reproduce/EVIDENCE.md) and [full runbook](../../reproduce/VERIFY.md) cover the remaining checks.


---
© 2026 Impulse AI. All rights reserved.
