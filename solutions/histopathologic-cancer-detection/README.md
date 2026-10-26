# histopathologic-cancer-detection — 🥇 gold — 0.99585 (ROC AUC)

- **Kaggle competition:** https://www.kaggle.com/c/histopathologic-cancer-detection
- **Best verified result:** 🥇 gold — **0.99585 (ROC AUC)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** `image_finetune` (Impulse proprietary operator family; source not published)

## Approach

End-to-end fine-tuning of a torchvision backbone on the histopathology patches, which the agent runs as a cloud GPU job on a single T4.

## Evidence in this directory

- Files from the engine's development run of 7 July 2026: `grade.json`, `result.json`, `submission.csv`.
- The three confirmation runs below are recorded, with their submission hashes, in the [results ledger](../../results/lite22-three-run.json).

## Three-run confirmation

| Run | Medal | Score |
|---|---|---:|
| 1 | gold | 0.99585 |
| 2 | gold | 0.99578 |
| 3 | gold | 0.99580 |

**Best verified result:** gold, `0.99585`.
**Method:** image fine-tuning on the public data.
**Best-grade evidence SHA-256:** `33febfd66c506da34a2492347210d67227013968e79a95257984307a8b710d83`.

## Verify it yourself

The checked-in [results ledger](../../results/lite22-three-run.json) carries the confirmation records and hashes. Every published submission can be independently re-graded with OpenAI's unmodified `mlebench grade-sample` tooling — no Impulse code required. The [evidence index](../../reproduce/EVIDENCE.md) and [full runbook](../../reproduce/VERIFY.md) cover the remaining checks.


---
© 2026 Impulse AI. All rights reserved.
